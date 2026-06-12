"""
Paper trading broker — simulates order fills with realistic slippage.
Backed by PortfolioAgent (SQLite). No real money involved.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pytz
import yaml

_IST = pytz.timezone("Asia/Kolkata")
_EOD_CLOSE_IST = time(15, 20)  # must match the EOD-close cron in orchestrator/workflow.py

from agents.signal_agent import Signal
from brokers.base_broker import BaseBroker
from monitoring.logger import get_logger

_logger = get_logger("PaperBroker")

_SLIPPAGE_PCT = 0.001  # 0.1% slippage on fills


class PaperBroker(BaseBroker):
    def __init__(self, portfolio_agent):
        self._portfolio = portfolio_agent
        self._orders: dict = {}  # order_id → order_dict
        # Partial-exit scale-out percentages (of original position) from risk config
        self._t1_exit_pct, self._t2_exit_pct = self._load_exit_pcts()
        # Per-strategy time-stop windows (calendar days) from trading config
        self._max_hold = self._load_max_hold()

    @staticmethod
    def _load_max_hold() -> dict:
        try:
            cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
            with open(cfg_path) as f:
                strategies = yaml.safe_load(f).get("strategies", {})
            out = {}
            for name, scfg in strategies.items():
                if isinstance(scfg, dict) and scfg.get("max_hold_days"):
                    out[name] = int(scfg["max_hold_days"])
            return out
        except Exception as e:
            _logger.warning(f"Could not load max_hold_days: {e}")
            return {}

    def _exceeded_max_hold(self, strategy: str, entry_time: str) -> bool:
        """True if a position has been held past its strategy's max-hold window."""
        max_days = self._max_hold.get(strategy)
        if not max_days or not entry_time:
            return False
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            if entry_dt.tzinfo is None:
                entry_dt = _IST.localize(entry_dt)
            return (datetime.now(_IST) - entry_dt).days >= max_days
        except Exception:
            return False

    @staticmethod
    def _load_exit_pcts() -> tuple:
        try:
            cfg_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
            with open(cfg_path) as f:
                targets = yaml.safe_load(f).get("targets", {})
            return (float(targets.get("t1_exit_pct", 50)),
                    float(targets.get("t2_exit_pct", 25)))
        except Exception as e:
            _logger.warning(f"Could not load exit pcts: {e} — defaulting 50/25")
            return 50.0, 25.0

    def execute_signal(self, signal: Signal, quantity: int) -> Optional[str]:
        """High-level: place order for an approved signal and open position."""
        if os.getenv("TRADING_ENABLED", "true").lower() in ("false", "0", "no"):
            _logger.warning("TRADING_ENABLED=false — trade blocked at broker layer")
            return None
        fill_price = self._simulate_fill(signal.entry_price, signal.signal_type)
        order_id = self.place_order(
            trade_id=signal.id,
            symbol=signal.symbol,
            transaction_type=signal.signal_type,
            quantity=quantity,
            price=fill_price,
        )
        if order_id:
            self._portfolio.open_position(
                trade_id=signal.id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                strategy=signal.strategy,
                entry_price=fill_price,
                stop_loss=signal.stop_loss,
                target_1=signal.target_1,
                target_2=signal.target_2,
                quantity=quantity,
                confidence=signal.confidence,
                claude_verdict=signal.claude_verdict or "",
            )
            _logger.info(f"Paper order filled: {signal.symbol} x{quantity} @ ₹{fill_price}")
        return order_id

    def check_exits(self, live_prices: dict, ta_data: dict = None,
                    day_highs: dict = None, day_lows: dict = None) -> list:
        """
        Check open positions against live prices for SL/TP hits.
        ta_data: optional dict of {symbol: enriched DataFrame} used for T1 reversal detection.
        day_highs/day_lows: optional dicts so T2 is detected even if price pulled back since touch.
        Returns list of closed trade dicts.
        """
        positions = self._portfolio.get_open_positions()
        closed = []

        for pos in positions:
            symbol = pos["symbol"]
            price = live_prices.get(symbol)
            if price is None:
                continue

            trade_id = pos["id"]
            sl = pos["stop_loss"]
            t1 = pos["target_1"]
            t2 = pos["target_2"]
            entry = pos["entry_price"]
            signal_type = pos["signal_type"]
            qty = pos["quantity"]
            orig_qty = pos.get("original_quantity") or qty
            t1_booked = bool(pos.get("t1_booked"))
            t2_booked = bool(pos.get("t2_booked"))

            # Use day high/low to catch hits that occurred between monitor ticks
            d_high = (day_highs or {}).get(symbol, price)
            d_low = (day_lows or {}).get(symbol, price)

            exit_price = None
            reason = None
            scaled_out = False  # True if a T1/T2 scale-out acted on this position this tick

            if signal_type == "BUY":
                if d_low <= sl:
                    # Full close: hard stop, breakeven stop (post-T1), or T1-trailed stop (post-T2)
                    exit_price, reason = sl, "STOP_LOSS"
                elif (not t1_booked) and d_high >= t1:
                    if self._is_reversal_at_t1(symbol, ta_data):
                        exit_price, reason = t1, "TARGET_1_REVERSAL"
                        _logger.info(f"{symbol}: reversal detected at T1 — booking full @ ₹{t1:.2f}")
                    else:
                        # Scale out at T1 and trail the stop to breakeven on the remainder
                        self._scale_out(closed, trade_id, symbol, orig_qty, qty, t1,
                                        self._t1_exit_pct, "t1", "TARGET_1_PARTIAL",
                                        new_sl=entry, sl_label="breakeven")
                        scaled_out = True
                elif t1_booked and (not t2_booked) and d_high >= t2:
                    # Scale out at T2 and trail the stop up to T1 — let the rest run
                    self._scale_out(closed, trade_id, symbol, orig_qty, qty, t2,
                                    self._t2_exit_pct, "t2", "TARGET_2_PARTIAL",
                                    new_sl=t1, sl_label="T1")
                    scaled_out = True
                elif self._should_exit_on_indicators(symbol, pos.get("strategy"), ta_data):
                    exit_price, reason = price, "INDICATOR_EXIT"
                    _logger.info(f"{symbol}: indicator exit — EMA bearish + MACD negative @ ₹{price:.2f}")
            elif signal_type == "SELL":
                if d_high >= sl:
                    exit_price, reason = sl, "STOP_LOSS"
                elif (not t1_booked) and d_low <= t1:
                    self._scale_out(closed, trade_id, symbol, orig_qty, qty, t1,
                                    self._t1_exit_pct, "t1", "TARGET_1_PARTIAL",
                                    new_sl=entry, sl_label="breakeven")
                    scaled_out = True
                elif t1_booked and (not t2_booked) and d_low <= t2:
                    self._scale_out(closed, trade_id, symbol, orig_qty, qty, t2,
                                    self._t2_exit_pct, "t2", "TARGET_2_PARTIAL",
                                    new_sl=t1, sl_label="T1")
                    scaled_out = True

            # Time-stop — exit a stalled position that hasn't reached T1 within its window.
            # Trades that already booked T1 are left to ride their trailed stop. The local
            # t1_booked flag is read before the elif chain, so also skip when a scale-out
            # just fired this tick — otherwise a trade tagging T1 on/after its max-hold day
            # would book the partial and then have its runner dumped at market (or, if the
            # scale-out fully closed it, trigger a double close returning pnl=None).
            if (exit_price is None and not scaled_out and not t1_booked
                    and self._exceeded_max_hold(pos.get("strategy"), pos.get("entry_time"))):
                exit_price, reason = price, "TIME_STOP"
                _logger.info(f"{symbol}: time-stop — held past max window without T1, exit @ ₹{price:.2f}")

            if exit_price and reason:
                pnl = self._portfolio.close_position(trade_id, exit_price, reason)
                if pnl is None:
                    _logger.warning(f"{symbol}: {reason} close skipped — position already closed")
                else:
                    closed.append({"symbol": symbol, "reason": reason, "pnl": pnl})

        return closed

    def _scale_out(self, closed: list, trade_id: str, symbol: str, orig_qty: int, qty: int,
                   target_price: float, pct: float, milestone: str, reason: str,
                   new_sl: float, sl_label: str) -> None:
        """
        Book a partial profit at a target and trail the stop on the remainder.
        Falls back to a full close if the scale-out would consume the whole position,
        or just advances state + trails the stop if the position is too small to split.
        """
        exit_qty = int(round(orig_qty * pct / 100.0))

        if exit_qty >= qty:
            # Nothing worth keeping as a runner — close the remainder fully at the target
            full_reason = reason.replace("_PARTIAL", "")
            pnl = self._portfolio.close_position(trade_id, target_price, full_reason)
            closed.append({"symbol": symbol, "reason": full_reason, "pnl": pnl})
            _logger.info(f"{symbol}: {full_reason} — closed remaining {qty} @ ₹{target_price:.2f}")
            return

        if exit_qty < 1:
            # Position too small to split — advance the milestone and trail the stop only
            self._portfolio.partial_close(trade_id, 0, target_price, reason, milestone)
            self._portfolio.update_stop_loss(trade_id, new_sl)
            _logger.info(f"{symbol}: {milestone.upper()} reached (too small to scale) — SL → {sl_label}")
            return

        realized = self._portfolio.partial_close(trade_id, exit_qty, target_price, reason, milestone)
        self._portfolio.update_stop_loss(trade_id, new_sl)
        closed.append({"symbol": symbol, "reason": reason, "pnl": realized})
        _logger.info(
            f"{symbol}: {reason} — booked {exit_qty}/{orig_qty} @ ₹{target_price:.2f}, "
            f"SL trailed to {sl_label} (₹{new_sl:.2f})"
        )

    def _is_reversal_at_t1(self, symbol: str, ta_data: dict) -> bool:
        """
        Returns True if RSI is overbought (>70) OR MACD histogram is declining.
        Either condition alone is enough to book profit at T1.
        """
        if not ta_data or symbol not in ta_data:
            return False
        df = ta_data.get(symbol)
        if df is None or len(df) < 2:
            return False

        row = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = row.get("RSI_14")
        macd_hist = row.get("MACDh_12_26_9")
        prev_macd_hist = prev.get("MACDh_12_26_9")

        rsi_overbought = rsi is not None and rsi > 70
        macd_declining = (
            macd_hist is not None
            and prev_macd_hist is not None
            and macd_hist < prev_macd_hist
        )

        return rsi_overbought or macd_declining

    def _should_exit_on_indicators(self, symbol: str, strategy: str, ta_data: dict) -> bool:
        """
        Detects bearish momentum shift on open BUY positions.
        Requires BOTH conditions to fire — reduces false exits from normal noise:
          1. EMA bearish alignment (EMA9 no longer above EMA21)
          2. MACD histogram negative AND declining
        Skips positional trades — daily candles are too noisy for a weekly strategy.
        Skips mean-reversion trades — they are entered counter-trend (below EMA21,
        oversold), so bearish indicators are the entry condition, not an exit signal.
        """
        if strategy in ("positional", "mean_reversion"):
            return False
        if not ta_data or symbol not in ta_data:
            return False
        df = ta_data.get(symbol)
        if df is None or len(df) < 2:
            return False

        row = df.iloc[-1]
        prev = df.iloc[-2]

        ema_bearish = not row.get("ema_bullish_align", True)

        macd_hist = row.get("MACDh_12_26_9")
        prev_macd_hist = prev.get("MACDh_12_26_9")
        macd_weak = (
            macd_hist is not None
            and prev_macd_hist is not None
            and macd_hist < 0
            and macd_hist < prev_macd_hist
        )

        return ema_bearish and macd_weak

    @staticmethod
    def _missed_eod_close(entry_time: str) -> bool:
        """True if an open intraday position's 3:20 PM EOD close has already passed:
        entered on a previous day, or entered today with IST now past the close time.
        Unparseable entry times return False — never force-close blind."""
        if not entry_time:
            return False
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            if entry_dt.tzinfo is None:
                entry_dt = _IST.localize(entry_dt)
        except (ValueError, TypeError):
            return False
        now = datetime.now(_IST)
        if entry_dt.astimezone(_IST).date() < now.date():
            return True
        return now.time() >= _EOD_CLOSE_IST

    def close_intraday_eod(self, live_prices: dict, stale_only: bool = False) -> list:
        """
        Force-close open intraday positions at EOD.
        Uses last available live price; positions without a quote stay open.

        stale_only: close only positions whose scheduled 3:20 PM close was missed
        (app was down when the cron should have fired). Used by startup recovery;
        the fill is the current quote, not the actual 3:20 price, so these are
        tagged EOD_CLOSE_LATE for the audit trail.
        """
        positions = self._portfolio.get_open_positions()
        closed = []
        for pos in positions:
            if pos.get("strategy") != "intraday":
                continue
            if stale_only and not self._missed_eod_close(pos.get("entry_time")):
                continue
            symbol = pos["symbol"]
            price = live_prices.get(symbol)
            if price is None:
                _logger.error(
                    f"EOD close SKIPPED for {symbol}: no live quote available. "
                    "Position remains open — manual close required."
                )
                continue
            reason = "EOD_CLOSE_LATE" if stale_only else "EOD_CLOSE"
            pnl = self._portfolio.close_position(pos["id"], price, reason)
            closed.append({"symbol": symbol, "reason": reason, "pnl": pnl, "exit_price": price})
            _logger.info(f"EOD auto-close ({reason}): {symbol} @ ₹{price:.2f} | P&L ₹{pnl:.2f}")
        return closed

    # ── BaseBroker interface ────────────────────────────────────────────────

    def place_order(self, trade_id: str, symbol: str, transaction_type: str,
                    quantity: int, price: float, order_type: str = "LIMIT") -> Optional[str]:
        order_id = str(uuid.uuid4())[:8]
        self._orders[order_id] = {
            "order_id": order_id, "trade_id": trade_id, "symbol": symbol,
            "type": transaction_type, "qty": quantity, "price": price,
            "status": "COMPLETE",
        }
        _logger.info(f"[PAPER] {transaction_type} {symbol} x{quantity} @ ₹{price:.2f} [{order_id}]")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> dict:
        return self._orders.get(order_id, {"status": "NOT_FOUND"})

    def get_positions(self) -> list:
        return self._portfolio.get_open_positions()

    def get_funds(self) -> dict:
        stats = self._portfolio.get_performance_stats()
        initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        open_pos = self._portfolio.get_open_positions()
        invested = sum(p["entry_price"] * p["quantity"] for p in open_pos)
        return {
            "available_cash": round(stats.get("current_equity", initial) - invested, 2),
            "total_equity": stats.get("current_equity", initial),
            "invested": round(invested, 2),
        }

    def _simulate_fill(self, price: float, signal_type: str) -> float:
        """Add realistic slippage to fill price."""
        slip = random.uniform(0, _SLIPPAGE_PCT)
        if signal_type == "BUY":
            return round(price * (1 + slip), 2)
        return round(price * (1 - slip), 2)
