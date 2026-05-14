"""
Paper trading broker — simulates order fills with realistic slippage.
Backed by PortfolioAgent (SQLite). No real money involved.
"""

from __future__ import annotations

import os
import random
import uuid
from typing import Optional

from agents.signal_agent import Signal
from brokers.base_broker import BaseBroker
from monitoring.logger import get_logger

_logger = get_logger("PaperBroker")

_SLIPPAGE_PCT = 0.001  # 0.1% slippage on fills


class PaperBroker(BaseBroker):
    def __init__(self, portfolio_agent):
        self._portfolio = portfolio_agent
        self._orders: dict = {}  # order_id → order_dict

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

    def check_exits(self, live_prices: dict, ta_data: dict = None) -> list:
        """
        Check open positions against live prices for SL/TP hits.
        ta_data: optional dict of {symbol: enriched DataFrame} used for T1 reversal detection.
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
            signal_type = pos["signal_type"]

            exit_price = None
            reason = None

            if signal_type == "BUY":
                if price <= sl:
                    exit_price, reason = sl, "STOP_LOSS"
                elif price >= t2:
                    exit_price, reason = t2, "TARGET_2"
                elif price >= t1:
                    if self._is_reversal_at_t1(symbol, ta_data):
                        exit_price, reason = t1, "TARGET_1_REVERSAL"
                        _logger.info(f"{symbol}: reversal detected at T1 — booking profit @ ₹{t1:.2f}")
                    else:
                        entry = pos["entry_price"]
                        self._portfolio.update_stop_loss(trade_id, entry)
                elif self._should_exit_on_indicators(symbol, pos.get("strategy"), ta_data):
                    exit_price, reason = price, "INDICATOR_EXIT"
                    _logger.info(f"{symbol}: indicator exit — EMA bearish + MACD negative @ ₹{price:.2f}")
            elif signal_type == "SELL":
                if price >= sl:
                    exit_price, reason = sl, "STOP_LOSS"
                elif price <= t2:
                    exit_price, reason = t2, "TARGET_2"

            if exit_price and reason:
                pnl = self._portfolio.close_position(trade_id, exit_price, reason)
                closed.append({"symbol": symbol, "reason": reason, "pnl": pnl})

        return closed

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
        """
        if strategy == "positional":
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

    def close_intraday_eod(self, live_prices: dict) -> list:
        """
        Force-close all open intraday positions at EOD.
        Uses last available live price; falls back to entry_price if quote unavailable.
        """
        positions = self._portfolio.get_open_positions()
        closed = []
        for pos in positions:
            if pos.get("strategy") != "intraday":
                continue
            symbol = pos["symbol"]
            price = live_prices.get(symbol) or pos["entry_price"]
            pnl = self._portfolio.close_position(pos["id"], price, "EOD_CLOSE")
            closed.append({"symbol": symbol, "reason": "EOD_CLOSE", "pnl": pnl, "exit_price": price})
            _logger.info(f"EOD auto-close: {symbol} @ ₹{price:.2f} | P&L ₹{pnl:.2f}")
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
