"""
Risk management: position sizing, portfolio exposure checks, daily loss limits.
Approves or rejects signals based on configured risk parameters.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
import yaml
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.signal_agent import Signal
from strategies.positional import get_sector


class RiskAgent(BaseAgent):
    def __init__(self, portfolio_agent=None):
        super().__init__("RiskAgent")
        risk_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
        with open(risk_path) as f:
            self._cfg = yaml.safe_load(f)
        self._portfolio_agent = portfolio_agent
        # Per-scan context (set in run()); safe defaults for standalone calls
        self._regime = "TRENDING"
        self._strategy_stats: dict = {}
        self._raw_data: dict = {}
        self._live_prices: dict = {}
        # Signals approved earlier in the CURRENT scan — the DB has no positions for
        # them yet, so sector-exposure and correlation caps must count them explicitly.
        self._scan_approved: List[Tuple[Signal, int]] = []

    def run(self, signals: List[Signal], context: dict = None) -> List[Tuple[Signal, int]]:
        """
        Filter and size each signal.
        Returns list of (signal, quantity) for approved signals.
        context: optional dict with 'earnings_blackout_symbols', 'expiry_context',
        'regime', 'strategy_stats', 'raw_data', and optionally 'live_prices'.
        """
        context = context or {}
        self._regime = context.get("regime", "TRENDING")
        self._strategy_stats = context.get("strategy_stats", {}) or {}
        self._raw_data = context.get("raw_data", {}) or {}
        # Mark open positions to market for the kill-switch checks below
        self._live_prices = context.get("live_prices") or self._get_open_live_prices()

        if not self._trading_enabled():
            self.log_warning("TRADING_ENABLED=false — all signals blocked")
            return []

        if self._daily_loss_exceeded():
            self.log_warning("Daily loss limit reached (incl. open positions) — blocking all signals")
            return []

        if self._drawdown_exceeded():
            self.log_warning("Max drawdown exceeded (incl. open positions) — blocking all signals")
            return []

        if self._daily_trade_limit_reached():
            self.log_warning("Daily trade limit reached — blocking all signals")
            return []

        open_positions = self._get_open_position_count()
        max_pos = self._cfg["portfolio_limits"]["max_open_positions"]
        # In a VOLATILE regime, tighten the simultaneous-position cap
        if self._regime == "VOLATILE":
            vol_cap = self._cfg.get("regime_risk", {}).get("volatile_max_positions", max_pos)
            if vol_cap < max_pos:
                self.log_info(f"VOLATILE regime — position cap tightened {max_pos}→{vol_cap}")
                max_pos = vol_cap

        approved = []
        approved_symbols: set = set()  # track in-scan approvals — DB not updated until execute step
        self._scan_approved = []       # reset per scan; read by sector/correlation checks
        for signal in signals:
            if open_positions >= max_pos:
                self.log_info(f"Max positions ({max_pos}) reached, skipping {signal.symbol}")
                break

            if signal.symbol in approved_symbols:
                self.log_info(f"REJECTED {signal.symbol}: already approved in this scan", symbol=signal.symbol)
                continue

            ok, reason = self._check_signal(signal, open_positions, context)
            if ok:
                qty = self._calculate_quantity(signal)
                if qty >= 1:
                    approved.append((signal, qty))
                    approved_symbols.add(signal.symbol)
                    self._scan_approved.append((signal, qty))
                    open_positions += 1
                    self.log_info(f"APPROVED {signal.symbol} qty={qty} conf={signal.confidence}", symbol=signal.symbol)
            else:
                self.log_info(f"REJECTED {signal.symbol}: {reason}", symbol=signal.symbol)

        return approved

    def _check_signal(self, signal: Signal, current_positions: int, context: dict = None) -> Tuple[bool, str]:
        """Returns (approved, rejection_reason)."""
        cfg = self._cfg

        # Regime gate — in CRISIS, stop opening new swing/positional risk
        if (self._regime == "CRISIS"
                and cfg.get("regime_risk", {}).get("crisis_block_new", True)
                and signal.strategy in ("swing", "positional")):
            return False, "CRISIS regime — new swing/positional entries blocked"

        # Duplicate position check — one open position per symbol maximum
        if self._portfolio_agent:
            open_syms = [p["symbol"] for p in self._portfolio_agent.get_open_positions()]
            if signal.symbol in open_syms:
                return False, f"Already have open position in {signal.symbol}"

        # Earnings blackout — hard block swing/positional within 3 trading days of board meeting
        blackout_syms = (context or {}).get("earnings_blackout_symbols", [])
        if signal.symbol in blackout_syms and signal.strategy in ("swing", "positional"):
            return False, f"Earnings blackout: board meeting within 3 trading days"

        # F&O expiry gating
        expiry_ctx = (context or {}).get("expiry_context", {})
        expiry_risk = expiry_ctx.get("expiry_risk", "NONE")
        if expiry_risk in ("EXPIRY_WEEK", "EXPIRY_DAY") and signal.strategy == "positional":
            return False, f"Positional blocked: F&O expiry within 48h ({expiry_ctx.get('next_expiry', '?')})"
        if expiry_risk == "EXPIRY_DAY" and signal.strategy == "intraday":
            if self._get_intraday_position_count() >= 1:
                return False, f"Intraday capped at 1 on F&O expiry day"

        # Min confidence check
        min_conf = 7.0  # After Claude validation, should be at least 7
        if signal.confidence < min_conf:
            return False, f"Confidence {signal.confidence} < {min_conf}"

        # Sector concentration cap — prevent the book from loading up on one sector
        ok, reason = self._check_sector_exposure(signal)
        if not ok:
            return False, reason

        # Correlation cap — avoid stacking highly-correlated names (hidden concentration)
        ok, reason = self._check_correlation(signal)
        if not ok:
            return False, reason

        # Stop-loss sanity check
        max_sl_pct = cfg["stop_loss"]["max_loss_pct"]
        if signal.sl_pct > max_sl_pct:
            return False, f"SL {signal.sl_pct}% > max {max_sl_pct}%"

        # Risk-reward minimum
        if signal.risk_reward < 1.0:
            return False, f"RR {signal.risk_reward} < 1.0"

        # Intraday position limit
        if signal.strategy == "intraday":
            intra_limit = cfg["portfolio_limits"]["max_intraday_positions"]
            intra_count = self._get_intraday_position_count()
            if intra_count >= intra_limit:
                return False, f"Max intraday positions ({intra_limit}) reached"

        return True, ""

    def _check_sector_exposure(self, signal: Signal) -> Tuple[bool, str]:
        """Reject a signal if adding it would push one sector past the configured cap
        (% of account equity). Unmapped names (sector OTHER) are not capped."""
        if self._portfolio_agent is None:
            return True, ""
        max_pct = self._cfg["portfolio_limits"].get("max_sector_exposure_pct")
        if not max_pct:
            return True, ""

        sector = get_sector(signal.symbol)
        if sector == "OTHER":
            return True, ""  # grab-bag bucket — don't penalise unmapped symbols

        try:
            open_positions = self._portfolio_agent.get_open_positions()
        except Exception:
            return True, ""

        account_size = self._get_account_size()
        if account_size <= 0:
            return True, ""

        existing_value = sum(
            p["entry_price"] * p["quantity"]
            for p in open_positions if get_sector(p["symbol"]) == sector
        )
        # Include signals already approved in this scan (not yet in the DB)
        existing_value += sum(
            s.entry_price * q
            for s, q in self._scan_approved if get_sector(s.symbol) == sector
        )
        new_value = self._calculate_quantity(signal) * signal.entry_price
        projected_pct = (existing_value + new_value) / account_size * 100

        if projected_pct > max_pct:
            return False, (f"Sector {sector} exposure {projected_pct:.1f}% > max {max_pct}%")
        return True, ""

    def _calculate_quantity(self, signal: Signal) -> int:
        """Risk-based position sizing. Risk-per-trade is edge-aware (Kelly), scaled by
        signal conviction and the current regime, then clamped and capped by position size."""
        account_size = self._get_account_size()
        risk_pct = self._risk_pct_for(signal) / 100
        risk_amount = account_size * risk_pct

        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if sl_distance <= 0:
            return 0

        quantity = int(risk_amount / sl_distance)

        # Cap at max position size
        max_pos_pct = self._cfg["position_sizing"]["max_position_pct"] / 100
        max_value = account_size * max_pos_pct
        max_qty_by_value = int(max_value / signal.entry_price)
        quantity = min(quantity, max_qty_by_value)

        return max(quantity, self._cfg["position_sizing"]["min_quantity"])

    def _risk_pct_for(self, signal: Signal) -> float:
        """Effective per-trade risk % = base (fixed or Kelly) × conviction × regime, clamped."""
        ps = self._cfg["position_sizing"]
        base = float(ps["risk_per_trade_pct"])

        # Edge-aware base via fractional Kelly when enabled and enough history exists
        if ps.get("method") == "kelly":
            kelly_pct = self._kelly_risk_pct(signal.strategy)
            if kelly_pct is not None:
                base = kelly_pct

        risk_pct = base * self._conviction_mult(signal.confidence)

        # Regime de-risking — shrink risk in VOLATILE markets
        if self._regime == "VOLATILE":
            risk_pct *= float(self._cfg.get("regime_risk", {}).get("volatile_risk_multiplier", 1.0))

        lo = float(ps.get("min_risk_per_trade_pct", 0.0))
        hi = float(ps.get("max_risk_per_trade_pct", base))
        return max(lo, min(risk_pct, hi))

    def _kelly_risk_pct(self, strategy: str):
        """Fractional-Kelly risk % from this strategy's realized edge, or None to fall back."""
        ps = self._cfg["position_sizing"]
        stats = self._strategy_stats.get(strategy)
        if not stats or stats.get("total_trades", 0) < ps.get("kelly_min_trades", 20):
            return None
        win_rate = stats.get("win_rate", 0) / 100.0
        avg_win = stats.get("avg_win", 0.0)
        avg_loss = abs(stats.get("avg_loss", 0.0))
        if avg_loss <= 0 or avg_win <= 0:
            return None
        payoff = avg_win / avg_loss
        kelly_f = win_rate - (1 - win_rate) / payoff   # full Kelly fraction of capital
        if kelly_f <= 0:
            return float(ps.get("min_risk_per_trade_pct", 0.5))  # no edge → minimum risk
        return float(ps["kelly_fraction"]) * kelly_f * 100.0

    def _conviction_mult(self, confidence: float) -> float:
        """1.0 at the reference confidence, scaling up to conviction_max_multiplier at max."""
        ps = self._cfg["position_sizing"]
        if not ps.get("conviction_sizing", False):
            return 1.0
        ref = float(ps.get("conviction_ref_confidence", 7.0))
        top = float(ps.get("conviction_max_confidence", 10.0))
        max_mult = float(ps.get("conviction_max_multiplier", 1.0))
        if top <= ref or confidence <= ref:
            return 1.0
        frac = min((confidence - ref) / (top - ref), 1.0)
        return 1.0 + frac * (max_mult - 1.0)

    def _check_correlation(self, signal: Signal) -> Tuple[bool, str]:
        """Reject if the candidate's returns are correlated above the cap with any holding."""
        if self._portfolio_agent is None or not self._raw_data:
            return True, ""
        max_corr = self._cfg["portfolio_limits"].get("max_correlation")
        if not max_corr:
            return True, ""

        lookback = int(self._cfg["portfolio_limits"].get("correlation_lookback_days", 120))
        cand = self._returns_series(signal.symbol, lookback)
        if cand is None:
            return True, ""  # no data for candidate — don't block

        try:
            open_syms = [p["symbol"] for p in self._portfolio_agent.get_open_positions()]
        except Exception:
            return True, ""
        # Include signals already approved in this scan (not yet in the DB) so two
        # highly-correlated names can't both slip through one flat-book scan
        open_syms += [s.symbol for s, _ in self._scan_approved]

        for sym in open_syms:
            other = self._returns_series(sym, lookback)
            if other is None:
                continue
            aligned = cand.align(other, join="inner")
            if len(aligned[0]) < 20:
                continue
            corr = aligned[0].corr(aligned[1])
            if corr is not None and corr > max_corr:
                return False, f"Correlation {corr:.2f} with open {sym} > max {max_corr}"
        return True, ""

    def _returns_series(self, symbol: str, lookback: int):
        """Daily returns for a symbol from raw_data, or None if unavailable."""
        df = self._raw_data.get(symbol)
        if df is None or len(df) < 20:
            return None
        close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if close_col is None:
            return None
        return df[close_col].tail(lookback).pct_change().dropna()

    def _get_open_live_prices(self) -> dict:
        """Fetch live quotes for open-position symbols (for MTM). Best-effort, never raises."""
        if self._portfolio_agent is None:
            return {}
        try:
            syms = list({p["symbol"] for p in self._portfolio_agent.get_open_positions()})
            if not syms:
                return {}
            from agents.data_agent import DataAgent
            da = DataAgent()
            prices = {}
            for s in syms:
                q = da.get_live_quote(s)
                if q:
                    prices[s] = q["last_price"]
            return prices
        except Exception as e:
            self.log_warning(f"MTM price fetch failed: {e} — risk checks use realized P&L only")
            return {}

    def get_position_info(self, signal: Signal) -> dict:
        """Return risk summary for a signal — used in notifications."""
        qty = self._calculate_quantity(signal)
        account_size = self._get_account_size()
        risk_amount = qty * abs(signal.entry_price - signal.stop_loss)
        investment = qty * signal.entry_price
        return {
            "quantity": qty,
            "investment_inr": round(investment, 2),
            "risk_inr": round(risk_amount, 2),
            "risk_pct_of_account": round((risk_amount / account_size) * 100, 2),
            "t1_profit_inr": round(qty * (signal.target_1 - signal.entry_price), 2),
            "t2_profit_inr": round(qty * (signal.target_2 - signal.entry_price), 2),
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _trading_enabled(self) -> bool:
        env = os.getenv("TRADING_ENABLED", "true").lower()
        return env not in ("false", "0", "no")

    def _daily_loss_exceeded(self) -> bool:
        if self._portfolio_agent is None:
            return False
        try:
            stats = self._portfolio_agent.get_daily_stats()
            daily_pnl = stats.get("daily_pnl", 0)
            # Include open-position mark-to-market so the halt fires on unrealized losses too
            unrealized = self._portfolio_agent.get_unrealized_pnl(self._live_prices)
            base = float(os.getenv("ACCOUNT_SIZE", "100000")) or 1.0
            combined_pct = (daily_pnl + unrealized) / base * 100
            max_loss = self._cfg["daily_limits"]["max_daily_loss_pct"]
            return combined_pct < -max_loss
        except Exception:
            return False

    def _daily_trade_limit_reached(self) -> bool:
        if self._portfolio_agent is None:
            return False
        try:
            stats = self._portfolio_agent.get_daily_stats()
            trades_today = stats.get("trades_today", 0)
            max_trades = self._cfg["daily_limits"]["max_daily_trades"]
            return trades_today >= max_trades
        except Exception:
            return False

    def _drawdown_exceeded(self) -> bool:
        if self._portfolio_agent is None:
            return False
        try:
            stats = self._portfolio_agent.get_performance_stats()
            max_dd = self._cfg["drawdown"]["max_drawdown_pct"]
            # Realized drawdown from the closed-trade equity curve
            if abs(stats.get("max_drawdown_pct", 0)) > max_dd:
                return True
            # Live drawdown — open losses can push equity below the prior peak
            unrealized = self._portfolio_agent.get_unrealized_pnl(self._live_prices)
            peak = stats.get("peak_equity") or 0
            if peak > 0:
                effective_equity = stats.get("current_equity", peak) + unrealized
                live_dd = (effective_equity - peak) / peak * 100
                if live_dd < -max_dd:
                    return True
            return False
        except Exception:
            return False

    def _get_open_position_count(self) -> int:
        if self._portfolio_agent is None:
            return 0
        try:
            return len(self._portfolio_agent.get_open_positions())
        except Exception:
            return 0

    def _get_intraday_position_count(self) -> int:
        if self._portfolio_agent is None:
            return 0
        try:
            positions = self._portfolio_agent.get_open_positions()
            return sum(1 for p in positions if p.get("strategy") == "intraday")
        except Exception:
            return 0

    def _get_account_size(self) -> float:
        account_size = float(os.getenv("ACCOUNT_SIZE", "100000"))
        if self._portfolio_agent:
            try:
                stats = self._portfolio_agent.get_performance_stats()
                account_size = stats.get("current_equity", account_size)
            except Exception:
                pass
        return account_size
