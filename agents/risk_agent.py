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


class RiskAgent(BaseAgent):
    def __init__(self, portfolio_agent=None):
        super().__init__("RiskAgent")
        risk_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
        with open(risk_path) as f:
            self._cfg = yaml.safe_load(f)
        self._portfolio_agent = portfolio_agent

    def run(self, signals: List[Signal], context: dict = None) -> List[Tuple[Signal, int]]:
        """
        Filter and size each signal.
        Returns list of (signal, quantity) for approved signals.
        context: optional dict with keys like 'earnings_blackout_symbols'.
        """
        if not self._trading_enabled():
            self.log_warning("TRADING_ENABLED=false — all signals blocked")
            return []

        if self._daily_loss_exceeded():
            self.log_warning("Daily loss limit reached — blocking all signals")
            return []

        if self._drawdown_exceeded():
            self.log_warning("Max drawdown exceeded — blocking all signals")
            return []

        open_positions = self._get_open_position_count()
        max_pos = self._cfg["portfolio_limits"]["max_open_positions"]

        approved = []
        for signal in signals:
            if open_positions >= max_pos:
                self.log_info(f"Max positions ({max_pos}) reached, skipping {signal.symbol}")
                break

            ok, reason = self._check_signal(signal, open_positions, context)
            if ok:
                qty = self._calculate_quantity(signal)
                if qty >= 1:
                    approved.append((signal, qty))
                    open_positions += 1
                    self.log_info(f"APPROVED {signal.symbol} qty={qty} conf={signal.confidence}", symbol=signal.symbol)
            else:
                self.log_info(f"REJECTED {signal.symbol}: {reason}", symbol=signal.symbol)

        return approved

    def _check_signal(self, signal: Signal, current_positions: int, context: dict = None) -> Tuple[bool, str]:
        """Returns (approved, rejection_reason)."""
        cfg = self._cfg

        # Duplicate position check — one open position per symbol maximum
        if self._portfolio_agent:
            open_syms = [p["symbol"] for p in self._portfolio_agent.get_open_positions()]
            if signal.symbol in open_syms:
                return False, f"Already have open position in {signal.symbol}"

        # Earnings blackout — hard block swing/positional within 3 trading days of board meeting
        blackout_syms = (context or {}).get("earnings_blackout_symbols", [])
        if signal.symbol in blackout_syms and signal.strategy in ("swing", "positional"):
            return False, f"Earnings blackout: board meeting within 3 trading days"

        # Min confidence check
        min_conf = 7.0  # After Claude validation, should be at least 7
        if signal.confidence < min_conf:
            return False, f"Confidence {signal.confidence} < {min_conf}"

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

    def _calculate_quantity(self, signal: Signal) -> int:
        """Fixed fractional position sizing: risk 2% of account per trade."""
        account_size = self._get_account_size()
        risk_pct = self._cfg["position_sizing"]["risk_per_trade_pct"] / 100
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
            daily_pnl_pct = stats.get("daily_pnl_pct", 0)
            max_loss = self._cfg["daily_limits"]["max_daily_loss_pct"]
            return daily_pnl_pct < -max_loss
        except Exception:
            return False

    def _drawdown_exceeded(self) -> bool:
        if self._portfolio_agent is None:
            return False
        try:
            stats = self._portfolio_agent.get_performance_stats()
            drawdown = stats.get("max_drawdown_pct", 0)
            return abs(drawdown) > self._cfg["drawdown"]["max_drawdown_pct"]
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
