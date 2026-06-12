"""
Reads closed trade history from the database and computes per-strategy
and per-symbol performance stats. Also provides the AI context string
injected into signal validation prompts and helpers for reading tuning state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents.base_agent import BaseAgent

_TUNING_STATE_PATH = Path(__file__).parent.parent / "data" / "tuning_state.json"
_MIN_TRADES_STRATEGY = 5   # Minimum closed trades before including a strategy in context
_MIN_TRADES_SYMBOL = 3     # Minimum closed trades before including a symbol in context


class PerformanceTracker(BaseAgent):
    """Computes per-strategy and per-symbol performance metrics from trade history."""

    def __init__(self):
        super().__init__("PerfTracker")

    def compute_strategy_stats(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Return {strategy: {win_rate, total_trades, avg_pnl, avg_r}} for strategies with enough data."""
        if df.empty or "strategy" not in df.columns:
            return {}

        result = {}
        for strategy, group in df.groupby("strategy"):
            if len(group) < _MIN_TRADES_STRATEGY:
                continue
            wins = group[group["pnl"] > 0]
            losses = group[group["pnl"] <= 0]
            win_rate = round(len(wins) / len(group) * 100, 1)
            avg_pnl = round(group["pnl"].mean(), 2)

            avg_r = 0.0
            if "entry_price" in group.columns and "stop_loss" in group.columns and "exit_price" in group.columns:
                valid = group.dropna(subset=["entry_price", "stop_loss", "exit_price"])
                if not valid.empty:
                    risk = (valid["entry_price"] - valid["stop_loss"]).abs()
                    reward = (valid["exit_price"] - valid["entry_price"]).abs()
                    valid_risk = risk[risk > 0]
                    if not valid_risk.empty:
                        avg_r = round((reward[valid_risk.index] / valid_risk).mean(), 2)

            result[strategy] = {
                "win_rate": win_rate,
                "total_trades": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "avg_pnl": avg_pnl,
                "avg_r": avg_r,
                "avg_win": round(wins["pnl"].mean(), 2) if not wins.empty else 0.0,
                "avg_loss": round(losses["pnl"].mean(), 2) if not losses.empty else 0.0,
            }
        return result

    def compute_symbol_stats(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Return {symbol: {win_rate, total_trades, avg_pnl}} for symbols with enough data."""
        if df.empty or "symbol" not in df.columns:
            return {}

        result = {}
        for symbol, group in df.groupby("symbol"):
            if len(group) < _MIN_TRADES_SYMBOL:
                continue
            wins = group[group["pnl"] > 0]
            result[symbol] = {
                "win_rate": round(len(wins) / len(group) * 100, 1),
                "total_trades": len(group),
                "wins": len(wins),
                "avg_pnl": round(group["pnl"].mean(), 2),
            }
        return result

    def build_ai_context(self, strategy_stats: Dict[str, dict], symbol_stats: Dict[str, dict]) -> str:
        """Build the performance context string injected into AI validation prompts."""
        if not strategy_stats and not symbol_stats:
            return ""

        parts = []

        if strategy_stats:
            strat_parts = []
            for strategy, s in sorted(strategy_stats.items()):
                r_note = f", avg {s['avg_r']}R" if s["avg_r"] else ""
                strat_parts.append(
                    f"{strategy}: {s['win_rate']}% WR ({s['total_trades']} trades{r_note})"
                )
            parts.append("Strategy performance — " + ". ".join(strat_parts) + ".")

        if symbol_stats:
            good = [(sym, s) for sym, s in symbol_stats.items() if s["win_rate"] >= 65]
            bad = [(sym, s) for sym, s in symbol_stats.items() if s["win_rate"] < 35]
            notes = []
            for sym, s in sorted(good, key=lambda x: -x[1]["win_rate"])[:3]:
                notes.append(f"{sym}: {s['wins']}/{s['total_trades']} wins ({s['win_rate']}% — strong)")
            for sym, s in sorted(bad, key=lambda x: x[1]["win_rate"])[:3]:
                notes.append(f"{sym}: {s['wins']}/{s['total_trades']} wins ({s['win_rate']}% — under-performing)")
            if notes:
                parts.append("Symbol notes — " + ". ".join(notes) + ".")

        context = " ".join(parts)
        self.log_info(f"Built AI context: {context[:120]}{'...' if len(context) > 120 else ''}")
        return context

    def get_tuning_state(self) -> dict:
        """Read and return the full tuning_state.json (empty defaults if absent)."""
        if not _TUNING_STATE_PATH.exists():
            return {"threshold_overrides": {}, "ai_context": "", "history": []}
        try:
            with open(_TUNING_STATE_PATH) as f:
                return json.load(f)
        except Exception as e:
            self.log_error(f"Could not read tuning_state.json: {e}")
            return {"threshold_overrides": {}, "ai_context": "", "history": []}

    def get_history(self) -> List[dict]:
        return self.get_tuning_state().get("history", [])

    def compute_momentum_rank(self, raw_data: Dict[str, "pd.DataFrame"]) -> Dict[str, float]:
        """12-1 month sector-neutral momentum rank (0.0–1.0) per symbol.

        Symbols with insufficient history (<252 bars) are EXCLUDED from the sector
        sort and assigned a neutral RANK of 0.5 directly — never a momentum value.
        (Putting 0.5 into the momentum dict would be read as a +50% return and rank
        data-poor symbols at the top of their sector.) Rank 0.5 sits below the 0.6
        positional gate, so unproven names are conservatively skipped."""
        from strategies.positional import get_sector

        momentum: Dict[str, float] = {}
        no_data: list = []
        for sym, df in raw_data.items():
            if df is None or len(df) < 252:
                no_data.append(sym)
                continue
            close_col = "Close" if "Close" in df.columns else "close"
            try:
                price_252 = float(df[close_col].iloc[-252])
                price_21 = float(df[close_col].iloc[-21])
                if price_252 > 0:
                    momentum[sym] = (price_21 - price_252) / price_252
                else:
                    no_data.append(sym)
            except Exception:
                no_data.append(sym)

        sector_groups: Dict[str, list] = {}
        for sym in momentum:
            sector = get_sector(sym)
            sector_groups.setdefault(sector, []).append(sym)

        ranks: Dict[str, float] = {}
        for sector, syms in sector_groups.items():
            sorted_syms = sorted(syms, key=lambda s: momentum[s])
            n = len(sorted_syms)
            for i, sym in enumerate(sorted_syms):
                ranks[sym] = round((i + 1) / n, 2)
        for sym in no_data:
            ranks[sym] = 0.5  # neutral rank — insufficient history to claim momentum

        self.log_info(
            f"Momentum ranks computed for {len(ranks)} symbols across "
            f"{len(sector_groups)} sectors ({len(no_data)} with insufficient history)"
        )
        return ranks

    def run(self, *args, **kwargs):
        pass
