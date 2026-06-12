"""
Analyses per-strategy win rates and adjusts confidence thresholds automatically.
All state is persisted in data/tuning_state.json — never touches trading_config.yaml.
Delete tuning_state.json to reset to YAML defaults.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytz

from agents.base_agent import BaseAgent

_TUNING_STATE_PATH = Path(__file__).parent.parent / "data" / "tuning_state.json"
_IST = pytz.timezone("Asia/Kolkata")


class AutoTuner(BaseAgent):
    TUNE_MIN_TRADES = 10    # Minimum trades before adjusting a strategy's threshold
    TUNE_HIGH_WR = 65.0     # With positive expectancy AND win rate above this → loosen
    TUNE_STEP_UP = 0.5      # How much to raise threshold per run
    TUNE_STEP_DOWN = 0.3    # How much to lower threshold per run
    THRESHOLD_MIN = 5.0
    THRESHOLD_MAX = 8.5

    def __init__(self):
        super().__init__("AutoTuner")
        _TUNING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def run(self, strategy_stats: Dict[str, dict], ai_context: str = "") -> List[dict]:
        """
        Evaluate strategy performance and adjust thresholds where warranted.
        Returns a list of action dicts describing what changed (empty if nothing changed).
        """
        state = self._load_state()
        actions = []

        for strategy, stats in strategy_stats.items():
            if stats["total_trades"] < self.TUNE_MIN_TRADES:
                self.log_info(
                    f"{strategy}: {stats['total_trades']} trades — insufficient data for tuning "
                    f"(need {self.TUNE_MIN_TRADES})"
                )
                continue

            wr = stats["win_rate"]
            expectancy = stats.get("avg_pnl", 0.0)   # signed ₹/trade = realized expectancy
            avg_r = stats.get("avg_r", 0.0)
            n = stats["total_trades"]
            current = state["threshold_overrides"].get(strategy, None)

            from pathlib import Path as _P
            import yaml
            cfg_path = _P(__file__).parent.parent / "config" / "trading_config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            default = float(cfg["signals"]["min_confidence_for_claude"])
            baseline = current if current is not None else default

            # Tune on expectancy, not win rate alone: a high-win-rate strategy can still
            # be net-negative (few big losers), and a low-win-rate strategy can be highly
            # profitable (fat winners). Win rate only decides how aggressively to loosen.
            if expectancy < 0:
                # Losing money on average → be more selective regardless of win rate
                new_val = round(min(baseline + self.TUNE_STEP_UP, self.THRESHOLD_MAX), 1)
                if new_val != baseline:
                    reason = (f"Negative expectancy ₹{expectancy}/trade "
                              f"(WR {wr}%, {avg_r}R avg, {n} trades)")
                    actions.append(self._record_action(state, strategy, baseline, new_val, reason))
                else:
                    self.log_info(f"{strategy}: negative expectancy but threshold already at cap")
            elif expectancy > 0 and wr > self.TUNE_HIGH_WR:
                # Profitable and hitting frequently → loosen to capture more signals
                new_val = round(max(baseline - self.TUNE_STEP_DOWN, self.THRESHOLD_MIN), 1)
                if new_val != baseline:
                    reason = (f"Profitable ₹{expectancy}/trade at {wr}% WR > {self.TUNE_HIGH_WR}% "
                              f"({avg_r}R avg, {n} trades)")
                    actions.append(self._record_action(state, strategy, baseline, new_val, reason))
                else:
                    self.log_info(f"{strategy}: profitable but threshold already at floor")
            else:
                self.log_info(
                    f"{strategy}: expectancy ₹{expectancy}/trade, WR {wr}% — no threshold change"
                )

        state["ai_context"] = ai_context
        state["last_updated"] = datetime.now(_IST).isoformat()
        self._save_state(state)

        if actions:
            self.log_info(f"AutoTuner applied {len(actions)} adjustment(s)")
        else:
            self.log_info("AutoTuner: no threshold changes needed this run")

        return actions

    def get_threshold_overrides(self) -> Dict[str, float]:
        """Return current per-strategy threshold overrides."""
        return self._load_state().get("threshold_overrides", {})

    # ── Private ────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if not _TUNING_STATE_PATH.exists():
            return {"threshold_overrides": {}, "ai_context": "", "history": [], "last_updated": ""}
        try:
            with open(_TUNING_STATE_PATH) as f:
                return json.load(f)
        except Exception as e:
            self.log_error(f"Could not read tuning_state.json: {e}")
            return {"threshold_overrides": {}, "ai_context": "", "history": [], "last_updated": ""}

    def _save_state(self, state: dict) -> None:
        try:
            with open(_TUNING_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log_error(f"Could not write tuning_state.json: {e}")

    def _record_action(self, state: dict, strategy: str, before: float, after: float, reason: str) -> dict:
        direction = "Raised" if after > before else "Lowered"
        action_str = f"{direction} {strategy} threshold {before} → {after}"
        self.log_info(f"{action_str} | {reason}")

        state["threshold_overrides"][strategy] = after
        entry = {
            "date": datetime.now(_IST).strftime("%Y-%m-%d %H:%M"),
            "strategy": strategy,
            "action": action_str,
            "reason": reason,
            "before": before,
            "after": after,
        }
        state.setdefault("history", []).append(entry)
        return entry
