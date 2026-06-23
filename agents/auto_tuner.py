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

    # MAE-based stop calibration. Stricter than threshold tuning because changing a
    # stop distance is more consequential: it needs more winners with captured MAE,
    # moves in small steps, and stays inside hard bounds.
    STOP_MIN_WINNERS = 8    # Winners with captured MAE before a stop is recalibrated
    STOP_STEP = 0.25        # Max change to a strategy's ATR multiplier per run
    STOP_MULT_MIN = 1.0
    STOP_MULT_MAX = 3.0

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

    def calibrate_stops(self, trades: List[dict]) -> List[dict]:
        """Recalibrate per-strategy ATR stop multipliers from captured MAE.

        Reads each closed trade's stored mae_price (no network), groups by strategy,
        and—once a strategy has enough winners whose MAE separates from losers'—nudges
        its ATR stop multiplier toward the winners' p90 MAE, step-limited and bounded.
        Overrides persist in tuning_state.json; SignalAgent reads them when sizing stops.
        Returns the list of change actions (empty if nothing changed)."""
        from agents.mae_calibration import (
            default_stop_mult, net_pnl, mae_atr, recommend_stop_multiplier,
        )
        state = self._load_state()
        overrides = state.setdefault("stop_multiplier_overrides", {})
        actions: List[dict] = []

        by_strategy: Dict[str, list] = {}
        for t in trades:
            if (t.get("status") or "").upper() == "OPEN":
                continue
            strat = t.get("strategy")
            if strat:
                by_strategy.setdefault(strat, []).append(t)

        for strategy, ts in by_strategy.items():
            current = float(overrides.get(strategy, default_stop_mult(strategy)))
            win, loss = [], []
            for t in ts:
                m = mae_atr(t, current)
                if m is None:
                    continue
                (win if net_pnl(t) > 0 else loss).append(m)

            target, reason = recommend_stop_multiplier(
                win, loss, current,
                min_winners=self.STOP_MIN_WINNERS,
                mult_min=self.STOP_MULT_MIN, mult_max=self.STOP_MULT_MAX,
            )
            if target is None:
                self.log_info(f"{strategy}: stop not recalibrated — {reason}")
                continue

            # Move gradually toward the target, then clamp to hard bounds.
            if target > current:
                new = round(min(current + self.STOP_STEP, target), 2)
            else:
                new = round(max(current - self.STOP_STEP, target), 2)
            new = round(min(max(new, self.STOP_MULT_MIN), self.STOP_MULT_MAX), 2)
            if abs(new - current) < 0.01:
                continue

            overrides[strategy] = new
            direction = "Widened" if new > current else "Tightened"
            action_str = f"{direction} {strategy} stop {current}× → {new}×ATR"
            full_reason = f"{reason}; target {target}× ({len(win)}W/{len(loss)}L)"
            self.log_info(f"{action_str} | {full_reason}")
            entry = {
                "date": datetime.now(_IST).strftime("%Y-%m-%d %H:%M"),
                "strategy": strategy,
                "action": action_str,
                "reason": full_reason,
                "before": current,
                "after": new,
            }
            state.setdefault("history", []).append(entry)
            actions.append(entry)

        state["last_updated"] = datetime.now(_IST).isoformat()
        self._save_state(state)
        if actions:
            self.log_info(f"AutoTuner recalibrated {len(actions)} stop(s)")
        return actions

    def get_threshold_overrides(self) -> Dict[str, float]:
        """Return current per-strategy threshold overrides."""
        return self._load_state().get("threshold_overrides", {})

    def get_stop_overrides(self) -> Dict[str, float]:
        """Return current per-strategy ATR stop-multiplier overrides."""
        return self._load_state().get("stop_multiplier_overrides", {})

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
