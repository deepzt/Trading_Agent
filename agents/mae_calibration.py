"""
Shared Maximum Adverse Excursion (MAE) → stop-multiplier calibration logic.

Pure functions, no I/O side effects, so both the offline analysis script
(analyze_mae.py) and the live self-improvement loop (AutoTuner) compute MAE and
recommend stop changes with ONE algorithm — no divergence.

MAE here is read from a trade's stored `mae_price` (captured live by the intraday
monitor). The ATR at entry is backed out of the original stop distance:
    stop = entry ± current_mult × ATR   ⇒   ATR = |entry - stop| / current_mult
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# Per-strategy default ATR stop multipliers, matching SignalAgent's stop logic:
# swing reads risk_config.atr_multiplier; intraday/positional are fixed in code.
_RISK_PATH = Path(__file__).parent.parent / "config" / "risk_config.yaml"
_FIXED_DEFAULTS = {"intraday": 1.0, "positional": 2.0}


def default_stop_mult(strategy: str) -> float:
    if strategy in _FIXED_DEFAULTS:
        return _FIXED_DEFAULTS[strategy]
    try:
        with open(_RISK_PATH) as f:
            return float(yaml.safe_load(f)["stop_loss"]["atr_multiplier"])
    except Exception:
        return 1.5


def net_pnl(trade: dict) -> float:
    """Realized P&L including any booked partials."""
    return (trade.get("pnl") or 0.0) + (trade.get("realized_pnl") or 0.0)


def mae_atr(trade: dict, current_mult: float) -> float | None:
    """MAE in ATR units from the stored mae_price, or None if not computable."""
    mae_price = trade.get("mae_price")
    entry = trade.get("entry_price")
    stop = trade.get("stop_loss")
    if mae_price is None or not entry or not stop or not current_mult:
        return None
    side = (trade.get("signal_type") or "BUY").upper()
    adverse = (entry - mae_price) if side == "BUY" else (mae_price - entry)
    adverse = max(adverse, 0.0)
    atr = abs(entry - stop) / current_mult
    if atr <= 0:
        return None
    return adverse / atr


def recommend_stop_multiplier(win_maes: list[float], loss_maes: list[float],
                              current_mult: float, *, min_winners: int,
                              mult_min: float, mult_max: float):
    """Return (target_multiplier, reason) or (None, reason).

    The target sits at the winners' 90th-percentile MAE — wide enough that 90% of
    winning trades survive their worst dip. Only recommended when (a) the winner
    sample is large enough AND (b) winners and losers actually separate: if losers
    routinely dip no further than the proposed stop, widening just holds losers
    longer with no benefit, so we return None. The target is NOT step-limited here;
    the caller moves toward it gradually.
    """
    if len(win_maes) < min_winners:
        return None, f"{len(win_maes)} winners with MAE (need {min_winners})"
    reco = float(np.percentile(win_maes, 90))
    if loss_maes and float(np.median(loss_maes)) <= reco:
        return None, (f"winner/loser MAE overlap (loser median "
                      f"{np.median(loss_maes):.2f}× ≤ proposed {reco:.2f}×) — "
                      f"stops have no separating power")
    target = round(min(max(reco, mult_min), mult_max), 2)
    if abs(target - current_mult) < 0.1:
        return None, f"target {target}× ≈ current {current_mult}×"
    return target, f"winners' p90 MAE {reco:.2f}×ATR clears loser median"
