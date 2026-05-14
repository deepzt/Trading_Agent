"""
Classifies the current market regime based on India VIX and Nifty 50 vs 20-day EMA.

Regimes:
  TRENDING  — VIX below volatile threshold and Nifty above its 20-day EMA
  VOLATILE  — VIX elevated OR Nifty below its 20-day EMA
  CRISIS    — VIX at or above crisis threshold (all intraday signals blocked)

Thresholds are read from config/trading_config.yaml under the `regime:` key.
"""
from __future__ import annotations

import pandas as pd

from agents.base_agent import BaseAgent


class RegimeDetector(BaseAgent):
    def __init__(self):
        super().__init__("RegimeDetector")

    def run(self, vix: float, nifty_series: pd.Series, cfg: dict) -> str:
        """
        Returns "TRENDING", "VOLATILE", or "CRISIS".
        Defaults to "TRENDING" if inputs are invalid.
        """
        if not vix or nifty_series is None or nifty_series.empty:
            self.log_warning("Insufficient data for regime detection — defaulting to TRENDING")
            return "TRENDING"

        volatile_thresh = cfg.get("vix_volatile_threshold", 18)
        crisis_thresh = cfg.get("vix_crisis_threshold", 24)

        if vix >= crisis_thresh:
            self.log_info(f"Regime: CRISIS (VIX {vix} >= {crisis_thresh})")
            return "CRISIS"

        ema20 = nifty_series.ewm(span=20, adjust=False).mean().iloc[-1]
        nifty_last = nifty_series.iloc[-1]

        if vix >= volatile_thresh or nifty_last < ema20:
            reason = f"VIX {vix} >= {volatile_thresh}" if vix >= volatile_thresh else f"Nifty {nifty_last:.0f} < EMA20 {ema20:.0f}"
            self.log_info(f"Regime: VOLATILE ({reason})")
            return "VOLATILE"

        self.log_info(f"Regime: TRENDING (VIX {vix}, Nifty above EMA20)")
        return "TRENDING"
