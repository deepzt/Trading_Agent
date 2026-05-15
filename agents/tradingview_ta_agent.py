"""
In-house composite TA score replacing TradingView's aggregated ratings.
Computes a 0–10 score from already-computed indicators in the enriched DataFrames,
then maps to the same 5-level labels (STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL)
that SignalAgent uses for confidence adjustment.

Components and weights:
  1. RSI zone              (max ±1.5) — ideal bullish 55-70, penalise extremes
  2. MACD histogram        (max ±1.5) — direction (±1) + momentum trend (±0.5)
  3. EMA alignment         (max ±1.5) — price > EMA9 > EMA21
  4. Above EMA 200         (max ±1.0) — primary trend filter
  5. ADX directional       (max ±1.0) — DI+ vs DI- when ADX ≥ 20
  6. OBV slope             (max ±1.0) — 5-bar accumulation/distribution
  7. Bollinger Band pct    (max ±0.5) — position within bands
  8. Above rolling VWAP    (max ±0.5) — 20-bar VWAP
  Max raw = ±8.5; normalised to 0–10.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

# Confidence score adjustment — unchanged so SignalAgent needs no edits
COMPOSITE_SCORE_ADJ: Dict[str, float] = {
    "STRONG_BUY": 1.5,
    "BUY": 1.0,
    "NEUTRAL": 0.0,
    "SELL": -1.0,
    "STRONG_SELL": -1.5,
    "UNKNOWN": 0.0,
}

# Normalised score (0–10) → label
_SCORE_THRESHOLDS = [
    (7.5, "STRONG_BUY"),
    (6.0, "BUY"),
    (4.0, "NEUTRAL"),
    (2.5, "SELL"),
    (0.0, "STRONG_SELL"),
]

_MAX_RAW = 8.5
_RANGE = 2 * _MAX_RAW  # 17.0


class CompositeTAAgent(BaseAgent):
    """Computes in-house composite TA score from enriched indicator DataFrames."""

    def __init__(self):
        super().__init__("CompositeTA")

    def run(self, enriched_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Return {symbol: label} for every symbol in enriched_data."""
        ratings: Dict[str, str] = {}
        for symbol, df in enriched_data.items():
            if df is None or df.empty:
                ratings[symbol] = "UNKNOWN"
                continue
            try:
                row = df.iloc[-1]
                raw = self._score_row(row)
                norm = (raw + _MAX_RAW) / _RANGE * 10.0
                norm = max(0.0, min(10.0, norm))
                ratings[symbol] = _to_label(norm)
            except Exception as e:
                self.log_warning(f"{symbol}: composite score failed — {e}")
                ratings[symbol] = "UNKNOWN"

        bullish = sum(1 for r in ratings.values() if r in ("BUY", "STRONG_BUY"))
        self.log_info(f"CompositeTA: scored {len(ratings)} symbols, {bullish} bullish")
        return ratings

    def _score_row(self, row: pd.Series) -> float:
        raw = 0.0

        # 1. RSI zone (max ±1.5)
        rsi = row.get("RSI_14")
        if _valid(rsi):
            if 55 <= rsi <= 70:
                raw += 1.5
            elif rsi > 70:
                raw += 0.5   # overbought — reduced bonus
            elif 45 < rsi < 55:
                raw += 0.25  # mild momentum
            elif 30 <= rsi <= 45:
                raw -= 1.0
            else:
                raw -= 1.5   # rsi < 30 — oversold/bearish trend

        # 2. MACD histogram direction + momentum (max ±1.5)
        macdh = row.get("MACDh_12_26_9")
        macdh_prev = row.get("MACDh_prev")
        if _valid(macdh):
            raw += 1.0 if macdh > 0 else -1.0
            if _valid(macdh_prev):
                raw += 0.5 if macdh > macdh_prev else -0.5

        # 3. EMA alignment (max ±1.5)
        if row.get("ema_bullish_align"):
            raw += 1.5
        elif row.get("ema_bearish_align"):
            raw -= 1.5

        # 4. Above EMA 200 (max ±1.0)
        above_200 = row.get("above_ema200")
        if _valid(above_200):
            raw += 1.0 if above_200 else -1.0

        # 5. ADX directional strength (max ±1.0)
        adx = row.get("ADX_14")
        dmp = row.get("DMP_14")
        dmn = row.get("DMN_14")
        if _valid(adx) and _valid(dmp) and _valid(dmn) and adx >= 20:
            raw += 1.0 if dmp > dmn else -1.0

        # 6. OBV slope (max ±1.0)
        obv_slope = row.get("OBV_slope")
        if _valid(obv_slope):
            raw += 1.0 if obv_slope > 0 else -1.0

        # 7. Bollinger Band percent position (max ±0.5)
        bbp = row.get("BBP_20_2.0")
        if _valid(bbp):
            if bbp > 0.6:
                raw += 0.5
            elif bbp < 0.4:
                raw -= 0.5

        # 8. Above rolling VWAP (max ±0.5)
        above_vwap = row.get("above_vwap")
        if _valid(above_vwap):
            raw += 0.5 if above_vwap else -0.5

        return raw


def _to_label(score: float) -> str:
    for threshold, label in _SCORE_THRESHOLDS:
        if score >= threshold:
            return label
    return "STRONG_SELL"


def _valid(val) -> bool:
    """True if val is a non-NaN scalar."""
    if val is None:
        return False
    try:
        return not np.isnan(val)
    except TypeError:
        return True  # booleans and other non-float types are always valid
