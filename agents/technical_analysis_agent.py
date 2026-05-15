"""
Computes technical indicators on OHLCV DataFrames using the `ta` library.
Indicators: RSI, MACD, Bollinger Bands, EMA(9/21/50/200), ATR, OBV, ADX.
Also detects key candlestick patterns.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from agents.base_agent import BaseAgent


class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("TAAgent")

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enrich each symbol's OHLCV DataFrame with indicator columns."""
        enriched = {}
        for symbol, df in data.items():
            result = self.compute_indicators(df)
            if result is not None:
                enriched[symbol] = result
        self.log_info(f"Computed indicators for {len(enriched)} symbols")
        return enriched

    def compute_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or len(df) < 30:
            return None
        df = df.copy()

        # Drop bars where High == Low (zero True Range) — causes ADX division warnings
        flat_mask = df["high"] == df["low"]
        if flat_mask.any():
            df = df[~flat_mask].copy()
        if len(df) < 30:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"].astype(float)

        # ── Trend: EMA ────────────────────────────────────────────────────
        for period in [9, 21, 50, 200]:
            df[f"EMA_{period}"] = EMAIndicator(close=close, window=period, fillna=False).ema_indicator()

        # ── Momentum: RSI ─────────────────────────────────────────────────
        df["RSI_14"] = RSIIndicator(close=close, window=14, fillna=False).rsi()

        # ── Momentum: MACD ────────────────────────────────────────────────
        macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        df["MACD_12_26_9"] = macd_obj.macd()
        df["MACDs_12_26_9"] = macd_obj.macd_signal()
        df["MACDh_12_26_9"] = macd_obj.macd_diff()

        # ── Momentum: Stochastic ──────────────────────────────────────────
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3, fillna=False)
        df["STOCHk_14_3_3"] = stoch.stoch()
        df["STOCHd_14_3_3"] = stoch.stoch_signal()

        # ── Volatility: Bollinger Bands ───────────────────────────────────
        bb = BollingerBands(close=close, window=20, window_dev=2, fillna=False)
        df["BBU_20_2.0"] = bb.bollinger_hband()
        df["BBM_20_2.0"] = bb.bollinger_mavg()
        df["BBL_20_2.0"] = bb.bollinger_lband()
        df["BBB_20_2.0"] = bb.bollinger_wband()  # Band width
        df["BBP_20_2.0"] = bb.bollinger_pband()  # % position within bands

        # ── Volatility: ATR ───────────────────────────────────────────────
        df["ATRr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14, fillna=False).average_true_range()

        # ── Volume: OBV ───────────────────────────────────────────────────
        df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume, fillna=False).on_balance_volume()
        df["volume_ma20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_ma20"]

        # ── Trend strength: ADX ───────────────────────────────────────────
        # np.errstate suppresses division-by-zero RuntimeWarnings from the ta
        # library when ATR is briefly zero; the library correctly returns NaN
        # for those bars so the warning is safe to silence.
        with np.errstate(invalid="ignore", divide="ignore"):
            adx_obj = ADXIndicator(high=high, low=low, close=close, window=14, fillna=False)
            df["ADX_14"] = adx_obj.adx()
            df["DMP_14"] = adx_obj.adx_pos()
            df["DMN_14"] = adx_obj.adx_neg()

        # ── Candlestick patterns ──────────────────────────────────────────
        df = self._detect_candle_patterns(df)

        # ── Derived composite signals ─────────────────────────────────────
        df = self._compute_derived(df)

        return df

    def _detect_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        full_range = h - l
        upper_wick = h - c.where(c > o, o)
        lower_wick = c.where(c < o, o) - l

        safe_range = full_range.replace(0, np.nan)

        df["pat_doji"] = (body / safe_range).fillna(1.0) < 0.1
        df["pat_hammer"] = (lower_wick > 2 * body) & (upper_wick < body)
        df["pat_shooting_star"] = (upper_wick > 2 * body) & (lower_wick < body)
        df["pat_bull_engulf"] = (c > o) & (o.shift(1) > c.shift(1)) & (o <= c.shift(1)) & (c >= o.shift(1))
        df["pat_bear_engulf"] = (o > c) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))

        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        ema9 = df.get("EMA_9")
        ema21 = df.get("EMA_21")
        ema50 = df.get("EMA_50")
        ema200 = df.get("EMA_200")

        if ema9 is not None and ema21 is not None:
            df["ema_bullish_align"] = (c > ema9) & (ema9 > ema21)
            df["ema_bearish_align"] = (c < ema9) & (ema9 < ema21)
            df["ema_cross_bull"] = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
            df["ema_cross_bear"] = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))

        if ema50 is not None:
            df["above_ema50"] = c > ema50
        if ema200 is not None:
            df["above_ema200"] = c > ema200

        df["prev_day_high"] = df["high"].shift(1)
        df["prev_day_low"] = df["low"].shift(1)
        df["breakout_up"] = c > df["prev_day_high"]
        df["breakout_down"] = c < df["prev_day_low"]

        if len(df) >= 252:
            df["52w_high"] = df["high"].rolling(252).max()
            df["52w_low"] = df["low"].rolling(252).min()
            df["pct_from_52w_high"] = ((c - df["52w_high"]) / df["52w_high"]) * 100
            df["pct_from_52w_low"] = ((c - df["52w_low"]) / df["52w_low"]) * 100

        # Rolling VWAP (20-bar volume-weighted average price, useful on daily bars)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].astype(float)
        df["VWAP_20"] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
        df["above_vwap"] = c > df["VWAP_20"]

        # OBV slope over 5 bars — positive = accumulation
        if "OBV" in df.columns:
            df["OBV_slope"] = df["OBV"] - df["OBV"].shift(5)

        # MACD histogram direction — needed by composite TA score
        if "MACDh_12_26_9" in df.columns:
            df["MACDh_prev"] = df["MACDh_12_26_9"].shift(1)

        return df

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Return a human-readable indicator summary for the latest bar."""
        if df is None or df.empty:
            return {}
        row = df.iloc[-1]

        def safe(col, fmt=".2f"):
            val = row.get(col)
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{float(val):{fmt}}"
            except Exception:
                return "N/A"

        return {
            "close": safe("close"),
            "rsi": safe("RSI_14"),
            "macd": safe("MACD_12_26_9"),
            "macd_signal": safe("MACDs_12_26_9"),
            "macd_hist": safe("MACDh_12_26_9"),
            "ema_9": safe("EMA_9"),
            "ema_21": safe("EMA_21"),
            "ema_50": safe("EMA_50"),
            "ema_200": safe("EMA_200"),
            "atr": safe("ATRr_14"),
            "adx": safe("ADX_14"),
            "bb_upper": safe("BBU_20_2.0"),
            "bb_lower": safe("BBL_20_2.0"),
            "volume_ratio": safe("volume_ratio"),
            "ema_cross_bull": bool(row.get("ema_cross_bull", False)),
            "ema_cross_bear": bool(row.get("ema_cross_bear", False)),
            "above_ema200": bool(row.get("above_ema200", False)),
        }
