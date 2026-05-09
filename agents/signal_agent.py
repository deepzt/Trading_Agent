"""
Generates precise BUY/SELL signals with entry, stop-loss, and target prices.
Runs all active strategy screeners and scores each signal 0-10.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import yaml
from pathlib import Path

from agents.base_agent import BaseAgent

_IST = pytz.timezone("Asia/Kolkata")

_TV_ADJ: Dict[str, float] = {
    "STRONG_BUY": 1.5, "BUY": 1.0, "NEUTRAL": 0.0,
    "SELL": -1.0, "STRONG_SELL": -1.5,
}


def _now_ist() -> str:
    return datetime.now(_IST).isoformat()


class Signal:
    """Represents a single trade signal with full entry/exit specification."""

    def __init__(self, symbol: str, signal_type: str, strategy: str,
                 entry_price: float, stop_loss: float, target_1: float,
                 target_2: float, confidence: float, reasons: List[str],
                 timeframe: str = "1d"):
        self.id = str(uuid.uuid4())[:8]
        self.symbol = symbol
        self.signal_type = signal_type        # "BUY" or "SELL"
        self.strategy = strategy              # "swing", "intraday", "positional"
        self.entry_price = round(entry_price, 2)
        self.stop_loss = round(stop_loss, 2)
        self.target_1 = round(target_1, 2)
        self.target_2 = round(target_2, 2)
        self.timeframe = timeframe
        self.confidence = round(confidence, 1)
        self.reasons = reasons
        self.timestamp = _now_ist()
        self.status = "PENDING"              # PENDING → APPROVED / REJECTED / EXECUTED
        self.claude_verdict = None
        self.claude_reasoning = None
        self.tv_rating: str = "UNKNOWN"

        # Computed
        sl_dist = abs(entry_price - stop_loss)
        t2_dist = abs(target_2 - entry_price)
        self.risk_reward = round(t2_dist / sl_dist, 2) if sl_dist > 0 else 0.0
        self.sl_pct = round((sl_dist / entry_price) * 100, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "strategy": self.strategy,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "risk_reward": self.risk_reward,
            "sl_pct": self.sl_pct,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "status": self.status,
            "claude_verdict": self.claude_verdict,
            "claude_reasoning": self.claude_reasoning,
            "tv_rating": self.tv_rating,
        }

    def format_alert(self) -> str:
        emoji = "🟢" if self.signal_type == "BUY" else "🔴"
        return (
            f"{emoji} *{self.signal_type} Signal: {self.symbol}*\n"
            f"Entry: ₹{self.entry_price:,.2f} | SL: ₹{self.stop_loss:,.2f}\n"
            f"T1: ₹{self.target_1:,.2f} | T2: ₹{self.target_2:,.2f}\n"
            f"Strategy: {self.strategy.title()} | RR: {self.risk_reward}x | Confidence: {self.confidence}/10\n"
            f"📋 {' · '.join(self.reasons[:3])}\n"
            f"🕐 {self.timestamp[:19]}"
        )


class SignalAgent(BaseAgent):
    def __init__(self):
        super().__init__("SignalAgent")
        cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)
        risk_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
        with open(risk_path) as f:
            self._risk = yaml.safe_load(f)

    def run(self, enriched_data: Dict[str, pd.DataFrame], active_strategies: Optional[List[str]] = None,
            tv_ratings: Optional[Dict[str, str]] = None) -> List[Signal]:
        """Screen all symbols and return list of signals above confidence threshold."""
        if active_strategies is None:
            active_strategies = self._cfg["strategies"]["active"]
        tv = tv_ratings or {}

        signals: List[Signal] = []
        for symbol, df in enriched_data.items():
            if df is None or len(df) < 50:
                continue
            for strategy in active_strategies:
                sig = self._screen(symbol, df, strategy, tv)
                if sig:
                    signals.append(sig)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        max_signals = self._cfg["signals"]["max_signals_per_run"]
        signals = signals[:max_signals]

        self.log_info(f"Generated {len(signals)} signals from {len(enriched_data)} symbols")
        return signals

    def _screen(self, symbol: str, df: pd.DataFrame, strategy: str, tv_ratings: dict) -> Optional[Signal]:
        try:
            if strategy == "swing":
                return self._swing_signal(symbol, df, tv_ratings)
            elif strategy == "intraday":
                return self._intraday_signal(symbol, df, tv_ratings)
            elif strategy == "positional":
                return self._positional_signal(symbol, df, tv_ratings)
        except Exception as e:
            self.log_error(f"Signal error {symbol}/{strategy}: {e}")
        return None

    # ── Strategy screeners ─────────────────────────────────────────────────

    def _swing_signal(self, symbol: str, df: pd.DataFrame, tv_ratings: dict) -> Optional[Signal]:
        """EMA 9/21 crossover + RSI in sweet spot + above EMA50."""
        cfg = self._cfg["strategies"]["swing"]
        row = df.iloc[-1]

        reasons = []
        score = 0.0

        # 1. EMA bullish crossover (high weight)
        if row.get("ema_cross_bull", False):
            score += 3.0
            reasons.append("EMA 9/21 bullish cross")
        elif row.get("ema_bullish_align", False):
            score += 1.5
            reasons.append("EMA bullish alignment")
        else:
            return None  # No trend structure — skip

        # 2. RSI in sweet spot (40-65)
        rsi = row.get("RSI_14")
        if rsi is not None and cfg["rsi_min"] <= rsi <= cfg["rsi_max"]:
            score += 2.0
            reasons.append(f"RSI {rsi:.0f} (momentum zone)")
        elif rsi and rsi < 35:
            score += 1.0
            reasons.append(f"RSI {rsi:.0f} (oversold bounce)")

        # 3. Above EMA50
        if row.get("above_ema50", False):
            score += 1.5
            reasons.append("Price above EMA50")

        # 4. Volume confirmation
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_ratio and vol_ratio >= 1.5:
            score += 1.5
            reasons.append(f"Volume surge {vol_ratio:.1f}x")
        elif vol_ratio and vol_ratio >= 1.2:
            score += 0.5
            reasons.append(f"Volume +{(vol_ratio-1)*100:.0f}%")

        # 5. ADX trend strength
        adx = row.get("ADX_14")
        if adx and adx >= 25:
            score += 1.0
            reasons.append(f"ADX {adx:.0f} (strong trend)")

        # 6. Above EMA200 (long-term bullish)
        if row.get("above_ema200", False):
            score += 0.5
            reasons.append("Above 200 EMA")

        # 7. TradingView independent cross-validation
        tv_rating = tv_ratings.get(symbol, "UNKNOWN")
        score += _TV_ADJ.get(tv_rating, 0.0)
        if tv_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"TradingView: {tv_rating}")

        if score < 5.0 or not reasons:
            return None

        entry = float(row["close"])
        atr = row.get("ATRr_14", entry * 0.02)
        sl_dist = atr * self._risk["stop_loss"]["atr_multiplier"]

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * self._risk["targets"]["t1_rr_ratio"]
        target_2 = entry + sl_dist * self._risk["targets"]["t2_rr_ratio"]

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="swing",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="1d"
        )
        sig.tv_rating = tv_rating
        return sig

    def _intraday_signal(self, symbol: str, df: pd.DataFrame, tv_ratings: dict) -> Optional[Signal]:
        """Breakout above previous day's high with volume surge."""
        cfg = self._cfg["strategies"]["intraday"]
        if len(df) < 5:
            return None
        row = df.iloc[-1]
        prev = df.iloc[-2]

        reasons = []
        score = 0.0

        # 1. Price breaking above previous day high
        prev_high = float(prev["high"])
        close = float(row["close"])
        breakout_level = prev_high * (1 + cfg["breakout_buffer_pct"])

        if close > breakout_level:
            score += 3.0
            reasons.append(f"Breakout above ₹{prev_high:.2f}")
        else:
            return None

        # 2. Volume surge
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_ratio and vol_ratio >= cfg["volume_surge_multiplier"]:
            score += 2.5
            reasons.append(f"Volume {vol_ratio:.1f}x average")
        else:
            return None  # Breakout without volume = false signal

        # 3. RSI not overbought
        rsi = row.get("RSI_14")
        if rsi is not None and rsi < 70:
            score += 1.5
            reasons.append(f"RSI {rsi:.0f}")

        # 4. MACD histogram positive (momentum)
        macd_hist = row.get("MACDh_12_26_9")
        if macd_hist and macd_hist > 0:
            score += 1.5
            reasons.append("MACD positive momentum")

        # 5. EMA support
        if row.get("ema_bullish_align", False):
            score += 1.0
            reasons.append("EMA aligned bullish")

        # 6. TradingView independent cross-validation
        tv_rating = tv_ratings.get(symbol, "UNKNOWN")
        score += _TV_ADJ.get(tv_rating, 0.0)
        if tv_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"TradingView: {tv_rating}")

        entry = close
        atr = row.get("ATRr_14", entry * 0.015)
        sl_dist = max(atr * 1.0, (entry - float(row["low"])) * 1.1)

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * 1.0
        target_2 = entry + sl_dist * 2.0

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="intraday",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="15m"
        )
        sig.tv_rating = tv_rating
        return sig

    def _positional_signal(self, symbol: str, df: pd.DataFrame, tv_ratings: dict) -> Optional[Signal]:
        """Weekly trend following — price above 200 EMA, strong momentum."""
        cfg = self._cfg["strategies"]["positional"]
        if len(df) < 200:
            return None
        row = df.iloc[-1]

        reasons = []
        score = 0.0

        # 1. Must be above 200 EMA (long-term bull)
        if not row.get("above_ema200", False):
            return None
        score += 2.0
        reasons.append("Above 200 EMA (long-term bullish)")

        # 2. RSI in 45-70 zone
        rsi = row.get("RSI_14")
        if rsi and cfg["rsi_min"] <= rsi <= cfg["rsi_max"]:
            score += 2.0
            reasons.append(f"RSI {rsi:.0f}")
        else:
            return None

        # 3. MACD bullish
        macd = row.get("MACD_12_26_9")
        macd_sig = row.get("MACDs_12_26_9")
        if macd and macd_sig and macd > macd_sig:
            score += 2.0
            reasons.append("MACD bullish")

        # 4. Near 52-week high (momentum, not extended)
        pct_from_52w = row.get("pct_from_52w_high", -100)
        if -15 <= pct_from_52w <= -2:
            score += 2.0
            reasons.append(f"{abs(pct_from_52w):.1f}% below 52-week high")
        elif pct_from_52w > -2:
            score += 1.0
            reasons.append("Near 52-week high (breakout)")

        # 5. ADX > 20 (trending)
        adx = row.get("ADX_14")
        if adx and adx >= 20:
            score += 1.0
            reasons.append(f"ADX {adx:.0f}")

        # 6. TradingView independent cross-validation
        tv_rating = tv_ratings.get(symbol, "UNKNOWN")
        score += _TV_ADJ.get(tv_rating, 0.0)
        if tv_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"TradingView: {tv_rating}")

        entry = float(row["close"])
        atr = row.get("ATRr_14", entry * 0.025)
        sl_dist = atr * 2.0  # Wider stop for positional

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * 1.5
        target_2 = entry + sl_dist * 3.0

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="positional",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="1wk"
        )
        sig.tv_rating = tv_rating
        return sig
