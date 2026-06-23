"""
Generates precise BUY/SELL signals with entry, stop-loss, and target prices.
Runs all active strategy screeners and scores each signal 0-10.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import yaml
from pathlib import Path
from sqlalchemy import create_engine, text

from agents.base_agent import BaseAgent

_IST = pytz.timezone("Asia/Kolkata")

_COMPOSITE_ADJ: Dict[str, float] = {
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
        self.status = "PENDING"              # PENDING → APPROVED → EXECUTED / REJECTED / SKIPPED
        self.rejection_reason = None         # why a risk/execute stage dropped an approved signal
        self.claude_verdict = None
        self.claude_reasoning = None
        self.composite_rating: str = "UNKNOWN"

        # Computed
        self.recompute_risk()

    def recompute_risk(self) -> None:
        """Refresh derived risk fields (risk_reward, sl_pct) after entry/SL/target
        mutation — e.g. AI-modified stop, conditional-approval tighten, or the
        live-price shift at execution. Risk gates must never see stale values."""
        sl_dist = abs(self.entry_price - self.stop_loss)
        t2_dist = abs(self.target_2 - self.entry_price)
        self.risk_reward = round(t2_dist / sl_dist, 2) if sl_dist > 0 else 0.0
        self.sl_pct = round((sl_dist / self.entry_price) * 100, 2) if self.entry_price else 0.0

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
            "rejection_reason": self.rejection_reason,
            "claude_verdict": self.claude_verdict,
            "claude_reasoning": self.claude_reasoning,
            "composite_rating": self.composite_rating,
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
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/trading.db")
        self._engine = create_engine(db_url, echo=False)
        # Per-strategy ATR stop multipliers, auto-calibrated from MAE by the AutoTuner.
        # Loaded once per scan; falls back to YAML/code defaults when no override exists.
        self._stop_overrides = {}
        try:
            from agents.auto_tuner import AutoTuner
            self._stop_overrides = AutoTuner().get_stop_overrides()
        except Exception as e:
            self.log_warning(f"Could not load stop overrides: {e}")

    def _stop_atr_mult(self, strategy: str) -> float:
        """Resolve the base ATR stop multiplier for a strategy: MAE-calibrated override
        if present, else the strategy default (swing from risk_config, intraday/positional fixed)."""
        from agents.mae_calibration import default_stop_mult
        return float(self._stop_overrides.get(strategy, default_stop_mult(strategy)))

    def run(self, enriched_data: Dict[str, pd.DataFrame], active_strategies: Optional[List[str]] = None,
            composite_ratings: Optional[Dict[str, str]] = None, regime: str = "TRENDING",
            momentum_ranks: Optional[Dict[str, float]] = None) -> List[Signal]:
        """Screen all symbols and return list of signals above confidence threshold."""
        if active_strategies is None:
            active_strategies = self._cfg["strategies"]["active"]
        cr = composite_ratings or {}
        ranks = momentum_ranks or {}

        if regime == "CRISIS":
            self.log_info("Regime CRISIS — intraday signals suppressed this scan")

        cooldown_hours = self._cfg["signals"].get("cooldown_hours", 4)
        recent_symbols = self._get_recent_signal_symbols(cooldown_hours)

        signals: List[Signal] = []
        for symbol, df in enriched_data.items():
            if df is None or len(df) < 50:
                continue
            if symbol in recent_symbols:
                self.log_info(f"Cooldown active for {symbol} — skipping (signalled within {cooldown_hours}h)")
                continue
            for strategy in active_strategies:
                sig = self._screen(symbol, df, strategy, cr, regime, ranks)
                if sig:
                    signals.append(sig)

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        max_signals = self._cfg["signals"]["max_signals_per_run"]
        signals = signals[:max_signals]

        self.log_info(f"Generated {len(signals)} signals from {len(enriched_data)} symbols (regime={regime})")
        return signals

    def _apply_vol_scaling(self, score: float, atr: float, entry: float,
                           regime: str = "TRENDING") -> float:
        """Scale score down when a stock's ATR% exceeds the regime baseline.
        In VOLATILE regime the whole market is wider, so the baseline is raised to avoid
        penalising every signal uniformly — only outlier-volatile stocks are penalised."""
        vs_cfg = self._cfg.get("volatility_scaling", {})
        if not vs_cfg.get("enabled", True):
            return score
        if entry <= 0 or atr <= 0:
            return score
        baseline_normal = vs_cfg.get("baseline_atr_pct", 1.5) / 100
        # In VOLATILE regime raise the baseline so market-wide widening doesn't kill scores
        baseline = baseline_normal * 1.5 if regime == "VOLATILE" else baseline_normal
        current_atr_pct = atr / entry
        vol_scale = min(baseline / current_atr_pct, 1.0)
        return round(score * vol_scale, 2)

    def _get_recent_signal_symbols(self, hours: int) -> set:
        """Return symbols with an APPROVED signal in the last `hours` hours.
        Only approved (executed) signals trigger a cooldown — a rejected/low-confidence
        signal must not block a clean setup on the same symbol the next scan."""
        cutoff = (datetime.now(_IST) - timedelta(hours=hours)).isoformat()
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT DISTINCT symbol FROM signals_log "
                         "WHERE timestamp >= :cutoff AND status = 'APPROVED'"),
                    {"cutoff": cutoff},
                ).fetchall()
            return {r[0] for r in rows}
        except Exception as e:
            self.log_warning(f"Cooldown check failed: {e} — proceeding without cooldown")
            return set()

    def _screen(self, symbol: str, df: pd.DataFrame, strategy: str, composite_ratings: dict,
                regime: str = "TRENDING", momentum_ranks: dict = None) -> Optional[Signal]:
        try:
            if strategy == "swing":
                return self._swing_signal(symbol, df, composite_ratings, regime)
            elif strategy == "intraday":
                return self._intraday_signal(symbol, df, composite_ratings, regime)
            elif strategy == "positional":
                return self._positional_signal(symbol, df, composite_ratings, momentum_ranks)
            elif strategy == "mean_reversion":
                return self._mean_reversion_signal(symbol, df, composite_ratings, regime)
        except Exception as e:
            self.log_error(f"Signal error {symbol}/{strategy}: {e}")
        return None

    # ── Strategy screeners ─────────────────────────────────────────────────

    def _swing_signal(self, symbol: str, df: pd.DataFrame, composite_ratings: dict, regime: str = "TRENDING") -> Optional[Signal]:
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

        # 4. Volume confirmation — a breakout/cross should be backed by above-average volume
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

        # 7. Composite TA cross-validation
        composite_rating = composite_ratings.get(symbol, "UNKNOWN")
        score += _COMPOSITE_ADJ.get(composite_rating, 0.0)
        if composite_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"CompositeTA: {composite_rating}")

        if score < 5.0 or not reasons:
            return None

        entry = float(row["close"])
        atr = row.get("ATRr_14", entry * 0.02)
        score = self._apply_vol_scaling(score, atr, entry, regime)

        atr_mult = self._stop_atr_mult("swing")
        if regime == "VOLATILE":
            atr_mult = round(atr_mult * 1.33, 2)  # Wider stop in volatile markets
        sl_dist = atr * atr_mult

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * self._risk["targets"]["t1_rr_ratio"]
        target_2 = entry + sl_dist * self._risk["targets"]["t2_rr_ratio"]

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="swing",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="1d"
        )
        sig.composite_rating = composite_rating
        return sig

    def _intraday_signal(self, symbol: str, df: pd.DataFrame, composite_ratings: dict, regime: str = "TRENDING") -> Optional[Signal]:
        """Breakout above previous day's high with volume surge."""
        cfg = self._cfg["strategies"]["intraday"]
        if not cfg.get("enabled", True):
            return None  # Intraday screener disabled — uses daily candles, not live intraday bars
        if regime == "CRISIS":
            return None  # No intraday breakout signals during market crisis
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

        # 6. Composite TA cross-validation
        composite_rating = composite_ratings.get(symbol, "UNKNOWN")
        score += _COMPOSITE_ADJ.get(composite_rating, 0.0)
        if composite_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"CompositeTA: {composite_rating}")

        entry = close
        atr = row.get("ATRr_14", entry * 0.015)
        score = self._apply_vol_scaling(score, atr, entry, regime)
        atr_mult = self._stop_atr_mult("intraday")
        if regime == "VOLATILE":
            atr_mult = round(atr_mult * 1.33, 2)
        sl_dist = max(atr * atr_mult, (entry - float(row["low"])) * 1.1)

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * 1.0
        target_2 = entry + sl_dist * 2.0

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="intraday",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="15m"
        )
        sig.composite_rating = composite_rating
        return sig

    def _positional_signal(self, symbol: str, df: pd.DataFrame, composite_ratings: dict,
                           momentum_ranks: dict = None) -> Optional[Signal]:
        """Weekly trend following — price above 200 EMA, strong momentum, top sector rank."""
        cfg = self._cfg["strategies"]["positional"]
        if len(df) < 200:
            return None
        row = df.iloc[-1]

        # Sector-neutral momentum gate: must be in top 40% within sector
        if momentum_ranks:
            rank = momentum_ranks.get(symbol, 0.5)
            if rank < 0.6:
                self.log_info(f"{symbol}: positional skipped — sector momentum rank {rank:.2f} < 0.6")
                return None

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

        # 6. Volume confirmation — favour a rally backed by participation (turnover-based
        #    persistence belongs to a multi-month measure, not a single day's ratio)
        vol_ratio = row.get("volume_ratio", 1.0)
        if vol_ratio and vol_ratio >= 1.2:
            score += 0.5
            reasons.append(f"Volume confirmation ({vol_ratio:.1f}x avg)")

        # 7. Composite TA cross-validation
        composite_rating = composite_ratings.get(symbol, "UNKNOWN")
        score += _COMPOSITE_ADJ.get(composite_rating, 0.0)
        if composite_rating not in ("UNKNOWN", "NEUTRAL"):
            reasons.append(f"CompositeTA: {composite_rating}")

        entry = float(row["close"])
        atr = row.get("ATRr_14", entry * 0.025)
        score = self._apply_vol_scaling(score, atr, entry)  # positional: always use normal baseline
        sl_dist = atr * self._stop_atr_mult("positional")  # default 2.0 — wider stop for positional

        stop_loss = entry - sl_dist
        target_1 = entry + sl_dist * 1.5
        target_2 = entry + sl_dist * 3.0

        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="positional",
            entry_price=entry, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="1wk"
        )
        sig.composite_rating = composite_rating
        return sig

    def _mean_reversion_signal(self, symbol: str, df: pd.DataFrame,
                                composite_ratings: dict, regime: str = "TRENDING") -> Optional[Signal]:
        """Oversold bounce signal — fires ONLY in VOLATILE regime.
        Buys stocks near lower Bollinger Band with RSI < 35, targeting return to EMA21.
        Based on De Bondt & Thaler mean-reversion and Gervais et al. volume research."""
        if regime not in ("VOLATILE",):
            return None
        if len(df) < 20:
            return None

        cfg = self._cfg["strategies"].get("mean_reversion", {})
        row = df.iloc[-1]
        close = float(row["close"])

        # Must be near or below lower Bollinger Band
        bb_lower = row.get("BBL_20_2.0")
        bb_mid = row.get("BBM_20_2.0")
        if bb_lower is None or bb_mid is None:
            return None
        bb_lower = float(bb_lower)
        bb_mid = float(bb_mid)
        band_tolerance = cfg.get("bb_band_tolerance", 0.005)
        if close > bb_lower * (1 + band_tolerance):
            return None

        # RSI must be in oversold zone
        rsi = row.get("RSI_14")
        if rsi is None:
            return None
        rsi = float(rsi)
        if rsi >= cfg.get("rsi_oversold", 35):
            return None

        # Price must be below EMA21 (confirming downward pressure)
        ema21 = row.get("EMA_21")
        if ema21 is None:
            return None
        ema21 = float(ema21)
        if close >= ema21:
            return None

        reasons = [
            "Near lower Bollinger Band (oversold)",
            f"RSI {rsi:.0f} — oversold zone",
            "Below EMA21 — mean reversion setup",
        ]
        score = 6.0

        if rsi < 25:
            score += 1.0
            reasons.append("RSI critically oversold")

        # Avoid catching falling knives: penalise strong bearish TA
        composite_rating = composite_ratings.get(symbol, "NEUTRAL")
        if composite_rating == "STRONG_SELL":
            score -= 1.5
        elif composite_rating == "SELL":
            score -= 0.5
        elif composite_rating in ("BUY", "STRONG_BUY"):
            score += 0.5
            reasons.append(f"CompositeTA: {composite_rating}")

        if score < 5.0:
            return None

        entry = close
        atr = float(row.get("ATRr_14", entry * 0.015))
        low = float(row["low"])
        sl_dist = max(atr * 1.5, (entry - low) * 1.1)
        stop_loss = entry - sl_dist

        target_1 = ema21        # First target: return to EMA21
        reward = target_1 - entry
        if reward <= 0 or reward / sl_dist < cfg.get("min_rr", 1.0):
            return None

        target_2 = round(entry + reward * 2, 2)
        score = self._apply_vol_scaling(score, atr, entry, regime)

        # NOTE: strategy must be "mean_reversion", NOT "intraday" — the bounce to EMA21
        # is a multi-day move on daily candles. Labelling it intraday force-closed it at
        # 3:20 PM, applied the intraday cost model, and pooled its stats/Kelly/auto-tune
        # under the (disabled) intraday breakout strategy.
        sig = Signal(
            symbol=symbol, signal_type="BUY", strategy="mean_reversion",
            entry_price=entry, stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2), target_2=target_2,
            confidence=min(score, 10.0), reasons=reasons, timeframe="1d"
        )
        sig.composite_rating = composite_rating
        return sig
