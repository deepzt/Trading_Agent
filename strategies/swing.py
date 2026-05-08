"""
Swing trading strategy helpers (2-7 day hold).
Core logic lives in SignalAgent._swing_signal().
"""

from typing import Optional
import pandas as pd


def get_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Find nearest support and resistance levels using recent price action."""
    if len(df) < lookback:
        return {}
    recent = df.tail(lookback)
    resistance = float(recent["high"].max())
    support = float(recent["low"].min())
    pivot = (float(df.iloc[-1]["high"]) + float(df.iloc[-1]["low"]) + float(df.iloc[-1]["close"])) / 3
    return {
        "resistance": round(resistance, 2),
        "support": round(support, 2),
        "pivot": round(pivot, 2),
        "r1": round(2 * pivot - support, 2),
        "s1": round(2 * pivot - resistance, 2),
    }


def fibonacci_targets(swing_low: float, swing_high: float) -> dict:
    """Fibonacci retracement/extension levels for swing trades."""
    diff = swing_high - swing_low
    return {
        "fib_236": round(swing_high - 0.236 * diff, 2),
        "fib_382": round(swing_high - 0.382 * diff, 2),
        "fib_500": round(swing_high - 0.500 * diff, 2),
        "fib_618": round(swing_high - 0.618 * diff, 2),
        "ext_1272": round(swing_high + 0.272 * diff, 2),   # T1 extension
        "ext_1618": round(swing_high + 0.618 * diff, 2),   # T2 extension
    }
