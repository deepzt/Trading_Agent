"""
Intraday strategy helpers.
Core logic lives in SignalAgent._intraday_signal().
This module provides supporting utilities for intraday trading.
"""

from datetime import datetime, time
import pytz

_IST = pytz.timezone("Asia/Kolkata")

ENTRY_START = time(9, 20)    # Start looking for entries after opening volatility settles
ENTRY_CUTOFF = time(14, 30)  # No new intraday entries after 2:30 PM
EXIT_BY = time(15, 15)       # Hard exit all intraday positions by 3:15 PM


def can_take_intraday_entry() -> bool:
    now = datetime.now(_IST).time()
    return ENTRY_START <= now <= ENTRY_CUTOFF


def must_exit_intraday() -> bool:
    now = datetime.now(_IST).time()
    return now >= EXIT_BY


def get_intraday_targets(entry: float, atr: float, signal_type: str = "BUY") -> dict:
    """Calculate intraday-specific tight SL and targets."""
    if signal_type == "BUY":
        return {
            "stop_loss": round(entry - atr * 1.0, 2),
            "target_1": round(entry + atr * 1.0, 2),
            "target_2": round(entry + atr * 2.0, 2),
        }
    return {
        "stop_loss": round(entry + atr * 1.0, 2),
        "target_1": round(entry - atr * 1.0, 2),
        "target_2": round(entry - atr * 2.0, 2),
    }
