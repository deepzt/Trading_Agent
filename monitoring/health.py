from datetime import datetime, time
import json
from pathlib import Path
import pytz

_IST = pytz.timezone("Asia/Kolkata")
_HOLIDAYS_PATH = Path(__file__).parent.parent / "config" / "market_holidays.json"


def _load_holidays() -> set:
    try:
        data = json.loads(_HOLIDAYS_PATH.read_text())
        holidays = set()
        for dates in data.values():
            holidays.update(dates)
        return holidays
    except Exception:
        return set()


def is_market_open() -> bool:
    now = datetime.now(_IST)
    if now.weekday() >= 5:
        return False
    date_str = now.strftime("%Y-%m-%d")
    if date_str in _load_holidays():
        return False
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= now.time() < market_close


def is_trading_day() -> bool:
    now = datetime.now(_IST)
    if now.weekday() >= 5:
        return False
    return now.strftime("%Y-%m-%d") not in _load_holidays()


def minutes_to_open() -> int:
    now = datetime.now(_IST)
    open_today = now.replace(hour=9, minute=15, second=0, microsecond=0)
    delta = (open_today - now).total_seconds() / 60
    return max(0, int(delta))


def current_ist_time() -> str:
    return datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S %Z")
