"""
Classifies the current day's F&O expiry risk for Indian derivatives markets.
Weekly expiry: every Thursday. Monthly expiry: last Thursday of the month.
If Thursday is an NSE holiday, expiry rolls back to Wednesday (or earlier).
"""
from __future__ import annotations

import pytz
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Set

import yaml

from agents.base_agent import BaseAgent

_HOLIDAYS_PATH = Path(__file__).parent.parent / "config" / "nse_holidays.yaml"
_IST = pytz.timezone("Asia/Kolkata")


def _load_holidays() -> Set[date]:
    try:
        with open(_HOLIDAYS_PATH) as f:
            data = yaml.safe_load(f)
        return {date.fromisoformat(str(d)) for d in data.get("holidays", [])}
    except Exception:
        return set()


def _last_thursday_of_month(year: int, month: int) -> date:
    next_month_first = date(year + (month // 12), (month % 12) + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


class FnOCalendarAgent(BaseAgent):
    def __init__(self):
        super().__init__("FnOCalendar")
        self._holidays = _load_holidays()

    def run(self, today: date = None) -> Dict[str, object]:
        """
        Returns:
          expiry_risk  — "NONE" | "EXPIRY_WEEK" (≤2 trading days away) | "EXPIRY_DAY"
          next_expiry  — ISO date string of the next F&O expiry
          is_monthly   — True if next_expiry is also the monthly (last Thursday) expiry
        """
        if today is None:
            today = datetime.now(_IST).date()

        # Days until this week's Thursday (0 = today IS Thursday)
        days_to_thursday = (3 - today.weekday()) % 7
        this_thursday = today + timedelta(days=days_to_thursday)

        # Roll back if Thursday is a holiday
        expiry = this_thursday
        while expiry in self._holidays:
            expiry -= timedelta(days=1)

        # Safety net: if rolled-back expiry is before today, move to next week
        if expiry < today:
            next_thursday = this_thursday + timedelta(days=7)
            expiry = next_thursday
            while expiry in self._holidays:
                expiry -= timedelta(days=1)

        days_to_expiry = (expiry - today).days
        if days_to_expiry == 0:
            risk = "EXPIRY_DAY"
        elif days_to_expiry <= 2:
            risk = "EXPIRY_WEEK"
        else:
            risk = "NONE"

        is_monthly = (this_thursday == _last_thursday_of_month(this_thursday.year, this_thursday.month))

        self.log_info(f"Expiry={expiry} risk={risk} monthly={is_monthly} days_away={days_to_expiry}")
        return {"expiry_risk": risk, "next_expiry": str(expiry), "is_monthly": is_monthly}
