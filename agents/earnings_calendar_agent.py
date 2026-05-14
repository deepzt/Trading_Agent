"""
Fetches NSE corporate event calendar and identifies watchlist symbols with an
upcoming board meeting (results/dividends) within N trading days.

Signals blocked by this agent get logged with claude_verdict = "EARNINGS_BLACKOUT"
in RiskAgent, preventing binary-event gap risk.

Falls back gracefully to an empty blackout list on any NSE API failure —
the trading pipeline must never crash due to this agent.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Set

import requests

from agents.base_agent import BaseAgent

_NSE_BASE = "https://www.nseindia.com"
_CALENDAR_URL = "https://www.nseindia.com/api/event-calendar"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}
_HOLIDAYS_PATH = Path(__file__).parent.parent / "config" / "market_holidays.json"


def _load_holidays() -> Set[date]:
    try:
        data = json.loads(_HOLIDAYS_PATH.read_text())
        holidays = set()
        for dates in data.values():
            if isinstance(dates, list):
                for d in dates:
                    try:
                        holidays.add(date.fromisoformat(d))
                    except Exception:
                        pass
        return holidays
    except Exception:
        return set()


def _trading_days_between(start: date, end: date, holidays: Set[date]) -> int:
    """Count trading days from start (exclusive) to end (inclusive)."""
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            count += 1
        current += timedelta(days=1)
    return count


class EarningsCalendarAgent(BaseAgent):
    def __init__(self):
        super().__init__("EarningsCalendar")
        self._holidays = _load_holidays()

    def run(self, symbols: List[str], blackout_days: int = 3) -> List[str]:
        """
        Returns list of symbols with a board meeting within `blackout_days` trading days.
        Never raises — returns [] on any failure.
        """
        try:
            events = self._fetch_calendar()
            if not events:
                self.log_warning("NSE calendar returned no events — no blackouts applied")
                return []

            today = date.today()
            blackout = []
            for sym in symbols:
                if self._within_blackout(sym, events, today, blackout_days):
                    blackout.append(sym)
                    self.log_info(f"{sym}: EARNINGS BLACKOUT — board meeting within {blackout_days} trading days")

            self.log_info(f"Earnings blackout: {len(blackout)} symbols — {blackout or 'none'}")
            return blackout

        except Exception as e:
            self.log_warning(f"Earnings calendar failed: {e} — no blackouts applied")
            return []

    def _fetch_calendar(self) -> list:
        try:
            session = requests.Session()
            session.headers.update(_NSE_HEADERS)
            session.get(_NSE_BASE, timeout=10)  # Acquire session cookie
            resp = session.get(_CALENDAR_URL, timeout=10)
            if resp.status_code == 200:
                return resp.json() if isinstance(resp.json(), list) else []
            self.log_warning(f"NSE calendar HTTP {resp.status_code}")
            return []
        except Exception as e:
            self.log_warning(f"NSE calendar fetch error: {e}")
            return []

    def _within_blackout(self, symbol: str, events: list, today: date, days: int) -> bool:
        for event in events:
            sym = event.get("symbol", "")
            if sym.upper() != symbol.upper():
                continue
            purpose = (event.get("purpose") or event.get("desc") or "").lower()
            if not any(kw in purpose for kw in ("board meeting", "results", "dividend", "financial result")):
                continue
            date_str = event.get("date") or event.get("bm_date") or ""
            try:
                event_date = date.fromisoformat(date_str[:10])
            except Exception:
                continue
            if event_date < today:
                continue
            td = _trading_days_between(today, event_date, self._holidays)
            if td <= days:
                return True
        return False
