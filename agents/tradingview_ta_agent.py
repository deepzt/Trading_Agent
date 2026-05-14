"""
Fetches TradingView's aggregated TA ratings for NSE symbols via tradingview-ta.
Uses get_multiple_analysis() to batch all symbols in one HTTP request instead of
one request per symbol (which triggers 429s on 50-symbol watchlists).
Ratings are cached for 60 minutes — TV daily ratings don't change intraday.
"""

from __future__ import annotations

import time
from typing import Dict, List

from agents.base_agent import BaseAgent

try:
    from tradingview_ta import get_multiple_analysis, Interval
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

# Confidence score adjustment per TV rating
TV_SCORE_ADJ: Dict[str, float] = {
    "STRONG_BUY": 1.5,
    "BUY": 1.0,
    "NEUTRAL": 0.0,
    "SELL": -1.0,
    "STRONG_SELL": -1.5,
    "UNKNOWN": 0.0,
}

_CACHE_TTL = 3600  # seconds — daily TV ratings don't change every 15 min


class TradingViewTAAgent(BaseAgent):
    def __init__(self):
        super().__init__("TVRatings")
        self._cache: Dict[str, str] = {}
        self._cache_ts: float = 0.0

    def run(self, symbols: List[str]) -> Dict[str, str]:
        """Returns {symbol: rating} for each symbol. Never raises."""
        if not _TV_AVAILABLE:
            self.log_warning("tradingview-ta not installed — TV ratings skipped")
            return {s: "UNKNOWN" for s in symbols}

        if time.time() - self._cache_ts < _CACHE_TTL and self._cache:
            self.log_info(f"TV ratings: served {len(symbols)} symbols from cache")
            return {s: self._cache.get(s, "UNKNOWN") for s in symbols}

        ratings = self._fetch_all(symbols)
        self._cache = ratings
        self._cache_ts = time.time()

        bullish = sum(1 for r in ratings.values() if r in ("BUY", "STRONG_BUY"))
        self.log_info(f"TV ratings: {len(ratings)} symbols fetched, {bullish} bullish")
        return ratings

    def _fetch_all(self, symbols: List[str], _retries: int = 2) -> Dict[str, str]:
        tv_symbols = [f"NSE:{s}" for s in symbols]
        for attempt in range(_retries + 1):
            try:
                results = get_multiple_analysis(
                    screener="india",
                    interval=Interval.INTERVAL_1_DAY,
                    symbols=tv_symbols,
                )
                ratings: Dict[str, str] = {}
                for s in symbols:
                    analysis = results.get(f"NSE:{s}")
                    if analysis:
                        ratings[s] = analysis.summary.get("RECOMMENDATION", "UNKNOWN")
                    else:
                        ratings[s] = "UNKNOWN"
                return ratings
            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < _retries:
                    wait = 20 * (attempt + 1)
                    self.log_warning(f"TV bulk 429, retrying in {wait}s (attempt {attempt + 1}/{_retries})")
                    time.sleep(wait)
                else:
                    self.log_warning(f"TV bulk fetch failed: {e} — all symbols set to UNKNOWN")
                    return {s: "UNKNOWN" for s in symbols}
        return {s: "UNKNOWN" for s in symbols}
