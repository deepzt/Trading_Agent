"""
Fetches TradingView's aggregated TA ratings for NSE symbols via tradingview-ta.
Free, no auth, no browser automation. Falls back to UNKNOWN on any error.
"""

from __future__ import annotations

from typing import Dict, List

from agents.base_agent import BaseAgent

try:
    from tradingview_ta import TA_Handler, Interval
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


class TradingViewTAAgent(BaseAgent):
    def __init__(self):
        super().__init__("TVRatings")

    def run(self, symbols: List[str]) -> Dict[str, str]:
        """Returns {symbol: rating} for each symbol. Never raises."""
        if not _TV_AVAILABLE:
            self.log_warning("tradingview-ta not installed — TV ratings skipped")
            return {s: "UNKNOWN" for s in symbols}

        ratings: Dict[str, str] = {}
        for symbol in symbols:
            ratings[symbol] = self._fetch_rating(symbol)

        bullish = sum(1 for r in ratings.values() if r in ("BUY", "STRONG_BUY"))
        self.log_info(f"TV ratings: {len(ratings)} symbols, {bullish} bullish")
        return ratings

    def _fetch_rating(self, symbol: str) -> str:
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener="india",
                exchange="NSE",
                interval=Interval.INTERVAL_1_DAY,
            )
            return handler.get_analysis().summary.get("RECOMMENDATION", "UNKNOWN")
        except Exception as e:
            self.log_warning(f"TV rating failed for {symbol}: {e}")
            return "UNKNOWN"
