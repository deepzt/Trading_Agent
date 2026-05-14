"""
Sentiment agent — fetches NSE corporate announcements and Google News RSS,
then scores sentiment using Claude API.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import feedparser
import requests
import yaml

from agents.base_agent import BaseAgent

_NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
}
_GOOGLE_NEWS_TEMPLATE = "https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("SentimentAgent")
        self._client = None
        self._init_claude()

    def _init_claude(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        except Exception:
            pass

    def run(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Returns sentiment dict per symbol + market-level context.
        Format: {symbol: {sentiment, headline, score}, "_market": {trend, vix}}
        """
        result: Dict[str, dict] = {}

        # Market-level context
        result["_market"] = self._get_market_context()

        # Per-symbol news
        for symbol in symbols[:15]:  # Limit to avoid rate limits
            try:
                sentiment = self._get_symbol_sentiment(symbol)
                if sentiment:
                    result[symbol] = sentiment
            except Exception as e:
                self.log_error(f"Sentiment error for {symbol}: {e}")

        self.log_info(f"Sentiment fetched for {len(result)-1} symbols")
        return result

    def _get_symbol_sentiment(self, symbol: str) -> Optional[dict]:
        headlines = self._fetch_news_headlines(symbol)
        if not headlines:
            return None

        if self._client:
            return self._score_with_claude(symbol, headlines)
        return self._simple_sentiment(headlines)

    def _fetch_news_headlines(self, symbol: str) -> List[str]:
        headlines = []
        try:
            url = _GOOGLE_NEWS_TEMPLATE.format(symbol=symbol)
            feed = feedparser.parse(url)
            cutoff = datetime.now() - timedelta(days=2)
            for entry in feed.entries[:5]:
                headlines.append(entry.get("title", ""))
        except Exception:
            pass
        return [h for h in headlines if h]

    def _score_with_claude(self, symbol: str, headlines: List[str]) -> dict:
        prompt = f"""Analyze these recent news headlines for {symbol} (NSE India):

{chr(10).join(f'- {h}' for h in headlines)}

Respond ONLY with JSON:
{{"sentiment": "POSITIVE"|"NEUTRAL"|"NEGATIVE", "score": <-1 to 1>, "headline": "<most impactful headline>", "risk_flag": "<earnings/results/litigation/none>"}}"""

        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",  # Use cheaper model for sentiment
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            import json
            return json.loads(response.content[0].text.strip())
        except Exception:
            return self._simple_sentiment(headlines)

    def _simple_sentiment(self, headlines: List[str]) -> dict:
        positive_words = ["surge", "rally", "record", "profit", "growth", "strong", "beat", "upgrade"]
        negative_words = ["fall", "decline", "loss", "weak", "miss", "downgrade", "concern", "crash", "sell-off"]
        text = " ".join(headlines).lower()
        pos = sum(1 for w in positive_words if w in text)
        neg = sum(1 for w in negative_words if w in text)
        if pos > neg:
            sentiment = "POSITIVE"
        elif neg > pos:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        score = (pos - neg) / max(pos + neg, 1)
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "headline": headlines[0] if headlines else "",
            "risk_flag": "none",
        }

    def _get_market_context(self) -> dict:
        """Get Nifty 50 trend, India VIX, and 30-day history for regime detection."""
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="30d")
            if hist.empty:
                return {}
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            change_pct = ((last_close - prev_close) / prev_close) * 100
            trend = "BULLISH" if change_pct > 0.5 else ("BEARISH" if change_pct < -0.5 else "NEUTRAL")

            vix_ticker = yf.Ticker("^INDIAVIX")
            vix_hist = vix_ticker.history(period="30d")
            vix = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else 0

            return {
                "nifty_close": last_close,
                "nifty_change_pct": round(change_pct, 2),
                "trend": trend,
                "vix": round(vix, 1),
                "nifty_history": hist["Close"],  # pd.Series — used by RegimeDetector for EMA20
            }
        except Exception:
            return {}
