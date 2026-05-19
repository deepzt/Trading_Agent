"""
Fetches OHLCV market data for NSE/BSE stocks via yfinance.
Caches results as parquet files to avoid repeated API calls.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import contextlib
import io
import logging
import time

import pandas as pd
import pytz
import yaml
import yfinance as yf

from agents.base_agent import BaseAgent

# Suppress yfinance's own logging and stderr prints — we log our own clean messages
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_IST = pytz.timezone("Asia/Kolkata")

_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}


def _yf_symbol(symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker format."""
    # M&M is a special case
    if symbol == "M&M":
        return "M%26M.NS"
    return f"{symbol}.NS"


class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAgent")
        self._config = self._load_config()

    def _load_config(self) -> dict:
        cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    def run(self, symbols: List[str], timeframe: str = "1d", days: int = 90) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols. Returns dict of {symbol: DataFrame}."""
        results = {}
        for symbol in symbols:
            df = self.get_historical_ohlcv(symbol, timeframe, days)
            if df is not None and not df.empty:
                results[symbol] = df
        self.log_info(f"Fetched data for {len(results)}/{len(symbols)} symbols", timeframe=timeframe)
        return results

    def get_historical_ohlcv(self, symbol: str, timeframe: str = "1d", days: int = 90) -> Optional[pd.DataFrame]:
        """Return OHLCV DataFrame for a symbol. Uses parquet cache."""
        cache_key = f"{symbol}_{timeframe}_{days}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_yfinance(symbol, timeframe, days)

        if df is not None and not df.empty:
            df = self._normalize_ohlcv(df)
            self._write_cache(cache_key, df)
        return df

    def get_live_quote(self, symbol: str) -> Optional[Dict]:
        """Return latest quote for a symbol using yfinance fast_info."""
        try:
            ticker = yf.Ticker(_yf_symbol(symbol))
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "last_price": float(info.last_price),
                "previous_close": float(info.previous_close),
                "open": float(info.open),
                "day_high": float(info.day_high),
                "day_low": float(info.day_low),
                "volume": int(info.shares),
                "timestamp": datetime.now(_IST).isoformat(),
            }
        except Exception as e:
            self.log_error(f"Live quote failed for {symbol}: {e}")
            return None

    def get_watchlist(self, group: str = "nifty50") -> List[str]:
        cfg_path = Path(__file__).parent.parent / "config" / "watchlist.yaml"
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        return data.get(group, [])

    # ── Private helpers ────────────────────────────────────────────────────

    def _fetch_yfinance(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        interval = _TIMEFRAME_MAP.get(timeframe, "1d")
        end = datetime.now()
        start = end - timedelta(days=days)
        ticker = yf.Ticker(_yf_symbol(symbol))

        for attempt in range(2):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index)
                    df.columns = [c.lower() for c in df.columns]
                    return df[["open", "high", "low", "close", "volume"]]
                if attempt == 0:
                    time.sleep(1.5)
            except Exception as e:
                if attempt == 1:
                    self.log_error(f"yfinance fetch failed for {symbol}: {e}")

        self.log_warning(f"{symbol}: no data returned — skipped for this scan")
        return None


    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df.sort_index()

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("&", "AND")
        return _CACHE_DIR / f"{safe}.pkl"

    def _read_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        # Cache TTL: 15 minutes for intraday, 15 minutes for daily (refreshes with dashboard)
        ttl_minutes = 15
        age_minutes = (datetime.now().timestamp() - path.stat().st_mtime) / 60
        if age_minutes > ttl_minutes:
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _write_cache(self, key: str, df: pd.DataFrame):
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(df, f)
