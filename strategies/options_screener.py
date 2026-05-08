"""
NSE F&O Options screener — OI analysis and Put-Call Ratio (PCR).
Uses NSE unofficial data endpoints.
Note: Requires F&O segment data access.
"""

from __future__ import annotations

import requests
from typing import Optional, Dict


_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com",
}


def get_option_chain(symbol: str = "NIFTY") -> Optional[Dict]:
    """Fetch option chain from NSE. Returns PCR and max pain."""
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, headers=_NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return _process_option_chain(data)
    except Exception:
        return None


def _process_option_chain(data: dict) -> dict:
    records = data.get("records", {}).get("data", [])
    total_call_oi = 0
    total_put_oi = 0
    strike_oi: Dict[float, dict] = {}

    for rec in records:
        strike = rec.get("strikePrice", 0)
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        call_oi = ce.get("openInterest", 0)
        put_oi = pe.get("openInterest", 0)
        total_call_oi += call_oi
        total_put_oi += put_oi
        strike_oi[strike] = {"call_oi": call_oi, "put_oi": put_oi}

    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0

    # Max pain: strike with maximum OI on both sides
    max_pain_strike = max(strike_oi, key=lambda s: strike_oi[s]["call_oi"] + strike_oi[s]["put_oi"], default=0)

    return {
        "pcr": pcr,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "max_pain": max_pain_strike,
        "sentiment": "BULLISH" if pcr > 1.2 else ("BEARISH" if pcr < 0.7 else "NEUTRAL"),
    }


def get_pcr_signal(pcr: float) -> str:
    """Contrarian PCR interpretation."""
    if pcr > 1.3:
        return "CONTRARIAN_BUY"    # Excess put buying = smart money not that bearish
    elif pcr < 0.7:
        return "CONTRARIAN_SELL"   # Excess call buying = retail over-optimistic
    return "NEUTRAL"
