"""
Positional strategy helpers (multi-week hold).
Core logic lives in SignalAgent._positional_signal().
"""

from typing import Dict, List
import pandas as pd


SECTORS = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
    "FMCG": ["HINDUNILVR", "ITC", "BRITANNIA", "NESTLEIND", "TATACONSUM"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "APOLLOHOSP"],
    "AUTO": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M"],
    "METALS": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "ENERGY": ["RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC"],
    "REALTY": ["ADANIENT", "ADANIPORTS"],
    "CONSUMER": ["TITAN", "ASIANPAINT", "BERGEPAINT", "HAVELLS", "VOLTAS"],
}


def get_sector(symbol: str) -> str:
    for sector, symbols in SECTORS.items():
        if symbol in symbols:
            return sector
    return "OTHER"


def sector_exposure(open_positions: List[dict]) -> Dict[str, float]:
    """Calculate % exposure per sector from open positions."""
    total_value = sum(p["entry_price"] * p["quantity"] for p in open_positions)
    if total_value == 0:
        return {}
    sector_values: Dict[str, float] = {}
    for pos in open_positions:
        sector = get_sector(pos["symbol"])
        value = pos["entry_price"] * pos["quantity"]
        sector_values[sector] = sector_values.get(sector, 0) + value
    return {s: round((v / total_value) * 100, 1) for s, v in sector_values.items()}
