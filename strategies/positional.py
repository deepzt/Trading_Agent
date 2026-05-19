"""
Positional strategy helpers (multi-week hold).
Core logic lives in SignalAgent._positional_signal().
"""

from typing import Dict, List
import pandas as pd


SECTORS = {
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",   # Nifty 50
        "BANKBARODA", "CANBK", "UNIONBANK", "FEDERALBNK", "IDFCFIRSTB",            # Next 50
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",                       # Nifty 50
        "MPHASIS", "OFSS", "TATAELXSI", "PERSISTENT", "COFORGE", "NAUKRI",         # Next 50 / midcap
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "BRITANNIA", "NESTLEIND", "TATACONSUM",               # Nifty 50
        "DABUR", "MARICO", "COLPAL", "GODREJCP", "VBL", "PATANJALI",               # Next 50
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "APOLLOHOSP",                 # Nifty 50
        "LUPIN", "BIOCON", "AUROPHARMA", "ZYDUSLIFE",                              # Next 50
    ],
    "AUTO": [
        "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M",    # Nifty 50
        "MOTHERSON", "BOSCHLTD",                                                    # Next 50
    ],
    "METALS": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA",                          # Nifty 50
        "JINDALSTEL", "VEDL", "NMDC",                                               # Next 50
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC",                          # Nifty 50
        "GAIL", "HINDPETRO", "NHPC", "JSWENERGY",                                  # Next 50
    ],
    "CONSUMER": [
        "TITAN", "ASIANPAINT", "BERGEPAINT", "HAVELLS", "VOLTAS",                  # Nifty 50 / midcap
        "PIDILITIND", "PAGEIND", "POLYCAB", "TRENT",                               # Next 50 / midcap
    ],
    "CEMENT": [
        "ULTRACEMCO", "GRASIM",                                                     # Nifty 50
        "AMBUJACEM",                                                                # Next 50
    ],
    "INSURANCE": [
        "SBILIFE", "HDFCLIFE",                                                      # Nifty 50
        "ICICIGI", "ICICIPRULI", "LICI",                                            # Next 50
    ],
    "FINANCE": [
        "BAJFINANCE", "BAJAJFINSV",                                                 # Nifty 50
        "CHOLAFIN", "RECLTD", "MUTHOOTFIN",                                        # Next 50 / midcap
    ],
    "INDUSTRIALS": [
        "LT",                                                                        # Nifty 50
        "SIEMENS", "BHEL", "CONCOR", "INDUSTOWER",                                 # Next 50
    ],
    "CHEMICALS": [
        "UPL",                                                                       # Nifty 50
        "PIIND", "SRF",                                                             # Next 50
    ],
    "RETAIL": [
        "DMART", "IRCTC", "INDHOTEL", "OBEROIRLTY",                                # Next 50
    ],
    "TELECOM": [
        "BHARTIARTL",                                                                # Nifty 50
        "TATACOMM",                                                                 # Next 50
    ],
    "REALTY": [
        "ADANIENT", "ADANIPORTS",                                                   # Nifty 50
    ],
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
