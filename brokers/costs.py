"""
Realistic Indian equity transaction-cost model (NSE cash segment).

Models the charges a discount broker actually levies so paper-trading P&L and
backtests aren't flattered by costless fills:
  - Brokerage (delivery often free; intraday ~0.03% capped at ₹20)
  - STT  (delivery 0.1% both legs; intraday 0.025% sell leg only)
  - Exchange transaction charge, SEBI turnover fee
  - Stamp duty (buy leg only)
  - GST 18% on (brokerage + exchange + SEBI)

All percentages are turnover-based and approximate the FY25 NSE/SEBI schedule.
"""
from __future__ import annotations

# Rate constants (fractions of traded value unless noted)
_BROKERAGE_PCT_INTRADAY = 0.0003   # 0.03% intraday
_BROKERAGE_CAP = 20.0              # ₹20 per executed order cap
_STT_DELIVERY = 0.001              # 0.1% on BOTH buy and sell legs
_STT_INTRADAY_SELL = 0.00025       # 0.025% on the sell leg only
_EXCHANGE_TXN = 0.0000297          # NSE ~0.00297%
_SEBI_FEE = 0.000001               # ₹10 per crore
_STAMP_DELIVERY_BUY = 0.00015      # 0.015% on buy (delivery)
_STAMP_INTRADAY_BUY = 0.00003      # 0.003% on buy (intraday)
_GST = 0.18


def leg_cost(value: float, side: str, is_intraday: bool) -> float:
    """Total statutory + brokerage charges in INR for one leg (BUY or SELL)."""
    if value <= 0:
        return 0.0
    side = side.upper()

    if is_intraday:
        brokerage = min(value * _BROKERAGE_PCT_INTRADAY, _BROKERAGE_CAP)
        stt = value * _STT_INTRADAY_SELL if side == "SELL" else 0.0
        stamp = value * _STAMP_INTRADAY_BUY if side == "BUY" else 0.0
    else:
        brokerage = 0.0  # delivery brokerage is typically zero at discount brokers
        stt = value * _STT_DELIVERY
        stamp = value * _STAMP_DELIVERY_BUY if side == "BUY" else 0.0

    exchange = value * _EXCHANGE_TXN
    sebi = value * _SEBI_FEE
    gst = _GST * (brokerage + exchange + sebi)
    return brokerage + stt + stamp + exchange + sebi + gst


def round_trip_cost(entry_price: float, exit_price: float, quantity: float,
                    position_side: str, is_intraday: bool) -> float:
    """
    Total cost (INR) to open AND close `quantity` shares.
    position_side: "BUY" for a long (entry BUY / exit SELL), "SELL" for a short.
    """
    if quantity <= 0:
        return 0.0
    entry_side = "BUY" if position_side.upper() == "BUY" else "SELL"
    exit_side = "SELL" if position_side.upper() == "BUY" else "BUY"
    return (leg_cost(entry_price * quantity, entry_side, is_intraday)
            + leg_cost(exit_price * quantity, exit_side, is_intraday))
