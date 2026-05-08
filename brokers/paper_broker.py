"""
Paper trading broker — simulates order fills with realistic slippage.
Backed by PortfolioAgent (SQLite). No real money involved.
"""

from __future__ import annotations

import os
import random
import uuid
from typing import Optional

from agents.signal_agent import Signal
from brokers.base_broker import BaseBroker
from monitoring.logger import get_logger

_logger = get_logger("PaperBroker")

_SLIPPAGE_PCT = 0.001  # 0.1% slippage on fills


class PaperBroker(BaseBroker):
    def __init__(self, portfolio_agent):
        self._portfolio = portfolio_agent
        self._orders: dict = {}  # order_id → order_dict

    def execute_signal(self, signal: Signal, quantity: int) -> Optional[str]:
        """High-level: place order for an approved signal and open position."""
        if os.getenv("TRADING_ENABLED", "true").lower() in ("false", "0", "no"):
            _logger.warning("TRADING_ENABLED=false — trade blocked at broker layer")
            return None
        fill_price = self._simulate_fill(signal.entry_price, signal.signal_type)
        order_id = self.place_order(
            trade_id=signal.id,
            symbol=signal.symbol,
            transaction_type=signal.signal_type,
            quantity=quantity,
            price=fill_price,
        )
        if order_id:
            self._portfolio.open_position(
                trade_id=signal.id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                strategy=signal.strategy,
                entry_price=fill_price,
                stop_loss=signal.stop_loss,
                target_1=signal.target_1,
                target_2=signal.target_2,
                quantity=quantity,
                confidence=signal.confidence,
                claude_verdict=signal.claude_verdict or "",
            )
            _logger.info(f"Paper order filled: {signal.symbol} x{quantity} @ ₹{fill_price}")
        return order_id

    def check_exits(self, live_prices: dict) -> list:
        """
        Check open positions against live prices for SL/TP hits.
        Returns list of closed trade dicts.
        """
        positions = self._portfolio.get_open_positions()
        closed = []

        for pos in positions:
            symbol = pos["symbol"]
            price = live_prices.get(symbol)
            if price is None:
                continue

            trade_id = pos["id"]
            sl = pos["stop_loss"]
            t1 = pos["target_1"]
            t2 = pos["target_2"]
            signal_type = pos["signal_type"]

            exit_price = None
            reason = None

            if signal_type == "BUY":
                if price <= sl:
                    exit_price, reason = sl, "STOP_LOSS"
                elif price >= t2:
                    exit_price, reason = t2, "TARGET_2"
                elif price >= t1:
                    # Partial exit: trail SL to breakeven
                    entry = pos["entry_price"]
                    self._portfolio.update_stop_loss(trade_id, entry)
            elif signal_type == "SELL":
                if price >= sl:
                    exit_price, reason = sl, "STOP_LOSS"
                elif price <= t2:
                    exit_price, reason = t2, "TARGET_2"

            if exit_price and reason:
                pnl = self._portfolio.close_position(trade_id, exit_price, reason)
                closed.append({"symbol": symbol, "reason": reason, "pnl": pnl})

        return closed

    # ── BaseBroker interface ────────────────────────────────────────────────

    def place_order(self, trade_id: str, symbol: str, transaction_type: str,
                    quantity: int, price: float, order_type: str = "LIMIT") -> Optional[str]:
        order_id = str(uuid.uuid4())[:8]
        self._orders[order_id] = {
            "order_id": order_id, "trade_id": trade_id, "symbol": symbol,
            "type": transaction_type, "qty": quantity, "price": price,
            "status": "COMPLETE",
        }
        _logger.info(f"[PAPER] {transaction_type} {symbol} x{quantity} @ ₹{price:.2f} [{order_id}]")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> dict:
        return self._orders.get(order_id, {"status": "NOT_FOUND"})

    def get_positions(self) -> list:
        return self._portfolio.get_open_positions()

    def get_funds(self) -> dict:
        stats = self._portfolio.get_performance_stats()
        initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        open_pos = self._portfolio.get_open_positions()
        invested = sum(p["entry_price"] * p["quantity"] for p in open_pos)
        return {
            "available_cash": round(stats.get("current_equity", initial) - invested, 2),
            "total_equity": stats.get("current_equity", initial),
            "invested": round(invested, 2),
        }

    def _simulate_fill(self, price: float, signal_type: str) -> float:
        """Add realistic slippage to fill price."""
        slip = random.uniform(0, _SLIPPAGE_PCT)
        if signal_type == "BUY":
            return round(price * (1 + slip), 2)
        return round(price * (1 - slip), 2)
