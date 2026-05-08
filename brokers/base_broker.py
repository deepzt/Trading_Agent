from abc import ABC, abstractmethod
from typing import Optional


class BaseBroker(ABC):
    """Abstract interface for all brokers (paper and live)."""

    @abstractmethod
    def place_order(self, trade_id: str, symbol: str, transaction_type: str,
                    quantity: int, price: float, order_type: str = "LIMIT") -> Optional[str]:
        """Place an order. Returns order_id or None on failure."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Return status of an order."""

    @abstractmethod
    def get_positions(self) -> list:
        """Return all current positions."""

    @abstractmethod
    def get_funds(self) -> dict:
        """Return available margin/funds."""
