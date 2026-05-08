"""Unit tests for risk management."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agents.risk_agent import RiskAgent
from agents.signal_agent import Signal


def _make_signal(entry=1000.0, stop=970.0, t1=1030.0, t2=1060.0, conf=8.0) -> Signal:
    return Signal(
        symbol="TESTCO", signal_type="BUY", strategy="swing",
        entry_price=entry, stop_loss=stop,
        target_1=t1, target_2=t2,
        confidence=conf, reasons=["test"]
    )


class TestRiskAgent:
    def test_position_sizing_uses_2pct_risk(self):
        """
        2% of 100k = 2000 INR risk. SL dist = 30 → 66 shares.
        Max position = 10% of 100k = 10000 INR → at entry 100 = 100 shares (not binding).
        Use a cheap stock so the max_position cap does NOT bind.
        """
        agent = RiskAgent()
        # entry=100, SL=70 → SL dist=30, qty_by_risk=2000/30=66
        # max_pos: 10000/100=100 shares → cap does not bind
        sig = _make_signal(entry=100.0, stop=70.0, t1=130.0, t2=160.0)
        qty = agent._calculate_quantity(sig)
        expected = int((100_000 * 0.02) / 30)
        assert abs(qty - expected) <= 2  # Allow rounding

    def test_quantity_at_least_one(self):
        agent = RiskAgent()
        sig = _make_signal(entry=50000.0, stop=47500.0)  # Very expensive stock
        qty = agent._calculate_quantity(sig)
        assert qty >= 1

    def test_quantity_capped_by_max_position(self):
        """Max 10% of account in one position."""
        agent = RiskAgent()
        sig = _make_signal(entry=100.0, stop=99.0)  # Very tight SL → large qty
        qty = agent._calculate_quantity(sig)
        max_value = 100_000 * 0.10
        assert qty * 100.0 <= max_value + 1  # +1 for rounding

    def test_get_position_info_structure(self):
        agent = RiskAgent()
        sig = _make_signal()
        info = agent.get_position_info(sig)
        assert "quantity" in info
        assert "risk_inr" in info
        assert "investment_inr" in info
        assert "t1_profit_inr" in info
        assert "t2_profit_inr" in info

    def test_risk_inr_is_2pct_of_account(self):
        agent = RiskAgent()
        # Use cheap stock so max_position_pct cap doesn't override the 2% risk rule
        sig = _make_signal(entry=100.0, stop=70.0, t1=130.0, t2=160.0)
        info = agent.get_position_info(sig)
        # Risk should be close to 2% of 100k = 2000 INR
        assert abs(info["risk_inr"] - 2000) <= 150

    def test_trading_disabled_blocks_signals(self, monkeypatch):
        monkeypatch.setenv("TRADING_ENABLED", "false")
        agent = RiskAgent()
        sig = _make_signal()
        result = agent.run([sig])
        assert result == []
