"""Unit tests for paper broker and portfolio agent."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agents.portfolio_agent import PortfolioAgent
from agents.signal_agent import Signal
from brokers.paper_broker import PaperBroker


@pytest.fixture
def temp_portfolio(tmp_path, monkeypatch):
    """PortfolioAgent backed by a temp SQLite database."""
    db_path = tmp_path / "test_trading.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ACCOUNT_SIZE", "100000")
    return PortfolioAgent()


@pytest.fixture
def paper_broker(temp_portfolio):
    return PaperBroker(temp_portfolio)


def _make_signal() -> Signal:
    return Signal(
        symbol="RELIANCE", signal_type="BUY", strategy="swing",
        entry_price=2845.50, stop_loss=2790.00,
        target_1=2900.00, target_2=2960.00,
        confidence=8.0, reasons=["EMA cross", "RSI ok"]
    )


class TestPaperBroker:
    def test_execute_signal_opens_position(self, paper_broker, temp_portfolio):
        sig = _make_signal()
        order_id = paper_broker.execute_signal(sig, quantity=10)
        assert order_id is not None
        positions = temp_portfolio.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "RELIANCE"
        assert positions[0]["quantity"] == 10

    def test_fill_price_has_slippage(self, paper_broker, temp_portfolio):
        sig = _make_signal()
        paper_broker.execute_signal(sig, quantity=10)
        pos = temp_portfolio.get_open_positions()[0]
        # Fill price should be >= entry (slippage adds cost for BUY)
        assert pos["entry_price"] >= sig.entry_price

    def test_check_exits_stop_loss(self, paper_broker, temp_portfolio):
        sig = _make_signal()
        paper_broker.execute_signal(sig, quantity=10)
        # Simulate price hitting stop loss
        closed = paper_broker.check_exits({"RELIANCE": 2780.0})  # Below SL of 2790
        assert len(closed) == 1
        assert closed[0]["reason"] == "STOP_LOSS"
        assert closed[0]["pnl"] < 0  # Loss on stop-loss hit

    def test_check_exits_partial_scale_out(self, paper_broker, temp_portfolio):
        """Price above both targets scales out in stages: T1 partial (50%) first tick,
        T2 partial (25%) next tick, leaving a 25% runner trailed to T1."""
        sig = _make_signal()
        paper_broker.execute_signal(sig, quantity=10)

        # Tick 1: books 50% (5 shares) at T1, trails SL to breakeven
        closed = paper_broker.check_exits({"RELIANCE": 2970.0})  # Above T2 of 2960
        assert len(closed) == 1
        assert closed[0]["reason"] == "TARGET_1_PARTIAL"
        assert closed[0]["pnl"] > 0
        pos = temp_portfolio.get_open_positions()[0]
        assert pos["quantity"] == 5          # 5 booked, 5 remain
        assert pos["t1_booked"] == 1

        # Tick 2: books 25% (2 shares) at T2, trails SL to T1 — 3-share runner remains
        closed = paper_broker.check_exits({"RELIANCE": 2970.0})
        assert len(closed) == 1
        assert closed[0]["reason"] == "TARGET_2_PARTIAL"
        pos = temp_portfolio.get_open_positions()[0]
        assert pos["quantity"] == 3
        assert pos["t2_booked"] == 1
        assert pos["stop_loss"] == pytest.approx(2900.0)  # trailed up to T1

        # Runner exits when price falls back to the trailed stop (now at T1)
        closed = paper_broker.check_exits({"RELIANCE": 2895.0})
        assert len(closed) == 1
        assert closed[0]["reason"] == "STOP_LOSS"
        assert len(temp_portfolio.get_open_positions()) == 0

    def test_no_exit_when_price_between_sl_and_t1(self, paper_broker, temp_portfolio):
        sig = _make_signal()
        paper_broker.execute_signal(sig, quantity=10)
        closed = paper_broker.check_exits({"RELIANCE": 2850.0})  # Between SL and T2
        assert len(closed) == 0  # No exit triggered


class TestPortfolioAgent:
    def test_performance_stats_empty(self, temp_portfolio):
        stats = temp_portfolio.get_performance_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_open_close_position(self, temp_portfolio):
        temp_portfolio.open_position(
            trade_id="test-1", symbol="TCS", signal_type="BUY", strategy="swing",
            entry_price=4000.0, stop_loss=3920.0, target_1=4080.0, target_2=4160.0,
            quantity=5, confidence=8.0
        )
        positions = temp_portfolio.get_open_positions()
        assert len(positions) == 1

        pnl = temp_portfolio.close_position("test-1", 4100.0, "TARGET_1")
        # (4100-4000)*5 = 500 gross, less realistic round-trip transaction costs
        from brokers.costs import round_trip_cost
        expected = 500.0 - round_trip_cost(4000.0, 4100.0, 5, "BUY", is_intraday=False)
        assert pnl == pytest.approx(expected, rel=0.01)
        assert 440.0 < pnl < 500.0

        positions = temp_portfolio.get_open_positions()
        assert len(positions) == 0

    def test_win_rate_calculation(self, temp_portfolio):
        for i, (exit_p, entry_p) in enumerate([(4100, 4000), (3900, 4000), (4200, 4000)]):
            temp_portfolio.open_position(
                trade_id=f"t{i}", symbol="TCS", signal_type="BUY", strategy="swing",
                entry_price=float(entry_p), stop_loss=3920.0, target_1=4080.0, target_2=4160.0,
                quantity=1, confidence=7.0
            )
            temp_portfolio.close_position(f"t{i}", float(exit_p), "test")

        stats = temp_portfolio.get_performance_stats()
        assert stats["total_trades"] == 3
        assert stats["win_rate"] == pytest.approx(66.7, rel=0.05)  # 2 wins out of 3

    def test_get_funds_from_broker(self, paper_broker, temp_portfolio):
        funds = paper_broker.get_funds()
        assert "available_cash" in funds
        assert "total_equity" in funds
        assert funds["total_equity"] == pytest.approx(100_000.0, rel=0.01)
