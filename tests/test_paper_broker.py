"""Unit tests for paper broker and portfolio agent."""

import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pytz
from sqlalchemy import text

from agents.portfolio_agent import PortfolioAgent
from agents.signal_agent import Signal
from brokers.paper_broker import PaperBroker

_IST = pytz.timezone("Asia/Kolkata")


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


def _make_intraday_signal() -> Signal:
    return Signal(
        symbol="INFY", signal_type="BUY", strategy="intraday",
        entry_price=1500.0, stop_loss=1480.0,
        target_1=1520.0, target_2=1540.0,
        confidence=8.0, reasons=["breakout"]
    )


def _backdate_entry(portfolio, trade_id: str, entry_dt) -> None:
    with portfolio._engine.connect() as conn:
        conn.execute(text("UPDATE trades SET entry_time = :et WHERE id = :id"),
                     {"et": entry_dt.isoformat(), "id": trade_id})
        conn.commit()


def _freeze_now(monkeypatch, fixed_dt):
    """Pin brokers.paper_broker.datetime.now() to fixed_dt (IST-aware)."""
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_dt if tz else fixed_dt.replace(tzinfo=None)
    monkeypatch.setattr("brokers.paper_broker.datetime", _FrozenDateTime)


class TestMissedEodCloseRecovery:
    """Startup recovery for intraday positions left open by a missed 3:20 PM close."""

    def test_previous_day_position_closed_late(self, paper_broker, temp_portfolio, monkeypatch):
        paper_broker.execute_signal(_make_intraday_signal(), quantity=10)
        pos = temp_portfolio.get_open_positions()[0]
        yesterday = datetime.now(_IST) - timedelta(days=1)
        _backdate_entry(temp_portfolio, pos["id"], yesterday.replace(hour=9, minute=30))
        # Next morning, before today's 3:20 PM — stale position must still close
        _freeze_now(monkeypatch, datetime.now(_IST).replace(hour=10, minute=0))

        closed = paper_broker.close_intraday_eod({"INFY": 1505.0}, stale_only=True)
        assert len(closed) == 1
        assert closed[0]["reason"] == "EOD_CLOSE_LATE"
        assert len(temp_portfolio.get_open_positions()) == 0

    def test_todays_position_kept_before_close_time(self, paper_broker, temp_portfolio, monkeypatch):
        paper_broker.execute_signal(_make_intraday_signal(), quantity=10)
        pos = temp_portfolio.get_open_positions()[0]
        today_morning = datetime.now(_IST).replace(hour=9, minute=30)
        _backdate_entry(temp_portfolio, pos["id"], today_morning)
        _freeze_now(monkeypatch, datetime.now(_IST).replace(hour=11, minute=0))

        closed = paper_broker.close_intraday_eod({"INFY": 1505.0}, stale_only=True)
        assert len(closed) == 0
        assert len(temp_portfolio.get_open_positions()) == 1

    def test_todays_position_closed_after_close_time(self, paper_broker, temp_portfolio, monkeypatch):
        paper_broker.execute_signal(_make_intraday_signal(), quantity=10)
        pos = temp_portfolio.get_open_positions()[0]
        today_morning = datetime.now(_IST).replace(hour=9, minute=30)
        _backdate_entry(temp_portfolio, pos["id"], today_morning)
        # Restart at 3:45 PM, after the missed 3:20 close
        _freeze_now(monkeypatch, datetime.now(_IST).replace(hour=15, minute=45))

        closed = paper_broker.close_intraday_eod({"INFY": 1505.0}, stale_only=True)
        assert len(closed) == 1
        assert closed[0]["reason"] == "EOD_CLOSE_LATE"

    def test_stale_only_skips_non_intraday(self, paper_broker, temp_portfolio, monkeypatch):
        paper_broker.execute_signal(_make_signal(), quantity=10)  # swing position
        pos = temp_portfolio.get_open_positions()[0]
        yesterday = datetime.now(_IST) - timedelta(days=1)
        _backdate_entry(temp_portfolio, pos["id"], yesterday)
        _freeze_now(monkeypatch, datetime.now(_IST).replace(hour=10, minute=0))

        closed = paper_broker.close_intraday_eod({"RELIANCE": 2850.0}, stale_only=True)
        assert len(closed) == 0
        assert len(temp_portfolio.get_open_positions()) == 1


class TestTimeStopScaleOutInteraction:
    """A position hitting T1 on/after its max-hold day must scale out and keep its
    runner — the time-stop must not fire off the stale pre-scale-out t1_booked flag."""

    def test_t1_on_max_hold_day_keeps_runner(self, paper_broker, temp_portfolio):
        paper_broker.execute_signal(_make_signal(), quantity=10)  # swing, max_hold 21d
        pos = temp_portfolio.get_open_positions()[0]
        _backdate_entry(temp_portfolio, pos["id"], datetime.now(_IST) - timedelta(days=30))

        # Price above T1 (2900): books the 50% partial; runner must survive the tick
        closed = paper_broker.check_exits({"RELIANCE": 2910.0})
        assert len(closed) == 1
        assert closed[0]["reason"] == "TARGET_1_PARTIAL"
        open_positions = temp_portfolio.get_open_positions()
        assert len(open_positions) == 1          # runner NOT time-stopped
        assert open_positions[0]["quantity"] == 5
        assert open_positions[0]["t1_booked"] == 1

    def test_time_stop_still_fires_without_t1(self, paper_broker, temp_portfolio):
        paper_broker.execute_signal(_make_signal(), quantity=10)
        pos = temp_portfolio.get_open_positions()[0]
        _backdate_entry(temp_portfolio, pos["id"], datetime.now(_IST) - timedelta(days=30))

        # Price between SL and T1 — stalled trade past max hold must time-stop
        closed = paper_broker.check_exits({"RELIANCE": 2850.0})
        assert len(closed) == 1
        assert closed[0]["reason"] == "TIME_STOP"
        assert closed[0]["pnl"] is not None
        assert len(temp_portfolio.get_open_positions()) == 0


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

    def test_partial_booking_counts_in_daily_stats_and_equity(self, temp_portfolio):
        """A T1 partial booked today on a still-open trade must show up in today's
        daily P&L and in current equity — not appear out of nowhere on close day."""
        temp_portfolio.open_position(
            trade_id="p1", symbol="TCS", signal_type="BUY", strategy="swing",
            entry_price=4000.0, stop_loss=3920.0, target_1=4080.0, target_2=4160.0,
            quantity=10, confidence=8.0
        )
        booked = temp_portfolio.partial_close("p1", 5, 4080.0, "TARGET_1_PARTIAL", milestone="t1")
        assert booked is not None and booked > 0

        daily = temp_portfolio.get_daily_stats()
        assert daily["daily_pnl"] == pytest.approx(booked, abs=0.01)
        assert daily["trades_today"] == 0  # trade itself is still open

        stats = temp_portfolio.get_performance_stats()
        assert stats["current_equity"] == pytest.approx(100_000.0 + booked, abs=0.01)

        # Close the runner today too — daily P&L must equal partial + runner exactly
        runner = temp_portfolio.close_position("p1", 4160.0, "TARGET_2")
        runner_only = runner - booked  # close_position folds the partial into trade pnl
        daily = temp_portfolio.get_daily_stats()
        assert daily["daily_pnl"] == pytest.approx(booked + runner_only, abs=0.01)
        assert daily["trades_today"] == 1
        stats = temp_portfolio.get_performance_stats()
        assert stats["current_equity"] == pytest.approx(100_000.0 + runner, abs=0.01)
