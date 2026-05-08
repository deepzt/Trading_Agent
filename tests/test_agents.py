"""Unit tests for core agents."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from agents.signal_agent import Signal, SignalAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 2000 + np.cumsum(np.random.randn(n) * 20)
    high = close + abs(np.random.randn(n) * 10)
    low = close - abs(np.random.randn(n) * 10)
    open_ = close + np.random.randn(n) * 5
    volume = (np.random.randint(500_000, 2_000_000, n)).astype(int)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


class TestTechnicalAnalysisAgent:
    def test_compute_indicators_returns_dataframe(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv()
        result = agent.compute_indicators(df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_rsi_column_present(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv()
        result = agent.compute_indicators(df)
        assert "RSI_14" in result.columns

    def test_ema_columns_present(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv()
        result = agent.compute_indicators(df)
        for ema in ["EMA_9", "EMA_21", "EMA_50", "EMA_200"]:
            assert ema in result.columns, f"Missing {ema}"

    def test_atr_column_present(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv()
        result = agent.compute_indicators(df)
        assert "ATRr_14" in result.columns

    def test_insufficient_data_returns_none(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv(n=10)
        result = agent.compute_indicators(df)
        assert result is None

    def test_get_summary_returns_dict(self):
        agent = TechnicalAnalysisAgent()
        df = _make_ohlcv()
        enriched = agent.compute_indicators(df)
        summary = agent.get_summary(enriched)
        assert isinstance(summary, dict)
        assert "rsi" in summary
        assert "close" in summary


class TestSignal:
    def test_signal_creation(self):
        sig = Signal(
            symbol="RELIANCE", signal_type="BUY", strategy="swing",
            entry_price=2845.50, stop_loss=2790.00,
            target_1=2900.00, target_2=2960.00,
            confidence=7.5, reasons=["EMA cross", "RSI ok"]
        )
        assert sig.symbol == "RELIANCE"
        assert sig.risk_reward > 0
        assert sig.sl_pct > 0

    def test_risk_reward_calculation(self):
        sig = Signal(
            symbol="TCS", signal_type="BUY", strategy="swing",
            entry_price=4000.0, stop_loss=3920.0,   # SL dist = 80
            target_1=4080.0, target_2=4160.0,       # T2 dist = 160
            confidence=7.0, reasons=["test"]
        )
        assert sig.risk_reward == pytest.approx(2.0, rel=0.01)

    def test_format_alert_contains_symbol(self):
        sig = Signal(
            symbol="INFY", signal_type="BUY", strategy="swing",
            entry_price=1800.0, stop_loss=1764.0,
            target_1=1836.0, target_2=1872.0,
            confidence=8.0, reasons=["test signal"]
        )
        alert = sig.format_alert()
        assert "INFY" in alert
        assert "₹" in alert

    def test_to_dict_has_required_keys(self):
        sig = Signal(
            symbol="SBIN", signal_type="BUY", strategy="positional",
            entry_price=700.0, stop_loss=672.0,
            target_1=728.0, target_2=756.0,
            confidence=7.5, reasons=["trend", "rsi"]
        )
        d = sig.to_dict()
        required = ["id", "symbol", "signal_type", "entry_price", "stop_loss", "target_1", "target_2", "confidence"]
        for key in required:
            assert key in d, f"Missing key: {key}"


class TestSignalAgent:
    def test_run_returns_list(self):
        agent = SignalAgent()
        df = _make_ohlcv(n=300)
        ta = TechnicalAnalysisAgent()
        enriched = ta.compute_indicators(df)
        result = agent.run({"RELIANCE": enriched}, active_strategies=["swing"])
        assert isinstance(result, list)

    def test_signals_have_positive_rr(self):
        agent = SignalAgent()
        ta = TechnicalAnalysisAgent()
        df = _make_ohlcv(n=300)
        enriched = ta.compute_indicators(df)
        signals = agent.run({"TESTSTOCK": enriched}, active_strategies=["swing", "positional"])
        for sig in signals:
            assert sig.risk_reward >= 1.0, f"Bad RR: {sig.risk_reward}"

    def test_signals_have_valid_prices(self):
        agent = SignalAgent()
        ta = TechnicalAnalysisAgent()
        df = _make_ohlcv(n=300)
        enriched = ta.compute_indicators(df)
        signals = agent.run({"TESTSTOCK": enriched}, active_strategies=["swing"])
        for sig in signals:
            assert sig.entry_price > 0
            assert sig.stop_loss > 0
            assert sig.target_1 > sig.stop_loss
            assert sig.target_2 >= sig.target_1
