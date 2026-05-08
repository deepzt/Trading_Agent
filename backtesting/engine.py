"""
Backtesting engine using vectorbt.
Supports multi-symbol, multi-strategy backtests with realistic assumptions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from agents.data_agent import DataAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from monitoring.logger import get_logger

_logger = get_logger("Backtester")


class BacktestResult:
    def __init__(self, portfolio: vbt.Portfolio, symbol: str, strategy: str):
        self.symbol = symbol
        self.strategy = strategy
        self.portfolio = portfolio

    @property
    def total_return_pct(self) -> float:
        return round(self.portfolio.total_return() * 100, 2)

    @property
    def sharpe(self) -> float:
        try:
            return round(float(self.portfolio.sharpe_ratio()), 2)
        except Exception:
            return 0.0

    @property
    def max_drawdown_pct(self) -> float:
        try:
            return round(float(self.portfolio.max_drawdown()) * 100, 2)
        except Exception:
            return 0.0

    @property
    def win_rate(self) -> float:
        try:
            trades = self.portfolio.trades.records_readable
            if trades.empty:
                return 0.0
            wins = (trades["PnL"] > 0).sum()
            return round(wins / len(trades) * 100, 1)
        except Exception:
            return 0.0

    @property
    def total_trades(self) -> int:
        try:
            return len(self.portfolio.trades.records_readable)
        except Exception:
            return 0

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "total_return_pct": self.total_return_pct,
            "sharpe": self.sharpe,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
        }

    def equity_curve(self) -> pd.Series:
        return self.portfolio.value()


class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self._data_agent = DataAgent()
        self._ta_agent = TechnicalAnalysisAgent()

    def run(self, symbols: List[str], strategy: str = "swing",
            start_date: Optional[str] = None, end_date: Optional[str] = None,
            days: int = 365) -> List[BacktestResult]:
        """
        Run backtest for given symbols and strategy.
        Returns list of BacktestResult objects.
        """
        if start_date is None:
            start = datetime.now() - timedelta(days=days)
            start_date = start.strftime("%Y-%m-%d")

        _logger.info(f"Backtest: {strategy} | {len(symbols)} symbols | from {start_date}")

        raw = self._data_agent.run(symbols, timeframe="1d", days=days + 50)
        enriched = self._ta_agent.run(raw)

        results = []
        for symbol, df in enriched.items():
            try:
                result = self._backtest_symbol(symbol, df, strategy)
                if result:
                    results.append(result)
                    _logger.info(f"{symbol}: return={result.total_return_pct}% sharpe={result.sharpe}")
            except Exception as e:
                _logger.error(f"Backtest failed for {symbol}: {e}")

        return results

    def _backtest_symbol(self, symbol: str, df: pd.DataFrame, strategy: str) -> Optional[BacktestResult]:
        entries, exits = self._get_signals(df, strategy)
        if entries is None or entries.sum() == 0:
            return None

        close = df["close"]
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission_pct,
            sl_stop=0.03,    # 3% stop-loss
            tp_stop=0.06,    # 6% take-profit (T2)
            freq="D",
        )
        return BacktestResult(portfolio, symbol, strategy)

    def _get_signals(self, df: pd.DataFrame, strategy: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        if strategy == "swing":
            return self._swing_signals(df)
        elif strategy == "momentum":
            return self._momentum_signals(df)
        elif strategy == "mean_reversion":
            return self._mean_reversion_signals(df)
        elif strategy == "positional":
            return self._positional_signals(df)
        return None, None

    def _swing_signals(self, df: pd.DataFrame):
        ema9 = df.get("EMA_9")
        ema21 = df.get("EMA_21")
        rsi = df.get("RSI_14")
        if ema9 is None or ema21 is None or rsi is None:
            return None, None
        entries = (
            (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1)) &  # EMA cross
            (rsi >= 40) & (rsi <= 65) &                             # RSI sweet spot
            (df["close"] > df.get("EMA_50", df["close"]))           # Above EMA50
        )
        exits = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))  # EMA bearish cross
        return entries, exits

    def _momentum_signals(self, df: pd.DataFrame):
        close = df["close"]
        returns_6m = close.pct_change(126)
        entries = returns_6m > returns_6m.rolling(252).quantile(0.75)
        exits = returns_6m < 0
        return entries, exits

    def _mean_reversion_signals(self, df: pd.DataFrame):
        bb_lower = df.get("BBL_20_2.0")
        bb_upper = df.get("BBU_20_2.0")
        if bb_lower is None:
            return None, None
        entries = df["close"] < bb_lower  # Price below lower BB → oversold
        exits = df["close"] > bb_upper    # Price above upper BB → overbought
        return entries, exits

    def _positional_signals(self, df: pd.DataFrame):
        ema200 = df.get("EMA_200")
        rsi = df.get("RSI_14")
        if ema200 is None or rsi is None:
            return None, None
        entries = (df["close"] > ema200) & (rsi >= 45) & (rsi <= 70)
        exits = (df["close"] < ema200) | (rsi > 80)
        return entries, exits

    def walk_forward(self, symbol: str, strategy: str = "swing",
                     train_days: int = 180, test_days: int = 30,
                     total_days: int = 365) -> List[dict]:
        """Walk-forward validation: train on N days, test on M days, roll forward."""
        _logger.info(f"Walk-forward: {symbol} | {strategy}")
        raw = self._data_agent.run([symbol], timeframe="1d", days=total_days + 50)
        enriched = self._ta_agent.run(raw)
        df = enriched.get(symbol)
        if df is None or len(df) < train_days + test_days:
            return []

        results = []
        idx = 0
        while idx + train_days + test_days <= len(df):
            test_df = df.iloc[idx + train_days: idx + train_days + test_days]
            try:
                entries, exits = self._get_signals(test_df, strategy)
                if entries is not None and entries.sum() > 0:
                    pf = vbt.Portfolio.from_signals(
                        close=test_df["close"], entries=entries, exits=exits,
                        init_cash=self.initial_capital, fees=self.commission_pct, freq="D",
                    )
                    results.append({
                        "period_start": str(test_df.index[0].date()),
                        "period_end": str(test_df.index[-1].date()),
                        "return_pct": round(pf.total_return() * 100, 2),
                        "trades": len(pf.trades.records_readable),
                    })
            except Exception:
                pass
            idx += test_days

        return results

    def compare_strategies(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Run all strategies on a symbol and return comparison table."""
        strategies = ["swing", "momentum", "mean_reversion", "positional"]
        rows = []
        for strategy in strategies:
            results = self.run([symbol], strategy=strategy, days=days)
            if results:
                rows.append(results[0].summary())
        return pd.DataFrame(rows) if rows else pd.DataFrame()
