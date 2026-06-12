"""
Backtesting engine.

The simulator replays the SAME exit mechanics the live system uses so backtest
results actually transfer to production:
  - ATR-based stop (entry − atr_multiplier × ATR), T1 at 1R, T2 at 2R
  - T1 partial scale-out + stop to breakeven, T2 partial + stop trailed to T1
  - a runner that rides the T1-trailed stop
  - realistic Indian round-trip transaction costs on every chunk
Entry conditions per strategy mirror the rules screener's core indicators.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from agents.data_agent import DataAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from brokers.costs import round_trip_cost
from monitoring.logger import get_logger

_logger = get_logger("Backtester")


class BacktestResult:
    """Summary of a simulated backtest built from an explicit trade list + equity curve."""

    def __init__(self, symbol: str, strategy: str, trades: List[dict],
                 equity: pd.Series, initial_capital: float):
        self.symbol = symbol
        self.strategy = strategy
        self.trades = trades
        self._equity = equity
        self._initial = initial_capital

    @property
    def total_return_pct(self) -> float:
        if self._equity.empty:
            return 0.0
        return round((self._equity.iloc[-1] / self._initial - 1) * 100, 2)

    @property
    def sharpe(self) -> float:
        try:
            rets = self._equity.pct_change().dropna()
            if len(rets) < 5 or rets.std() == 0:
                return 0.0
            return round(float(rets.mean() / rets.std() * (252 ** 0.5)), 2)
        except Exception:
            return 0.0

    @property
    def max_drawdown_pct(self) -> float:
        try:
            peak = self._equity.cummax()
            dd = (self._equity - peak) / peak * 100
            return round(float(dd.min()), 2)
        except Exception:
            return 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        return round(wins / len(self.trades) * 100, 1)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

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
        return self._equity


class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct  # retained for API compatibility (costs now modeled)
        self._data_agent = DataAgent()
        self._ta_agent = TechnicalAnalysisAgent()
        risk_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
        with open(risk_path) as f:
            self._risk_cfg = yaml.safe_load(f)

    def run(self, symbols: List[str], strategy: str = "swing",
            start_date: Optional[str] = None, end_date: Optional[str] = None,
            days: int = 365) -> List[BacktestResult]:
        """Run backtest for given symbols and strategy. Returns list of BacktestResult."""
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
        trades, equity = self._simulate(df, entries, exits, strategy)
        return BacktestResult(symbol, strategy, trades, equity, self.initial_capital)

    def _simulate(self, df: pd.DataFrame, entries: pd.Series, exits: Optional[pd.Series],
                  strategy: str) -> Tuple[List[dict], pd.Series]:
        """Bar-by-bar replay of the live ATR-stop + T1/T2 partial + trailing exit machine."""
        risk = self._risk_cfg
        atr_mult = risk["stop_loss"]["atr_multiplier"]
        t1_rr = risk["targets"]["t1_rr_ratio"]
        t2_rr = risk["targets"]["t2_rr_ratio"]
        t1_pct = risk["targets"].get("t1_exit_pct", 50) / 100.0
        t2_pct = risk["targets"].get("t2_exit_pct", 25) / 100.0
        is_intraday = strategy == "intraday"

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr_arr = df["ATRr_14"].values if "ATRr_14" in df.columns else None
        idx = df.index
        ent = np.asarray(entries.values, dtype=bool)
        ex = np.asarray(exits.values, dtype=bool) if exits is not None else np.zeros(len(df), dtype=bool)

        n = len(df)
        equity = self.initial_capital
        marks: List[Tuple[object, float]] = []
        trades: List[dict] = []

        def book(entry, px, q):
            """Net P&L for closing q shares of a long at px, after round-trip costs."""
            return (px - entry) * q - round_trip_cost(entry, px, q, "BUY", is_intraday)

        i = 0
        while i < n:
            if not ent[i]:
                i += 1
                continue
            entry = close[i]
            if not entry or np.isnan(entry) or entry <= 0:
                i += 1
                continue
            atr = atr_arr[i] if atr_arr is not None and not np.isnan(atr_arr[i]) else entry * 0.02
            sl_dist = atr * atr_mult
            if sl_dist <= 0:
                i += 1
                continue
            t1 = entry + sl_dist * t1_rr
            t2 = entry + sl_dist * t2_rr

            orig_qty = int(equity / entry)   # all-in sizing for a single-symbol backtest
            if orig_qty < 1:
                break
            remaining = orig_qty
            realized = 0.0
            t1_booked = t2_booked = False
            cur_sl = entry - sl_dist
            reason = None

            j = i + 1
            while j < n:
                if low[j] <= cur_sl:
                    realized += book(entry, cur_sl, remaining)
                    remaining = 0
                    reason = "STOP_LOSS"
                    break
                if (not t1_booked) and high[j] >= t1:
                    chunk = min(int(round(orig_qty * t1_pct)), remaining)
                    if chunk >= remaining:
                        realized += book(entry, t1, remaining)
                        remaining = 0
                        reason = "TARGET_1"
                        break
                    if chunk >= 1:
                        realized += book(entry, t1, chunk)
                        remaining -= chunk
                    t1_booked = True
                    cur_sl = entry      # breakeven
                    j += 1
                    continue
                if t1_booked and (not t2_booked) and high[j] >= t2:
                    chunk = min(int(round(orig_qty * t2_pct)), remaining)
                    if chunk >= remaining:
                        realized += book(entry, t2, remaining)
                        remaining = 0
                        reason = "TARGET_2"
                        break
                    if chunk >= 1:
                        realized += book(entry, t2, chunk)
                        remaining -= chunk
                    t2_booked = True
                    cur_sl = t1         # trail runner up to T1
                    j += 1
                    continue
                if ex[j]:
                    realized += book(entry, close[j], remaining)
                    remaining = 0
                    reason = "INDICATOR_EXIT"
                    break
                j += 1

            if reason is None:           # ran out of data — close at last bar
                j = n - 1
                realized += book(entry, close[j], remaining)
                reason = "EOD_DATA"

            equity += realized
            trades.append({
                "entry_date": idx[i], "exit_date": idx[j],
                "entry": round(float(entry), 2), "exit_reason": reason,
                "pnl": round(float(realized), 2),
                "return_pct": round(float(realized) / (entry * orig_qty) * 100, 2),
            })
            marks.append((idx[j], equity))
            i = j + 1                    # stay flat until after the exit bar

        equity_series = pd.Series(np.nan, index=df.index, dtype=float)
        equity_series.iloc[0] = self.initial_capital
        for ts, val in marks:
            equity_series.loc[ts] = val
        equity_series = equity_series.ffill()
        return trades, equity_series

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
        return entries.fillna(False), exits.fillna(False)

    def _momentum_signals(self, df: pd.DataFrame):
        close = df["close"]
        returns_6m = close.pct_change(126)
        entries = returns_6m > returns_6m.rolling(252).quantile(0.75)
        exits = returns_6m < 0
        return entries.fillna(False), exits.fillna(False)

    def _mean_reversion_signals(self, df: pd.DataFrame):
        bb_lower = df.get("BBL_20_2.0")
        bb_upper = df.get("BBU_20_2.0")
        if bb_lower is None:
            return None, None
        entries = df["close"] < bb_lower  # Price below lower BB → oversold
        exits = df["close"] > bb_upper    # Price above upper BB → overbought
        return entries.fillna(False), exits.fillna(False)

    def _positional_signals(self, df: pd.DataFrame):
        ema200 = df.get("EMA_200")
        rsi = df.get("RSI_14")
        if ema200 is None or rsi is None:
            return None, None
        entries = (df["close"] > ema200) & (rsi >= 45) & (rsi <= 70)
        exits = (df["close"] < ema200) | (rsi > 80)
        return entries.fillna(False), exits.fillna(False)

    def walk_forward(self, symbol: str, strategy: str = "swing",
                     train_days: int = 180, test_days: int = 30,
                     total_days: int = 365) -> List[dict]:
        """Walk-forward validation: roll a test window forward, simulate on each out-of-sample slice."""
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
                    trades, equity = self._simulate(test_df, entries, exits, strategy)
                    ret = round((equity.iloc[-1] / self.initial_capital - 1) * 100, 2) if not equity.empty else 0.0
                    results.append({
                        "period_start": str(test_df.index[0].date()),
                        "period_end": str(test_df.index[-1].date()),
                        "return_pct": ret,
                        "trades": len(trades),
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
