"""
MAE stop-loss calibration analysis.

For every closed trade, measures the Maximum Adverse Excursion (MAE) — how far the
price moved AGAINST the entry before the trade closed — expressed in ATR units, and
compares the MAE distribution of winners vs losers. The point of the comparison: a
stop is only worth widening if winners routinely dip further than the current stop
WHILE losers do not. If winners' and losers' MAE distributions overlap, stop
placement has no separating power and the recommendation says so.

MAE source per trade, in priority order:
  1. Forward-captured `mae_price` (recorded live by the intraday monitor).
  2. Reconstructed from daily OHLC over the holding window (DataAgent / yfinance).

Daily reconstruction is exact for swing/positional trades. Intraday trades older
than ~60 days cannot get true intraday bars, so their MAE uses the daily low/high as
a coarse proxy and is flagged.

Usage:
    python analyze_mae.py                 # all closed trades
    python analyze_mae.py --strategy swing
    python analyze_mae.py --min-trades 8  # suppress per-strategy reco below N trades
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from agents.data_agent import DataAgent

# Windows consoles default to cp1252, which can't encode ₹ / × / → in the report.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = "data/trading.db"

# Effective ATR stop multiplier each strategy currently uses (from signal_agent.py /
# risk_config.yaml). Used only to compare the reconstructed MAE against today's stop.
CURRENT_MULT = {"swing": 1.5, "intraday": 1.0, "positional": 2.0}


def _parse_ist(ts: str) -> datetime | None:
    """Parse an IST ISO string to a tz-naive datetime (pandas chokes on +05:30)."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=None)
    except ValueError:
        return None


def _atr_at(df: pd.DataFrame, entry_date, period: int = 14) -> float | None:
    """Wilder-style ATR on daily bars, value as of the last bar on/before entry_date."""
    upto = df[df.index.normalize() <= pd.Timestamp(entry_date).normalize()]
    if len(upto) < period + 1:
        return None
    high, low, close = upto["high"], upto["low"], upto["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if atr and atr > 0 else None


def _reconstruct_mae(da: DataAgent, t: dict):
    """Return (mae_atr, mae_pct, note) for a trade, or (None, None, reason)."""
    entry_dt = _parse_ist(t["entry_time"])
    exit_dt = _parse_ist(t["exit_time"]) or datetime.now()
    if entry_dt is None:
        return None, None, "unparseable entry_time"

    span_days = (datetime.now() - entry_dt).days + 45  # +buffer for ATR warmup
    df = da.get_historical_ohlcv(t["symbol"], "1d", days=span_days)
    if df is None or df.empty:
        return None, None, "no OHLC data"
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    atr = _atr_at(df, entry_dt)
    if atr is None:
        return None, None, "insufficient history for ATR"

    window = df[(df.index.normalize() >= pd.Timestamp(entry_dt).normalize())
                & (df.index.normalize() <= pd.Timestamp(exit_dt).normalize())]
    if window.empty:
        return None, None, "no bars in holding window"

    entry = t["entry_price"]
    # Sanity guard: the recorded entry must sit within the real market range on the
    # entry bar (allow ±15% for gaps/intrabar). Seeded/synthetic test trades carry
    # fictitious prices (e.g. RELIANCE @2846 vs real ~1435) that otherwise produce
    # absurd MAE values — skip them rather than poison the distribution.
    bar = window.iloc[0]
    if not (bar["low"] * 0.85 <= entry <= bar["high"] * 1.15):
        return None, None, "entry price mismatch (seeded/synthetic?)"
    if t["signal_type"] == "BUY":
        adverse = entry - float(window["low"].min())
    else:
        adverse = float(window["high"].max()) - entry
    adverse = max(adverse, 0.0)
    note = "intraday: daily-bar proxy" if t["strategy"] == "intraday" else ""
    return adverse / atr, adverse / entry * 100, note


def _mae_from_stored(t: dict):
    """(mae_atr, mae_pct) from the live-captured mae_price, needing an ATR estimate.
    ATR is backed out of the original stop distance: stop = entry ± mult*ATR."""
    mae_price = t.get("mae_price")
    if mae_price is None:
        return None
    entry = t["entry_price"]
    adverse = (entry - mae_price) if t["signal_type"] == "BUY" else (mae_price - entry)
    adverse = max(adverse, 0.0)
    mult = CURRENT_MULT.get(t["strategy"], 1.5)
    atr_est = abs(entry - t["stop_loss"]) / mult
    if atr_est <= 0:
        return None
    return adverse / atr_est, adverse / entry * 100


def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def _report_group(name: str, rows: list[dict], min_trades: int):
    winners = [r for r in rows if r["net"] > 0]
    losers = [r for r in rows if r["net"] <= 0]
    w_mae = [r["mae_atr"] for r in winners]
    l_mae = [r["mae_atr"] for r in losers]
    cur = CURRENT_MULT.get(name)

    print(f"\n{'='*66}\n{name.upper()}  —  {len(rows)} trades "
          f"({len(winners)} win / {len(losers)} loss)\n{'='*66}")
    if not rows:
        print("  no trades with a usable MAE.")
        return

    if w_mae:
        print(f"  Winners MAE (×ATR): median {np.median(w_mae):.2f}  "
              f"p75 {_pct(w_mae,75):.2f}  p90 {_pct(w_mae,90):.2f}  max {max(w_mae):.2f}")
    if l_mae:
        print(f"  Losers  MAE (×ATR): median {np.median(l_mae):.2f}  "
              f"p75 {_pct(l_mae,75):.2f}  p90 {_pct(l_mae,90):.2f}  max {max(l_mae):.2f}")

    if cur is not None:
        stopped = [r for r in winners if r["mae_atr"] > cur]
        print(f"  Current stop: {cur:.2f}×ATR. "
              f"Winners that dipped PAST it (premature-stop candidates): "
              f"{len(stopped)}/{len(winners)}")

    # Recommendation — only when the sample is meaningful AND stops actually separate
    # winners from losers. Otherwise report the honest 'no signal' conclusion.
    if len(winners) < min_trades:
        print(f"  → Sample too small ({len(winners)} winners < {min_trades}); "
              f"no recommendation. Keep collecting trades.")
        return
    reco = _pct(w_mae, 90)
    if l_mae and np.median(l_mae) <= reco:
        print(f"  → CAUTION: losers' median MAE ({np.median(l_mae):.2f}×) is within the "
              f"proposed stop ({reco:.2f}×). Widening keeps winners AND lets losers run — "
              f"validate on expectancy (₹/trade), not win-rate, before changing.")
    else:
        print(f"  → Candidate stop ≈ {reco:.2f}×ATR (covers 90% of winners' dips, "
              f"clears of losers). Current {cur if cur else '?'}×.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None, help="limit to one strategy")
    ap.add_argument("--min-trades", type=int, default=8,
                    help="min winners before a per-strategy stop is recommended")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM trades WHERE status != 'OPEN'"
    params = ()
    if args.strategy:
        q += " AND strategy = ?"
        params = (args.strategy,)
    trades = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()

    if not trades:
        print("No closed trades to analyze.")
        return

    da = DataAgent()
    rows, skipped, proxies = [], [], 0
    for t in trades:
        t["net"] = (t.get("pnl") or 0.0) + (t.get("realized_pnl") or 0.0)
        stored = _mae_from_stored(t)
        if stored:
            mae_atr, mae_pct = stored
            note = "live-captured"
        else:
            mae_atr, mae_pct, note = _reconstruct_mae(da, t)
        if mae_atr is None:
            skipped.append((t["symbol"], note))
            continue
        if "proxy" in note:
            proxies += 1
        rows.append({**t, "mae_atr": mae_atr, "mae_pct": mae_pct})

    print(f"\nAnalyzed {len(rows)} trades with a usable MAE "
          f"({len(skipped)} skipped, {proxies} intraday daily-proxy).")
    print("NOTE: small samples make these DIRECTIONAL, not tuned constants. "
          "Optimize on expectancy (₹/trade), not win-rate — wider stops shrink "
          "position size and enlarge losses.")

    if not args.strategy:
        _report_group("ALL", rows, args.min_trades)
    for strat in sorted({r["strategy"] for r in rows}):
        _report_group(strat, [r for r in rows if r["strategy"] == strat], args.min_trades)

    if skipped:
        print(f"\nSkipped ({len(skipped)}): " +
              ", ".join(f"{s}[{why}]" for s, why in skipped[:15]))


if __name__ == "__main__":
    main()
