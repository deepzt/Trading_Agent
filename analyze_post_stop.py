"""
MFE / post-stop recovery analysis — a counterfactual stop-widening study.

Realized-winner MAE (analyze_mae.py / the live AutoTuner) can only TIGHTEN stops:
a winner that dipped past its stop would already be a stopped-out loser, so winners'
MAE is bounded below the current multiplier. To learn whether stops are too TIGHT —
how often a stop-out was premature and the trade would have recovered to its target —
we must look at what price did AFTER the stop was hit. That is this script.

For every trade that exited via STOP_LOSS it reconstructs daily OHLC over a
strategy-appropriate holding window, removes the stop (counterfactual), and measures:
  - whether price later reached Target-1 within the window (a premature stop),
  - Maximum Favorable / Adverse Excursion (MFE / MAE) in ATR units,
  - an EXPECTANCY SWEEP over candidate wider stop multipliers to find the one that
    maximizes net P&L across the stopped trades (targets rescued minus the extra loss
    on trades that still fail).

Honest limitations — read before acting:
  - Counterfactual assumes the trade is held to Target-1 (or window end) with the
    original target; it ignores partial T1/T2 scale-outs, time/indicator exits and
    transaction costs, so it APPROXIMATES expectancy.
  - Daily bars hide intrabar order; on a bar that touches both the wider stop and the
    target we conservatively assume the STOP filled first (never over-credits widening).
  - Intraday trades are EOD-closed, so daily-bar recovery isn't meaningful for them.
  - Small samples make this directional, not a tuned constant.

Usage:
    python analyze_post_stop.py
    python analyze_post_stop.py --strategy swing
    python analyze_post_stop.py --max-mult 3.0 --min-trades 8
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from agents.data_agent import DataAgent
# Reuse the exact OHLC date-parsing and ATR helpers used by the MAE analysis.
from analyze_mae import _parse_ist, _atr_at, CURRENT_MULT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = "data/trading.db"

# Holding window per strategy, in trading bars (mirrors the time-stops in the system).
MAX_HOLD = {"intraday": 1, "swing": 21, "positional": 84, "mean_reversion": 10}


def _load_window(da: DataAgent, t: dict, max_hold: int):
    """Return (window_df, atr, entry) for the holding window after entry, or
    (None, None, reason)."""
    entry_dt = _parse_ist(t["entry_time"])
    if entry_dt is None:
        return None, None, "unparseable entry_time"
    # +60 calendar days of pre-entry buffer so ATR_14 has enough warmup bars.
    df = da.get_historical_ohlcv(t["symbol"], "1d", days=(datetime.now() - entry_dt).days + 60)
    if df is None or df.empty:
        return None, None, "no OHLC data"
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    atr = _atr_at(df, entry_dt)
    if atr is None:
        return None, None, "insufficient history for ATR"

    fwd = df[df.index.normalize() >= pd.Timestamp(entry_dt).normalize()].iloc[: max_hold + 1]
    if fwd.empty:
        return None, None, "no bars in holding window"

    bar = fwd.iloc[0]
    if not (bar["low"] * 0.85 <= t["entry_price"] <= bar["high"] * 1.15):
        return None, None, "entry price mismatch (seeded/synthetic?)"
    return fwd, atr, t["entry_price"]


def _analyze_trade(window: pd.DataFrame, atr: float, entry: float, t: dict):
    """Per-trade counterfactual primitives. BUY-side (the system is long-only)."""
    lows = window["low"].to_numpy()
    highs = window["high"].to_numpy()
    closes = window["close"].to_numpy()
    t1 = t["target_1"]

    # First bar (counterfactual: no stop) that reaches Target-1.
    t1_hits = np.where(highs >= t1)[0]
    i_t1 = int(t1_hits[0]) if len(t1_hits) else None

    return {
        "lows": lows, "highs": highs, "closes": closes, "entry": entry,
        "atr": atr, "t1": t1, "i_t1": i_t1,
        "reached_t1": i_t1 is not None,
        "mfe_atr": (highs.max() - entry) / atr,
        "mae_atr": (entry - lows.min()) / atr,
        "as_traded_mult": (entry - t["stop_loss"]) / atr,  # the stop actually used
    }


def _outcome_atr(a: dict, mult: float) -> float:
    """Counterfactual P&L (in ATR units) if the stop were at `mult`×ATR, holding to
    Target-1 or window end. Stop-first on same-bar ties (conservative)."""
    new_stop = a["entry"] - mult * a["atr"]
    stop_hits = np.where(a["lows"] <= new_stop)[0]
    i_stop = int(stop_hits[0]) if len(stop_hits) else None
    i_t1 = a["i_t1"]
    if i_t1 is not None and (i_stop is None or i_t1 < i_stop):
        return (a["t1"] - a["entry"]) / a["atr"]   # target reached first
    if i_stop is not None:
        return -mult                                # stopped at the wider stop
    return (a["closes"][-1] - a["entry"]) / a["atr"]  # marked out at window end


def _report(name: str, analyses: list[dict], max_mult: float, min_trades: int):
    cur = CURRENT_MULT.get(name)
    n = len(analyses)
    print(f"\n{'='*70}\n{name.upper()}  —  {n} stopped trade(s)\n{'='*70}")
    if not analyses:
        print("  none.")
        return

    premature = sum(1 for a in analyses if a["reached_t1"])
    print(f"  Premature stop-outs (would have reached T1 with no stop): {premature}/{n}")
    print(f"  MFE after entry (×ATR): median {np.median([a['mfe_atr'] for a in analyses]):.2f}  "
          f"max {max([a['mfe_atr'] for a in analyses]):.2f}")
    baseline = sum(-a["as_traded_mult"] for a in analyses)
    print(f"  As-traded net (current stops): {baseline:+.2f} ATR ({baseline/n:+.2f}/trade)")

    # Expectancy sweep over candidate (wider) stop multipliers.
    lo = round(min(a["as_traded_mult"] for a in analyses), 2)
    cands = [round(m, 2) for m in np.arange(max(lo, 1.0), max_mult + 1e-9, 0.25)]
    print("\n  M(×ATR) | rescued | net(ATR) | net/trade")
    best_m, best_net, best_rescued = None, baseline, 0
    for m in cands:
        outs = [_outcome_atr(a, m) for a in analyses]
        net = sum(outs)
        rescued = sum(1 for a, o in zip(analyses, outs) if o > 0 and a["reached_t1"])
        flag = "  <= as-traded" if abs(m - (cur or -1)) < 0.01 else ""
        print(f"   {m:5.2f}  |   {rescued:3d}   | {net:+7.2f} | {net/n:+6.2f}{flag}")
        if net > best_net + 1e-6:
            best_net, best_m, best_rescued = net, m, rescued

    if name == "intraday":
        print("\n  → Intraday is EOD-closed; daily-bar recovery isn't meaningful. No recommendation.")
        return
    if n < min_trades:
        print(f"\n  → Sample too small ({n} < {min_trades}); directional only, do not auto-apply.")
    # Only recommend WIDENING when it actually rescues stop-outs to target. A net gain
    # with zero rescues is just mark-to-market mean-reversion to the time-stop on a few
    # trades — i.e. holding losers — and must NOT be sold as a stop improvement.
    if best_m is None or best_rescued == 0:
        print(f"  → None of the {n} stop-outs would have reached T1 at any tested stop "
              f"({premature}/{n} premature) — these were genuine losers, not premature stops. "
              f"Widening NOT supported here; any net change is just drift to the time-stop.")
    else:
        print(f"  → Widening {name} to ~{best_m}×ATR rescues {best_rescued}/{n} stop-outs to "
              f"T1, improving net by {best_net - baseline:+.2f} ATR "
              f"({(best_net-baseline)/n:+.2f}/trade). Validate against the cost model first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--max-mult", type=float, default=3.0, help="widest multiplier to test")
    ap.add_argument("--min-trades", type=int, default=8)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM trades WHERE status='CLOSED' AND exit_reason='STOP_LOSS'"
    params = ()
    if args.strategy:
        q += " AND strategy = ?"
        params = (args.strategy,)
    trades = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()

    if not trades:
        print("No STOP_LOSS exits to analyze.")
        return

    da = DataAgent()
    by_strat: dict[str, list] = {}
    skipped = []
    for t in trades:
        strat = t.get("strategy") or "unknown"
        window, atr, entry_or_reason = _load_window(da, t, MAX_HOLD.get(strat, 21))
        if window is None:
            skipped.append((t["symbol"], entry_or_reason))
            continue
        by_strat.setdefault(strat, []).append(_analyze_trade(window, atr, entry_or_reason, t))

    total = sum(len(v) for v in by_strat.values())
    print(f"\nAnalyzed {total} STOP_LOSS trades ({len(skipped)} skipped).")
    print("APPROXIMATE expectancy: ignores partial scale-outs, time/indicator exits, "
          "and transaction costs. Intraday recovery is not meaningful on daily bars.")

    for strat in sorted(by_strat):
        _report(strat, by_strat[strat], args.max_mult, args.min_trades)

    if skipped:
        print(f"\nSkipped ({len(skipped)}): " +
              ", ".join(f"{s}[{why}]" for s, why in skipped[:15]))


if __name__ == "__main__":
    main()
