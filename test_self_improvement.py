"""
Test script for the self-improvement pipeline.

Seeds the database with fake closed trades at known win rates, then runs
PerformanceTracker and AutoTuner so you can see the full feedback loop
without waiting for real trades to accumulate.

Usage:
  python test_self_improvement.py          # Seed + run + show results
  python test_self_improvement.py --clean  # Also delete tuning_state.json first (fresh run)
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

_DB_PATH = Path("data/trading.db")
_TUNING_PATH = Path("data/tuning_state.json")


# ── Seed data ──────────────────────────────────────────────────────────────

def seed_fake_trades():
    """Insert fake closed trades so performance tracker has data to work with."""
    import sqlite3

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()

    # Make sure trades table exists (PortfolioAgent will have created it, but just in case)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_1 REAL NOT NULL,
            target_2 REAL NOT NULL,
            quantity INTEGER NOT NULL,
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            pnl REAL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            exit_reason TEXT,
            confidence REAL,
            claude_verdict TEXT
        )
    """)

    now = datetime.now()
    inserted = 0

    # Strategy scenarios:
    #   intraday  → 3/12 wins (25% WR) → should RAISE threshold (below 40%)
    #   swing     → 9/13 wins (69% WR) → should LOWER threshold (above 65%)
    #   positional → 6/10 wins (60% WR) → no change (within 40-65%)

    scenarios = [
        # (strategy, symbol, win, entry, stop, target1, target2, qty)
        *[("intraday", "TATASTEEL", True,  110.0, 106.0, 115.0, 120.0, 100)] * 3,
        *[("intraday", "HDFCBANK",  False, 1600.0, 1560.0, 1650.0, 1700.0, 10)] * 9,
        *[("swing",    "RELIANCE",  True,  2400.0, 2320.0, 2520.0, 2640.0, 5)] * 9,
        *[("swing",    "INFY",      False, 1450.0, 1410.0, 1510.0, 1570.0, 15)] * 4,
        *[("positional", "TCS",     True,  3700.0, 3570.0, 3900.0, 4100.0, 3)] * 6,
        *[("positional", "WIPRO",   False, 480.0,  460.0,  510.0,  540.0,  20)] * 4,
    ]

    for i, (strategy, symbol, win, entry, stop, t1, t2, qty) in enumerate(scenarios):
        # Tag every seeded row with a "seed-" id prefix so cleanup can remove them
        # exactly, without guessing from symbol/exit_reason (a brittle match that
        # silently failed before and left fake trades polluting the live stats).
        trade_id = "seed-" + str(uuid.uuid4())
        entry_time = (now - timedelta(days=30 - i)).isoformat()
        exit_time = (now - timedelta(days=29 - i)).isoformat()
        exit_price = t1 if win else stop
        pnl = (exit_price - entry) * qty if win else (stop - entry) * qty
        cur.execute("""
            INSERT OR IGNORE INTO trades
              (id, symbol, signal_type, strategy, entry_price, stop_loss,
               target_1, target_2, quantity, status, exit_price, pnl,
               entry_time, exit_time, exit_reason, confidence, claude_verdict)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade_id, symbol, "BUY", strategy, entry, stop, t1, t2, qty,
            "CLOSED", exit_price, round(pnl, 2),
            entry_time, exit_time,
            "target_1" if win else "stop_loss",
            7.5, "APPROVE"
        ))
        inserted += 1

    conn.commit()
    conn.close()
    console.print(f"[green]Seeded {inserted} fake closed trades into {_DB_PATH}[/]")


# ── Run pipeline ───────────────────────────────────────────────────────────

def run_test():
    from agents.portfolio_agent import PortfolioAgent
    from agents.performance_tracker import PerformanceTracker
    from agents.auto_tuner import AutoTuner

    portfolio = PortfolioAgent()
    tracker = PerformanceTracker()
    tuner = AutoTuner()

    # 1. Load trade history
    df = portfolio.get_trade_history(limit=500)
    console.print(f"\n[cyan]Loaded {len(df)} closed trades from database[/]")

    # 2. Compute strategy stats
    strategy_stats = tracker.compute_strategy_stats(df)
    symbol_stats = tracker.compute_symbol_stats(df)

    # Show strategy stats
    strat_table = Table(title="Strategy Performance", show_header=True)
    strat_table.add_column("Strategy", style="cyan")
    strat_table.add_column("Trades", justify="right")
    strat_table.add_column("Wins", justify="right")
    strat_table.add_column("Win Rate", justify="right")
    strat_table.add_column("Avg PnL", justify="right")
    strat_table.add_column("Expected Action")

    for strat, s in sorted(strategy_stats.items()):
        wr = s["win_rate"]
        if wr < 40:
            action = "[red]RAISE threshold[/]"
        elif wr > 65:
            action = "[green]LOWER threshold[/]"
        else:
            action = "[dim]No change[/]"
        strat_table.add_row(
            strat, str(s["total_trades"]), str(s["wins"]),
            f"{wr}%", f"Rs.{s['avg_pnl']:,.0f}", action
        )
    console.print(strat_table)

    # 3. Build AI context
    ai_context = tracker.build_ai_context(strategy_stats, symbol_stats)
    console.print(Panel(ai_context or "(no context — insufficient data)", title="AI Context String", border_style="blue"))

    # 4. Run auto-tuner
    console.print("\n[cyan]Running AutoTuner...[/]")
    actions = tuner.run(strategy_stats, ai_context)

    if actions:
        action_table = Table(title="Threshold Adjustments Applied", show_header=True)
        action_table.add_column("Strategy", style="cyan")
        action_table.add_column("Before", justify="right")
        action_table.add_column("After", justify="right")
        action_table.add_column("Reason")
        for a in actions:
            color = "red" if a["after"] > a["before"] else "green"
            action_table.add_row(
                a["strategy"],
                str(a["before"]),
                f"[{color}]{a['after']}[/]",
                a["reason"],
            )
        console.print(action_table)
    else:
        console.print("[yellow]AutoTuner: no threshold changes (all strategies in healthy range or insufficient data)[/]")

    # 5. Show tuning_state.json
    if _TUNING_PATH.exists():
        with open(_TUNING_PATH) as f:
            state = json.load(f)
        console.print(Panel(
            json.dumps(state, indent=2),
            title=f"[bold]{_TUNING_PATH}[/]",
            border_style="dim",
        ))

    # 6. Verify threshold overrides are read back by ClaudeValidationAgent
    console.print("\n[cyan]Verifying ClaudeValidationAgent loads overrides...[/]")
    from agents.claude_validation_agent import ClaudeValidationAgent
    agent = ClaudeValidationAgent()
    if agent._threshold_overrides:
        for strat, val in agent._threshold_overrides.items():
            console.print(f"  {strat}: threshold now [bold]{val}[/] (default 6.0)")
    else:
        console.print("  [dim]No overrides loaded (expected if no adjustments were made)[/]")

    console.print("\n[bold green]Test complete.[/] Open the dashboard → '📈 System Health' tab to see the full UI.")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Delete tuning_state.json and seeded trades, then run")
    parser.add_argument("--clean-only", action="store_true", help="Remove seeded test data and exit — no re-seed")
    args = parser.parse_args()

    def _cleanup():
        if _TUNING_PATH.exists():
            _TUNING_PATH.unlink()
            console.print(f"[yellow]Deleted {_TUNING_PATH}[/]")
        if _DB_PATH.exists():
            import sqlite3
            conn = sqlite3.connect(_DB_PATH)
            # Seeded rows carry a "seed-" id prefix — match on that alone so we can
            # never delete a real trade, regardless of its symbol or exit_reason.
            conn.execute("DELETE FROM partial_exits WHERE trade_id LIKE 'seed-%'")
            deleted = conn.execute("DELETE FROM trades WHERE id LIKE 'seed-%'").rowcount
            conn.commit()
            conn.close()
            console.print(f"[yellow]Removed {deleted} seeded test trades from database[/]")

    if args.clean_only:
        _cleanup()
        console.print("[green]Done. Your real portfolio data is untouched.[/]")
    else:
        if args.clean:
            _cleanup()
        console.print("[bold]Seeding fake trades...[/]")
        seed_fake_trades()
        console.print("\n[bold]Running self-improvement pipeline...[/]\n")
        run_test()
