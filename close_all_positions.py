"""
Close ALL open paper-trading positions at current live market prices.
Run:  python close_all_positions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.data_agent import DataAgent
from agents.portfolio_agent import PortfolioAgent


def main():
    portfolio = PortfolioAgent()
    positions = portfolio.get_open_positions()

    if not positions:
        print("No open positions.")
        return

    print(f"Found {len(positions)} open position(s):\n")
    for p in positions:
        print(f"  {p['symbol']:15s} {p['signal_type']:4s} qty={p['quantity']:4d} "
              f"entry={p['entry_price']:.2f} SL={p['stop_loss']:.2f} "
              f"T1={p['target_1']:.2f} T2={p['target_2']:.2f} "
              f"strategy={p['strategy']} since={p['entry_time'][:10]}")

    confirm = input("\nClose ALL of these at live prices? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    data_agent = DataAgent()
    total_pnl = 0.0
    closed = 0

    for p in positions:
        sym = p["symbol"]
        quote = data_agent.get_live_quote(sym)
        if not quote:
            print(f"  {sym}: no live quote — SKIPPED")
            continue
        price = quote["last_price"]
        pnl = portfolio.close_position(p["id"], price, reason="MANUAL_CLOSE_ALL")
        if pnl is None:
            print(f"  {sym}: close failed (already closed?)")
        else:
            total_pnl += pnl
            closed += 1
            print(f"  {sym}: closed @ {price:.2f}  P&L = {pnl:+.2f}")

    print(f"\nDone. {closed}/{len(positions)} closed. Total P&L: {total_pnl:+.2f}")


if __name__ == "__main__":
    main()
