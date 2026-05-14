"""Manually close stale intraday positions at last available market price."""
import sqlite3
from datetime import datetime
import pytz
import yfinance as yf

IST = pytz.timezone("Asia/Kolkata")

conn = sqlite3.connect("data/trading.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute(
    "SELECT id, symbol, strategy, entry_price, quantity, stop_loss, target_1, target_2 "
    "FROM trades WHERE status='OPEN' AND strategy='intraday'"
)
positions = [dict(r) for r in cur.fetchall()]

print(f"Found {len(positions)} stale intraday position(s):\n")
for p in positions:
    print(f"  {p['symbol']} | entry=Rs{p['entry_price']} | qty={p['quantity']} | id={p['id']}")

print()
for p in positions:
    sym = p["symbol"]
    trade_id = p["id"]

    # Fetch last close price
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        hist = ticker.history(period="2d")
        exit_price = float(hist["Close"].iloc[-1]) if not hist.empty else p["entry_price"]
    except Exception as e:
        print(f"  Price fetch failed for {sym}: {e} — using entry price")
        exit_price = p["entry_price"]

    pnl = (exit_price - p["entry_price"]) * p["quantity"]
    exit_time = datetime.now(IST).isoformat()

    cur.execute(
        "UPDATE trades SET status='CLOSED', exit_price=?, pnl=?, exit_time=?, exit_reason='MANUAL_EOD_CLOSE' "
        "WHERE id=? AND status='OPEN'",
        (exit_price, pnl, exit_time, trade_id)
    )
    print(f"  Closed {sym} @ Rs{exit_price:.2f} | P&L Rs{pnl:+.2f}")

conn.commit()
conn.close()
print("\nDone. Refresh the dashboard to see updated portfolio.")
