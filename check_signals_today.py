import sqlite3
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
today = datetime.now(IST).strftime("%Y-%m-%d")
print("Today:", today)

conn = sqlite3.connect("data/trading.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute(
    "SELECT symbol, strategy, signal_type, confidence, claude_verdict, status, tv_rating, "
    "entry_price, stop_loss, target_1, target_2, timestamp "
    "FROM signals_log WHERE timestamp LIKE ? ORDER BY timestamp DESC",
    (today + "%",)
)
rows = cur.fetchall()
print(f"Signals today: {len(rows)}\n")
for r in rows:
    d = dict(r)
    print(
        d["timestamp"][:19],
        f"| {d['symbol']:<20}",
        f"| {d['strategy']:<10}",
        f"| {d['signal_type']:<4}",
        f"| conf={d['confidence']}",
        f"| verdict={d['claude_verdict']}",
        f"| status={d['status']}",
        f"| TV={d['tv_rating']}",
        f"| entry={d['entry_price']}",
    )

# Also check open trades opened today
print("\n--- Trades opened today ---")
cur.execute(
    "SELECT symbol, strategy, signal_type, entry_price, stop_loss, target_1, target_2, quantity, confidence, status, entry_time "
    "FROM trades WHERE entry_time LIKE ? ORDER BY entry_time DESC",
    (today + "%",)
)
trades = cur.fetchall()
print(f"Trades today: {len(trades)}\n")
for t in trades:
    d = dict(t)
    print(
        d["entry_time"][:19],
        f"| {d['symbol']:<20}",
        f"| {d['strategy']:<10}",
        f"| {d['signal_type']:<4}",
        f"| qty={d['quantity']}",
        f"| entry={d['entry_price']}",
        f"| SL={d['stop_loss']}",
        f"| T1={d['target_1']}",
        f"| T2={d['target_2']}",
        f"| conf={d['confidence']}",
        f"| status={d['status']}",
    )

conn.close()
