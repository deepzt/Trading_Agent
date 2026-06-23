import sqlite3
conn = sqlite3.connect("data/trading.db")
cur = conn.cursor()
cur.execute("SELECT id, symbol, strategy, entry_time, status, exit_reason, entry_price, exit_price FROM trades WHERE strategy='intraday' ORDER BY entry_time DESC LIMIT 20")
rows = cur.fetchall()
print("=== Recent intraday trades ===")
for r in rows:
    print(r)
print()
cur.execute("SELECT id, symbol, strategy, entry_time, status, exit_reason FROM trades WHERE status='OPEN'")
open_rows = cur.fetchall()
print("=== Currently OPEN positions ===")
for r in open_rows:
    print(r)
conn.close()
