import sqlite3
conn = sqlite3.connect('data/trading.db')
c = conn.cursor()
# Closed trades with PnL
c.execute("SELECT symbol, strategy, pnl, exit_reason, confidence FROM trades WHERE status='CLOSED' ORDER BY exit_time DESC")
rows = c.fetchall()
print("Closed trades (symbol, strategy, pnl, exit_reason, confidence):")
for r in rows:
    print(" ", r)

# Check if tuning_state.json exists
import json, os
tf = 'data/tuning_state.json'
if os.path.exists(tf):
    with open(tf) as f:
        state = json.load(f)
    print("\ntuning_state.json:", json.dumps(state, indent=2))
else:
    print("\ntuning_state.json: NOT FOUND")
conn.close()
