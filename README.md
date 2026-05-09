# Trading Agent

A multi-agent algorithmic trading system for Indian stock markets (NSE/BSE) that generates precise entry and exit signals using technical analysis, validated by AI before any trade is executed — and **improves its own accuracy automatically** over time based on real trade outcomes.

> **Paper trading only** — no real money is placed until you connect a live broker.

---

## What Makes This Different

Most algo-trading systems are static. This one is not:

- Every scan, the AI validator receives a live summary of which strategies are working and which aren't
- Confidence thresholds adjust automatically — if a strategy is losing, the system becomes more selective about its signals
- Every change is logged and visible in the dashboard so you always know what the system is doing and why

---

## Features

- **AI-Validated Signals** — Every signal is reviewed by OpenAI GPT-4o or Anthropic Claude before execution, with plain-English reasoning
- **Dual AI Provider** — Works with OpenAI or Anthropic API key; auto-detects whichever is configured
- **TradingView Second Opinion** — Free TradingView TA ratings (STRONG_BUY to STRONG_SELL) added to every signal score
- **Self-Improving** — Performance feedback loop: AI validation prompt and confidence thresholds update automatically based on win/loss history
- **Multiple Strategies** — Swing (1D), intraday (15m), and positional (weekly)
- **Live Dashboard** — 8-tab Streamlit dashboard with market pulse, portfolio P&L, signals, backtesting, and system health
- **One Command Startup** — `python main.py` starts both the scheduler and dashboard together
- **Telegram Alerts** — Signal notifications and EOD P&L reports; supports multiple recipients
- **Backtesting** — Test any strategy on historical data with equity curve and drawdown analysis
- **Architecture Review** — CLI tool to evaluate proposed features against the current system design

---

## Architecture

```
LangGraph Pipeline (runs at 9:15 AM IST every trading day)
│
├── Data Agent           → Fetches OHLCV from yfinance (Nifty 50)
├── Technical Analysis   → RSI, MACD, EMA, Bollinger Bands, ATR, ADX, OBV
├── Sentiment Agent      → Google News + India VIX market context
├── TradingView Agent    → Free TA ratings per symbol (no auth required)
├── Performance Tracker  → Reads trade history; builds AI context string  ← NEW
├── Signal Agent         → Generates entry/SL/T1/T2; applies TV score adjustments
├── AI Validation        → GPT-4o/Claude reviews signal + injects performance context
├── Risk Agent           → Position sizing (2% risk/trade), daily loss limits
├── Portfolio Agent      → Tracks paper trades, P&L, Sharpe, drawdown (SQLite)
├── Notification Agent   → Telegram alerts for approved signals + EOD report
└── Auto-Tuner           → Adjusts confidence thresholds based on win rates  ← NEW
```

---

## Self-Improvement Loop

After each scan, the system automatically:

1. **Computes per-strategy win rates** from closed trade history
2. **Injects performance context** into the AI validation prompt:
   > *"Strategy performance — swing: 67% WR (18 trades, avg 1.4R). intraday: 33% WR (9 trades)."*
   The AI uses this to be more cautious about underperforming strategies.
3. **Auto-tunes confidence thresholds** — if a strategy's win rate drops below 40% (minimum 10 trades), its threshold is raised by 0.5 so only stronger signals pass. If it exceeds 65%, the threshold lowers slightly.

All adjustments are stored in `data/tuning_state.json`. Delete this file to reset to defaults instantly.

| Win Rate | Trades Required | Action |
|----------|----------------|--------|
| < 40% | ≥ 10 | Raise threshold +0.5 (max 8.5) |
| 40–65% | ≥ 10 | No change |
| > 65% | ≥ 10 | Lower threshold −0.3 (min 5.0) |

---

## Confidence Score System

| Score | Action |
|-------|--------|
| Below per-strategy threshold (default 6.0) | Rejected before reaching AI |
| At threshold, below 7.0 | Sent to AI for review but not traded |
| ≥ 7.0 + AI approval | Paper trade executed |

Thresholds adjust automatically per strategy based on performance history.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/deepzt/Trading_Agent.git
cd Trading_Agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your keys:

```env
# At least one AI key is required for signal validation
OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here   # Alternative — system auto-detects

# Optional — for Telegram signal alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id       # Comma-separated for multiple recipients

# Paper trading settings (safe defaults)
PAPER_TRADING=true
ACCOUNT_SIZE=100000
```

The system works with either OpenAI or Anthropic — whichever key is present. If both are set, Anthropic is used.

### 3. Run

```bash
# Recommended — starts dashboard + scheduler together in one command
python main.py

# Scheduler only (server / background use, no UI)
python main.py --headless

# Dashboard only (view only, no automated scans)
python main.py --dashboard

# One-shot scan and exit
python main.py --scan

# Backtest a symbol
python main.py --backtest RELIANCE --strategy swing --days 365

# View portfolio in terminal
python main.py --portfolio

# Architecture review for a proposed feature (uses AI)
python main.py --review "Add options chain screener"
```

---

## Dashboard

| Tab | Contents |
|-----|----------|
| 📡 Market Pulse | Live Nifty/Sensex/BankNifty, sector heatmap, top movers |
| 💼 Portfolio | Equity curve, drawdown chart, win rate, Sharpe, open positions |
| 📊 Technical Chart | Candlestick + RSI + MACD + Volume for any symbol |
| 🔔 Live Signals | All signals with entry/SL/T1/T2, TV rating, and AI reasoning |
| 🔬 Backtesting | Run and compare strategies on historical data |
| 📋 Watchlist | RSI heatmap and indicator snapshot for Nifty 50 |
| ⚙️ Settings | API key status, risk config, Telegram test button, quick commands |
| 📈 System Health | Strategy win rates, auto-tuning status, AI context preview, improvement history |

### System Health Tab

The **System Health** tab shows everything the self-improvement loop is doing:

- **Strategy performance cards** — win rate per strategy, colour-coded green/yellow/red
- **Auto-tuning table** — current threshold vs default, with the reason for each adjustment
- **AI context preview** — the exact performance summary being sent to GPT-4o/Claude on every validation call
- **Symbol leaderboard** — best and worst performing symbols by win rate
- **Improvement history** — timeline of every threshold change with date and reason

---

## Telegram Setup

1. Open Telegram → search **@BotFather** → send `/newbot`
2. Copy the token → set as `TELEGRAM_BOT_TOKEN` in `.env`
3. Send any message to your new bot, then open:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find `"chat":{"id": 123456789}` → set as `TELEGRAM_CHAT_ID`

**Multiple recipients** — comma-separate chat IDs, or use a group/channel ID (negative number):
```env
TELEGRAM_CHAT_ID=123456789,987654321
```

---

## Testing the Self-Improvement Feature

The self-improvement loop only activates once real closed trades accumulate. To test it immediately with synthetic data:

```bash
# Seed fake trades + run the full feedback pipeline + show results
python test_self_improvement.py

# Same, but clear previous test state first (clean run)
python test_self_improvement.py --clean

# Remove all seeded test data from the database (restore real state)
python test_self_improvement.py --clean-only
```

The test seeds 35 synthetic trades across three strategies at known win rates, runs the Performance Tracker and Auto-Tuner, and prints a table showing exactly what changed and why. After testing, always run `--clean-only` to remove the synthetic trades before using the dashboard normally.

---

## Signal Format

Every approved signal contains:

```
BUY Signal: RELIANCE
Strategy:   Swing | Confidence: 8.2/10 | TradingView: BUY
Entry:      Rs.2,845
Stop Loss:  Rs.2,790  (ATR-based, 1.5x ATR)
Target 1:   Rs.2,900  (1:1 risk-reward — exit 50%)
Target 2:   Rs.2,960  (1:2 risk-reward — exit remaining 50%)
AI Verdict: Strong EMA 9/21 crossover on above-average volume...
```

---

## Scheduled Jobs (runs automatically when `python main.py` is running)

| Time (IST) | Job |
|------------|-----|
| 9:15 AM | Full signal scan — fetch data, generate signals, validate, trade |
| Every 15 min (9:30–3:15 PM) | Intraday monitor — check open positions for SL/TP hits |
| 3:45 PM | EOD report — P&L summary sent to Telegram |

All jobs check whether the market is open before executing. No action is taken on weekends or market holidays.

---

## Connecting a Live Broker

The system is built for easy broker integration:

1. Create `brokers/your_broker.py` implementing `BaseBroker`
2. Update the import in `orchestrator/workflow.py`
3. Add broker API keys to `.env`
4. Set `PAPER_TRADING=false` in `.env`

Supported broker APIs: **Zerodha (Kite Connect)**, **Upstox**, **Angel One (SmartAPI)**

> Groww does not have a public trading API.

---

## Project Structure

```
agents/           Core agents (data, TA, signal, validation, risk, portfolio, etc.)
  performance_tracker.py   Computes per-strategy/symbol win rates from trade history
  auto_tuner.py            Adjusts confidence thresholds; writes tuning_state.json
  architecture_review_agent.py  AI-powered feature proposal reviewer (CLI tool)
brokers/          Paper broker + abstract base for live brokers
orchestrator/     LangGraph workflow + APScheduler
backtesting/      vectorbt backtesting engine
dashboard/        Streamlit UI (8 tabs)
config/           Strategy params, risk limits, watchlist, market holidays
monitoring/       Structured JSON logging + market hours health checks
data/             SQLite database + tuning_state.json (auto-created)
test_self_improvement.py  Test script for the feedback loop (seeds synthetic trades)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| AI Validation | OpenAI GPT-4o or Anthropic Claude (auto-detected) |
| TA Second Opinion | TradingView (free, no auth) |
| Orchestration | LangGraph |
| Market Data | yfinance |
| Dashboard | Streamlit + Plotly |
| Database | SQLite via SQLAlchemy |
| Backtesting | vectorbt |
| Scheduling | APScheduler (IST timezone) |
| Notifications | Telegram Bot API |

---

## Disclaimer

This project is for **educational and paper trading purposes only**. It is not financial advice. Always do your own research before investing real money. Ensure compliance with SEBI regulations before deploying any algorithmic trading system with real funds.
