# Trading Agent 📈

A multi-agent algorithmic trading system for Indian stock markets (NSE/BSE) that generates precise entry and exit signals using technical analysis, validated by Claude AI before any trade is executed.

> **Paper trading only** — no real money is placed until you connect a live broker.

---

## Features

- **AI-Validated Signals** — Every signal is reviewed by Claude AI (Anthropic) before execution, with plain-English reasoning
- **Precise Entry/Exit** — Each signal includes entry price, stop-loss (ATR-based), Target 1 (1:1 RR), and Target 2 (1:2 RR)
- **Multiple Strategies** — Swing, intraday, positional, and F&O options screener
- **Live Dashboard** — Real-time Streamlit dashboard with market pulse, portfolio P&L, technical charts, and backtesting
- **Auto-Scheduling** — Runs automatically during NSE market hours (9:15 AM – 3:30 PM IST)
- **Backtesting** — Test any strategy on historical data with equity curve and drawdown analysis
- **Notifications** — Telegram bot and email alerts for every signal
- **SEBI Compliant** — Full audit logging, kill switch, paper-only mode

---

## Architecture

```
LangGraph Orchestrator
│
├── Data Agent          → Fetches OHLCV from yfinance (Nifty 50 + midcap)
├── Technical Analysis  → RSI, MACD, EMA, Bollinger Bands, ATR, ADX, OBV
├── Signal Agent        → Generates entry/SL/T1/T2 for swing, intraday, positional
├── Claude Validation   → AI reviews signal quality, news context, risk-reward
├── Risk Agent          → Position sizing (2% risk/trade), daily loss limits
├── Portfolio Agent     → Tracks paper trades, P&L, Sharpe, drawdown (SQLite)
├── Sentiment Agent     → Google News + India VIX market context
└── Notification Agent  → Telegram + email signal alerts
```

---

## Confidence Score System

| Score | Action |
|-------|--------|
| < 6.0 | Rejected — never reaches Claude |
| 6.0 – 6.9 | Sent to Claude but not traded |
| ≥ 7.0 | Traded if Claude approves |

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

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=your_key_here        # Required for AI validation
TELEGRAM_BOT_TOKEN=your_token          # Optional — for signal alerts
TELEGRAM_CHAT_ID=your_chat_id          # Optional
```

### 3. Run

```bash
# Launch the dashboard (recommended)
python main.py --dashboard

# Run a one-shot signal scan
python main.py --scan

# Start the full scheduled system (auto-runs at market open)
python main.py

# Backtest a symbol
python main.py --backtest RELIANCE
```

---

## Dashboard

The Streamlit dashboard includes:

| Tab | Contents |
|-----|----------|
| 📡 Market Pulse | Live Nifty/Sensex/BankNifty, sector performance, top movers |
| 💼 Portfolio | Equity curve, drawdown, win rate, Sharpe ratio, open positions |
| 📊 Technical Chart | Candlestick + RSI + MACD + Volume for any symbol |
| 🔔 Live Signals | All signals with entry/SL/T1/T2 and Claude's reasoning |
| 🔬 Backtesting | Run and compare strategies on historical data |
| 📋 Watchlist | RSI heatmap and indicator snapshot for Nifty 50 |
| ⚙️ Settings | System status, API key status, risk config |

---

## Signal Format

Every signal contains:

```
BUY Signal: RELIANCE
Entry:     ₹2,845
Stop Loss: ₹2,790  (ATR-based, 1.5× ATR)
Target 1:  ₹2,900  (1:1 risk-reward — exit 50%)
Target 2:  ₹2,960  (1:2 risk-reward — exit remaining 50%)
Strategy:  Swing | Confidence: 8.2/10
Claude:    Strong EMA 9/21 crossover on above-average volume...
```

---

## Connecting a Live Broker

The system is built for easy broker integration:

1. Create `brokers/your_broker.py` implementing `BaseBroker`
2. Update the import in `orchestrator/workflow.py`
3. Add your broker API keys to `.env`
4. Set `PAPER_TRADING=false` in `.env`

Brokers with public APIs: **Zerodha (Kite Connect)**, **Upstox**, **Angel One (SmartAPI)**

> Groww does not have a public trading API.

---

## Project Structure

```
agents/          Core agents (data, TA, signal, risk, portfolio, sentiment, notification)
brokers/         Paper broker + abstract base for live brokers
strategies/      Strategy logic (swing, intraday, positional, options)
orchestrator/    LangGraph workflow + APScheduler
backtesting/     vectorbt backtesting engine
dashboard/       Streamlit UI
config/          Watchlist, strategy params, risk limits, market holidays
monitoring/      Structured JSON logging + market hours health checks
tests/           pytest unit tests
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| AI Validation | Anthropic Claude (claude-sonnet-4-6) |
| Orchestration | LangGraph |
| Market Data | yfinance |
| Technical Analysis | ta library |
| Dashboard | Streamlit + Plotly |
| Database | SQLite (SQLAlchemy) |
| Backtesting | vectorbt |
| Scheduling | APScheduler (IST timezone) |
| Notifications | Telegram Bot + SMTP |

---

## Disclaimer

This project is for **educational and paper trading purposes only**. It is not financial advice. Always do your own research before investing real money. Ensure compliance with SEBI regulations before deploying any algorithmic trading system with real funds.
