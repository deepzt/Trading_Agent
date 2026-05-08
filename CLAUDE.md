# Trading Agent — Project Guide

## What This Is
A multi-agent algorithmic trading system for Indian markets (NSE/BSE) using:
- **Hybrid AI**: Rule-based technical indicators + Claude API validation
- **Paper trading first** — no real money until broker is configured
- **LangGraph orchestration** — agents flow as a directed graph

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env

# Run the Streamlit dashboard (recommended for beginners)
python main.py --dashboard

# Or run a single signal scan
python main.py --scan

# Or start the full scheduled system (runs at market open 9:15 AM IST)
python main.py
```

## Project Structure
```
agents/          Core agents (data, TA, signal, risk, portfolio, sentiment, notification)
brokers/         Paper broker + base interface for future live brokers
strategies/      Strategy helpers (intraday, swing, positional, options)
orchestrator/    LangGraph workflow + APScheduler
backtesting/     vectorbt backtesting engine
dashboard/       Streamlit user interface
config/          YAML config files + watchlist
data/            Cache + SQLite database
monitoring/      Logger + market hours health checks
tests/           pytest unit tests
```

## Key Files
- `orchestrator/workflow.py` — Main LangGraph graph + scheduler (START HERE)
- `agents/signal_agent.py` — Where signals are generated with entry/SL/TP
- `agents/claude_validation_agent.py` — Claude API integration
- `config/trading_config.yaml` — All strategy parameters
- `config/risk_config.yaml` — Risk limits (position size, drawdown, daily loss)

## Adding a Real Broker
1. Create `brokers/kite_broker.py` implementing `BaseBroker`
2. Change `from brokers.paper_broker import PaperBroker` in `orchestrator/workflow.py`
3. Add your broker API keys to `.env`
4. Set `PAPER_TRADING=false` in `.env`

## Signal Format
Every signal contains:
- `entry_price` — exact price to enter
- `stop_loss` — ATR-based (1.5× ATR from entry)
- `target_1` — 1:1 risk-reward (exit 50% here)
- `target_2` — 1:2 risk-reward (exit remaining 50% here)
- `confidence` — 0-10 (rules score averaged with Claude's score)
- `claude_verdict` — APPROVE / REJECT / MODIFY
- `claude_reasoning` — plain English explanation

## Supported Strategies
- **Swing** (`1d`): EMA 9/21 crossover + RSI 40-65 + above EMA50
- **Intraday** (`15m`): Breakout above previous day high + volume surge
- **Positional** (`1wk`): Price above 200 EMA + RSI 45-70 + MACD bullish
- **Options**: PCR screener + OI analysis (separate module)

## Running Tests
```bash
pytest tests/ -v
```

## Required API Keys (.env)
- `ANTHROPIC_API_KEY` — for Claude validation (required for AI features)
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — for signal alerts (optional)
- `EMAIL_SENDER` + `EMAIL_PASSWORD` — for email reports (optional)

## SEBI Compliance Notes
- Paper trading only until SEBI algo registration obtained
- All signals are logged with timestamp + rationale in `logs/audit.log`
- Kill switch: set `TRADING_ENABLED=false` in `.env` to halt immediately
- Max 10 paper trades per day (configurable in `risk_config.yaml`)
