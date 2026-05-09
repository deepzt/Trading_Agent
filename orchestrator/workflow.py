"""
LangGraph-based orchestrator — coordinates all agents through the daily trading workflow.
Also includes APScheduler jobs for market-hours scheduling.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, TypedDict

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from langgraph.graph import END, START, StateGraph

from agents.claude_validation_agent import ClaudeValidationAgent
from agents.data_agent import DataAgent
from agents.notification_agent import NotificationAgent
from agents.portfolio_agent import PortfolioAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.signal_agent import SignalAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.tradingview_ta_agent import TradingViewTAAgent
from brokers.paper_broker import PaperBroker
from monitoring.health import is_market_open, is_trading_day
from monitoring.logger import get_logger

_IST = pytz.timezone("Asia/Kolkata")
_logger = get_logger("Orchestrator")


# ── Shared state flowing through the graph ─────────────────────────────────

class TradingState(TypedDict):
    symbols: List[str]
    raw_data: Dict[str, Any]
    enriched_data: Dict[str, Any]
    signals: List[Any]
    validated_signals: List[Any]
    approved_trades: List[Any]
    sentiment: Dict[str, Any]
    tv_ratings: Dict[str, str]
    portfolio_stats: Dict[str, Any]
    closed_today: List[Any]
    errors: List[str]


# ── Node functions (each agent step) ──────────────────────────────────────

def fetch_data(state: TradingState) -> TradingState:
    _logger.info("Step 1: Fetching market data")
    agent = DataAgent()
    symbols = state["symbols"]
    state["raw_data"] = agent.run(symbols, timeframe="1d", days=365)
    return state


def run_technical_analysis(state: TradingState) -> TradingState:
    _logger.info("Step 2: Running technical analysis")
    agent = TechnicalAnalysisAgent()
    state["enriched_data"] = agent.run(state["raw_data"])
    return state


def fetch_sentiment(state: TradingState) -> TradingState:
    _logger.info("Step 3: Fetching sentiment and news")
    agent = SentimentAgent()
    active_symbols = list(state["enriched_data"].keys())[:20]
    state["sentiment"] = agent.run(active_symbols)
    return state


def fetch_tv_ratings(state: TradingState) -> TradingState:
    _logger.info("Step 3b: Fetching TradingView TA ratings")
    agent = TradingViewTAAgent()
    symbols = list(state["enriched_data"].keys())[:20]
    state["tv_ratings"] = agent.run(symbols)
    return state


def generate_signals(state: TradingState) -> TradingState:
    _logger.info("Step 4: Generating signals")
    agent = SignalAgent()
    state["signals"] = agent.run(state["enriched_data"], tv_ratings=state.get("tv_ratings", {}))
    return state


def validate_with_claude(state: TradingState) -> TradingState:
    _logger.info("Step 5: Claude signal validation")
    agent = ClaudeValidationAgent()
    state["validated_signals"] = agent.run(
        state["signals"], state["sentiment"], state.get("tv_ratings", {})
    )
    return state


def check_risk(state: TradingState) -> TradingState:
    _logger.info("Step 6: Risk checks and position sizing")
    portfolio = PortfolioAgent()
    agent = RiskAgent(portfolio)
    approved_with_qty = agent.run([s for s in state["validated_signals"] if s.status == "APPROVED"])

    # Build list of (signal, quantity, position_info)
    trades = []
    for signal, qty in approved_with_qty:
        pos_info = agent.get_position_info(signal)
        trades.append({"signal": signal, "quantity": qty, "position_info": pos_info})
    state["approved_trades"] = trades
    return state


def execute_paper_trades(state: TradingState) -> TradingState:
    _logger.info("Step 7: Executing paper trades")
    portfolio = PortfolioAgent()
    broker = PaperBroker(portfolio)

    for trade in state["approved_trades"]:
        signal = trade["signal"]
        qty = trade["quantity"]
        broker.execute_signal(signal, qty)

    state["portfolio_stats"] = portfolio.get_performance_stats()
    return state


def send_notifications(state: TradingState) -> TradingState:
    _logger.info("Step 8: Sending notifications")
    agent = NotificationAgent()
    portfolio = PortfolioAgent()

    for trade in state["approved_trades"]:
        signal = trade["signal"]
        qty = trade["quantity"]
        agent.send_signal_alert(signal, qty)

    # Log all signals (including rejected) for audit
    for signal in state["validated_signals"]:
        portfolio.log_signal(signal.to_dict())

    return state


# ── Graph definition ───────────────────────────────────────────────────────

def build_graph() -> Any:
    workflow = StateGraph(TradingState)

    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("run_ta", run_technical_analysis)
    workflow.add_node("fetch_sentiment", fetch_sentiment)
    workflow.add_node("fetch_tv_ratings", fetch_tv_ratings)
    workflow.add_node("generate_signals", generate_signals)
    workflow.add_node("validate_claude", validate_with_claude)
    workflow.add_node("check_risk", check_risk)
    workflow.add_node("execute_trades", execute_paper_trades)
    workflow.add_node("send_notifications", send_notifications)

    workflow.add_edge(START, "fetch_data")
    workflow.add_edge("fetch_data", "run_ta")
    workflow.add_edge("run_ta", "fetch_sentiment")
    workflow.add_edge("fetch_sentiment", "fetch_tv_ratings")
    workflow.add_edge("fetch_tv_ratings", "generate_signals")
    workflow.add_edge("generate_signals", "validate_claude")
    workflow.add_edge("validate_claude", "check_risk")
    workflow.add_edge("check_risk", "execute_trades")
    workflow.add_edge("execute_trades", "send_notifications")
    workflow.add_edge("send_notifications", END)

    return workflow.compile()


# ── Scheduler ──────────────────────────────────────────────────────────────

class TradingScheduler:
    def __init__(self, symbols: List[str]):
        self._symbols = symbols
        self._scheduler = BackgroundScheduler(timezone=_IST)
        self._graph = build_graph()

    def _run_morning_scan(self):
        if not is_market_open():
            _logger.info("Market closed — skipping morning scan")
            return
        _logger.info("=== MORNING SCAN STARTED ===")
        self._run_workflow()

    def _run_intraday_monitor(self):
        if not is_market_open():
            return
        _logger.info("Intraday monitor — checking SL/TP")
        data_agent = DataAgent()
        portfolio = PortfolioAgent()
        broker = PaperBroker(portfolio)

        live_prices = {}
        for sym in self._symbols[:20]:
            q = data_agent.get_live_quote(sym)
            if q:
                live_prices[sym] = q["last_price"]

        closed = broker.check_exits(live_prices)
        if closed:
            notif = NotificationAgent()
            for trade in closed:
                emoji = "✅" if (trade.get("pnl") or 0) >= 0 else "❌"
                notif.send_alert(
                    f"{emoji} {trade['symbol']} closed: {trade['reason']} | P&L ₹{trade.get('pnl', 0):.2f}"
                )

    def _run_eod_report(self):
        if not is_trading_day():
            return
        _logger.info("=== EOD REPORT ===")
        portfolio = PortfolioAgent()
        stats = portfolio.get_performance_stats()
        daily = portfolio.get_daily_stats()
        notif = NotificationAgent()
        notif.send_eod_report(stats, [], datetime.now(_IST).strftime("%Y-%m-%d"))
        _logger.info(f"EOD: P&L={daily['daily_pnl']} trades={daily['trades_today']}")

    def _run_workflow(self):
        initial_state: TradingState = {
            "symbols": self._symbols,
            "raw_data": {}, "enriched_data": {},
            "signals": [], "validated_signals": [],
            "approved_trades": [], "sentiment": {}, "tv_ratings": {},
            "portfolio_stats": {}, "closed_today": [], "errors": [],
        }
        try:
            self._graph.invoke(initial_state)
        except Exception as e:
            _logger.error(f"Workflow error: {e}")

    def run_now(self):
        """Run the full workflow immediately (for testing or manual trigger)."""
        self._run_workflow()

    def start(self):
        # Morning scan at 9:15 AM IST
        self._scheduler.add_job(self._run_morning_scan, "cron",
                                 hour=9, minute=15, day_of_week="mon-fri")
        # Intraday monitor every 15 minutes 9:30 AM – 3:15 PM
        self._scheduler.add_job(self._run_intraday_monitor, "cron",
                                 hour="9-15", minute="*/15", day_of_week="mon-fri")
        # EOD report at 3:45 PM
        self._scheduler.add_job(self._run_eod_report, "cron",
                                 hour=15, minute=45, day_of_week="mon-fri")
        self._scheduler.start()
        _logger.info("Trading scheduler started (IST timezone)")

    def stop(self):
        self._scheduler.shutdown()
        _logger.info("Trading scheduler stopped")
