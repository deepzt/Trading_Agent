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
from agents.earnings_calendar_agent import EarningsCalendarAgent
from agents.fno_calendar_agent import FnOCalendarAgent
from agents.notification_agent import NotificationAgent
from agents.portfolio_agent import PortfolioAgent
from agents.regime_detector import RegimeDetector
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.signal_agent import SignalAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.tradingview_ta_agent import CompositeTAAgent
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
    composite_ratings: Dict[str, str]
    perf_context: str
    strategy_stats: Dict[str, Any]
    tuning_actions: List[Any]
    portfolio_stats: Dict[str, Any]
    closed_today: List[Any]
    errors: List[str]
    regime: str
    earnings_blackout_symbols: List[str]
    momentum_ranks: Dict[str, float]
    expiry_context: Dict[str, Any]
    active_strategies: List[str]


# ── Node functions (each agent step) ──────────────────────────────────────

def fetch_data(state: TradingState) -> TradingState:
    _logger.info("Step 1: Fetching market data")
    agent = DataAgent()
    symbols = state["symbols"]
    state["raw_data"] = agent.run(symbols, timeframe="1d", days=400)
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


def fetch_composite_ratings(state: TradingState) -> TradingState:
    _logger.info("Step 3b: Computing in-house composite TA ratings")
    state["composite_ratings"] = CompositeTAAgent().run(state["enriched_data"])
    return state


def detect_regime(state: TradingState) -> TradingState:
    _logger.info("Step 3c: Detecting market regime")
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        regime_cfg = cfg.get("regime", {})
        market = state.get("sentiment", {}).get("_market", {})
        vix = market.get("vix", 0)
        nifty_history = market.get("nifty_history")
        state["regime"] = RegimeDetector().run(vix, nifty_history, regime_cfg)
    except Exception as e:
        _logger.error(f"Regime detection failed: {e} — defaulting to TRENDING")
        state["regime"] = "TRENDING"
    return state


def check_earnings(state: TradingState) -> TradingState:
    _logger.info("Step 3d: Checking earnings blackout calendar")
    try:
        symbols = list(state["enriched_data"].keys())
        state["earnings_blackout_symbols"] = EarningsCalendarAgent().run(symbols)
    except Exception as e:
        _logger.error(f"Earnings calendar node failed: {e} — no blackouts applied")
        state["earnings_blackout_symbols"] = []
    return state


def check_fno_expiry(state: TradingState) -> TradingState:
    _logger.info("Step 3e: Checking F&O expiry calendar")
    try:
        state["expiry_context"] = FnOCalendarAgent().run()
    except Exception as e:
        _logger.error(f"F&O expiry check failed: {e} — no expiry gating applied")
        state["expiry_context"] = {"expiry_risk": "NONE"}
    return state


def compute_performance(state: TradingState) -> TradingState:
    _logger.info("Step 4a: Computing strategy performance context")
    from agents.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    portfolio = PortfolioAgent()
    df = portfolio.get_trade_history(limit=500)
    strategy_stats = tracker.compute_strategy_stats(df)
    symbol_stats = tracker.compute_symbol_stats(df)
    state["strategy_stats"] = strategy_stats
    state["perf_context"] = tracker.build_ai_context(strategy_stats, symbol_stats)
    try:
        state["momentum_ranks"] = tracker.compute_momentum_rank(state.get("raw_data", {}))
    except Exception as e:
        _logger.error(f"Momentum rank computation failed: {e}")
        state["momentum_ranks"] = {}
    return state


def _is_rbi_mpc_blackout() -> bool:
    """Return True if today falls within the configured blackout window around an RBI MPC date."""
    import yaml
    from pathlib import Path
    from datetime import date, timedelta
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "trading_config.yaml").read_text())
    mpc_cfg = cfg.get("rbi_mpc", {})
    decision_dates = mpc_cfg.get("decision_dates", [])
    days_before = mpc_cfg.get("blackout_days_before", 2)
    days_after = mpc_cfg.get("blackout_days_after", 2)
    today = date.today()
    for ds in decision_dates:
        try:
            d = date.fromisoformat(ds)
            if (d - timedelta(days=days_before)) <= today <= (d + timedelta(days=days_after)):
                return True
        except Exception:
            continue
    return False


def generate_signals(state: TradingState) -> TradingState:
    _logger.info("Step 4b: Generating signals")
    agent = SignalAgent()

    active_strategies = state.get("active_strategies") or None  # None → use config default
    if _is_rbi_mpc_blackout():
        _logger.warning("RBI MPC blackout window — swing and positional signals suppressed")
        cfg_strategies = _get_active_strategies()
        active_strategies = [s for s in cfg_strategies if s == "intraday"]

    state["signals"] = agent.run(
        state["enriched_data"],
        active_strategies=active_strategies,
        composite_ratings=state.get("composite_ratings", {}),
        regime=state.get("regime", "TRENDING"),
        momentum_ranks=state.get("momentum_ranks", {}),
    )

    # Ensure Claude validation has news context for the symbols we actually signalled —
    # the early sentiment pass only covered the first 20 watchlist names (#17).
    sentiment = state.get("sentiment", {})
    missing = sorted({s.symbol for s in state["signals"] if s.symbol not in sentiment})
    if missing:
        try:
            extra = SentimentAgent().run(missing)
            for sym, data in extra.items():
                if sym != "_market":
                    sentiment[sym] = data
            state["sentiment"] = sentiment
            _logger.info(f"Fetched news sentiment for {len(missing)} signalled symbol(s)")
        except Exception as e:
            _logger.warning(f"Signalled-symbol sentiment fetch failed: {e}")
    return state


def _get_active_strategies() -> list:
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "trading_config.yaml").read_text())
    return cfg.get("strategies", {}).get("active", ["swing", "intraday", "positional"])


def validate_with_claude(state: TradingState) -> TradingState:
    _logger.info("Step 5: Claude signal validation")
    agent = ClaudeValidationAgent()
    state["validated_signals"] = agent.run(
        state["signals"],
        state["sentiment"],
        state.get("composite_ratings", {}),
        state.get("perf_context", ""),
    )
    return state


def check_risk(state: TradingState) -> TradingState:
    _logger.info("Step 6: Risk checks and position sizing")
    portfolio = PortfolioAgent()
    agent = RiskAgent(portfolio)
    context = {
        "earnings_blackout_symbols": state.get("earnings_blackout_symbols", []),
        "expiry_context": state.get("expiry_context", {"expiry_risk": "NONE"}),
        "regime": state.get("regime", "TRENDING"),
        "strategy_stats": state.get("strategy_stats", {}),
        "raw_data": state.get("raw_data", {}),
    }
    approved_with_qty = agent.run(
        [s for s in state["validated_signals"] if s.status == "APPROVED"],
        context=context,
    )

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
    data_agent = DataAgent()

    import yaml
    from pathlib import Path
    _tcfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "trading_config.yaml").read_text())
    _rcfg = yaml.safe_load((Path(__file__).parent.parent / "config" / "risk_config.yaml").read_text())
    max_gap_pct = _tcfg.get("strategies", {}).get("intraday", {}).get("max_entry_gap_pct", 0.02)
    max_position_pct = _rcfg.get("position_sizing", {}).get("max_position_pct", 10.0) / 100

    account_size = float(os.getenv("ACCOUNT_SIZE", "100000"))
    try:
        account_size = portfolio.get_available_cash()
    except Exception:
        pass

    for trade in state["approved_trades"]:
        signal = trade["signal"]
        qty = trade["quantity"]

        # Fetch live price to use as actual fill price (signal.entry_price is yesterday's close)
        quote = data_agent.get_live_quote(signal.symbol)
        if quote:
            live_price = quote["last_price"]
            gap = (live_price - signal.entry_price) / signal.entry_price
            # For intraday breakout signals, reject if price has already gapped up too far
            if signal.strategy == "intraday" and gap > max_gap_pct:
                _logger.warning(
                    f"Skipping {signal.symbol} intraday — live ₹{live_price:.2f} is "
                    f"{gap*100:.1f}% above planned entry ₹{signal.entry_price:.2f} (max {max_gap_pct*100:.0f}%)"
                )
                continue
            # Shift SL and targets by the same delta to preserve ATR-based risk/reward
            price_shift = live_price - signal.entry_price
            signal.entry_price = round(live_price, 2)
            signal.stop_loss = round(signal.stop_loss + price_shift, 2)
            signal.target_1 = round(signal.target_1 + price_shift, 2)
            signal.target_2 = round(signal.target_2 + price_shift, 2)
            signal.recompute_risk()  # keep sl_pct/risk_reward consistent with new levels

            # Re-apply max_position_pct cap against live price (quantity was sized on stale price)
            max_qty_by_value = int((account_size * max_position_pct) / signal.entry_price)
            if qty > max_qty_by_value:
                _logger.info(
                    f"{signal.symbol}: qty capped {qty}→{max_qty_by_value} after live price update"
                )
                qty = max_qty_by_value
            if qty < 1:
                _logger.warning(f"Skipping {signal.symbol}: qty=0 after live price cap")
                continue

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


def run_auto_tune(state: TradingState) -> TradingState:
    _logger.info("Step 9: Auto-tuning confidence thresholds")
    from agents.auto_tuner import AutoTuner
    from agents.performance_tracker import PerformanceTracker
    portfolio = PortfolioAgent()
    tracker = PerformanceTracker()
    df = portfolio.get_trade_history(limit=500)
    strategy_stats = tracker.compute_strategy_stats(df)
    tuner = AutoTuner()
    actions = tuner.run(strategy_stats, state.get("perf_context", ""))
    state["tuning_actions"] = actions
    return state


# ── Graph definition ───────────────────────────────────────────────────────

def build_graph() -> Any:
    workflow = StateGraph(TradingState)

    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("run_ta", run_technical_analysis)
    workflow.add_node("fetch_sentiment", fetch_sentiment)
    workflow.add_node("detect_regime", detect_regime)
    workflow.add_node("fetch_composite_ratings", fetch_composite_ratings)
    workflow.add_node("check_earnings", check_earnings)
    workflow.add_node("check_fno_expiry", check_fno_expiry)
    workflow.add_node("compute_performance", compute_performance)
    workflow.add_node("generate_signals", generate_signals)
    workflow.add_node("validate_claude", validate_with_claude)
    workflow.add_node("check_risk", check_risk)
    workflow.add_node("execute_trades", execute_paper_trades)
    workflow.add_node("send_notifications", send_notifications)
    workflow.add_node("auto_tune", run_auto_tune)

    workflow.add_edge(START, "fetch_data")
    workflow.add_edge("fetch_data", "run_ta")
    workflow.add_edge("run_ta", "fetch_sentiment")
    workflow.add_edge("fetch_sentiment", "detect_regime")
    workflow.add_edge("detect_regime", "fetch_composite_ratings")
    workflow.add_edge("fetch_composite_ratings", "check_earnings")
    workflow.add_edge("check_earnings", "check_fno_expiry")
    workflow.add_edge("check_fno_expiry", "compute_performance")
    workflow.add_edge("compute_performance", "generate_signals")
    workflow.add_edge("generate_signals", "validate_claude")
    workflow.add_edge("validate_claude", "check_risk")
    workflow.add_edge("check_risk", "execute_trades")
    workflow.add_edge("execute_trades", "send_notifications")
    workflow.add_edge("send_notifications", "auto_tune")
    workflow.add_edge("auto_tune", END)

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

        # Fetch prices for all open position symbols (not just watchlist[:20])
        open_syms = list({p["symbol"] for p in portfolio.get_open_positions()})
        live_prices = {}
        day_highs = {}
        day_lows = {}
        for sym in open_syms:
            q = data_agent.get_live_quote(sym)
            if q:
                live_prices[sym] = q["last_price"]
                day_highs[sym] = q["day_high"]
                day_lows[sym] = q["day_low"]

        # Fetch TA data for open position symbols only (for T1 reversal detection)
        ta_data = {}
        try:
            if open_syms:
                raw = data_agent.run(open_syms, timeframe="1d", days=30)
                ta_data = TechnicalAnalysisAgent().run(raw)
        except Exception as e:
            _logger.warning(f"T1 reversal TA fetch failed: {e} — reversal check skipped")

        closed = broker.check_exits(live_prices, ta_data=ta_data, day_highs=day_highs, day_lows=day_lows)
        if closed:
            notif = NotificationAgent()
            for trade in closed:
                emoji = "✅" if (trade.get("pnl") or 0) >= 0 else "❌"
                notif.send_alert(
                    f"{emoji} {trade['symbol']} closed: {trade['reason']} | P&L ₹{(trade.get('pnl') or 0):.2f}"
                )

    @staticmethod
    def _auto_close_eod_enabled() -> bool:
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return bool(cfg.get("strategies", {}).get("intraday", {}).get("auto_close_eod", True))

    def _close_intraday_positions(self, stale_only: bool = False) -> list:
        """Fetch quotes for open intraday positions and close them via the broker."""
        portfolio = PortfolioAgent()
        open_positions = portfolio.get_open_positions()
        intraday_syms = {p["symbol"] for p in open_positions if p.get("strategy") == "intraday"}
        if not intraday_syms:
            return []

        data_agent = DataAgent()
        live_prices = {}
        for sym in intraday_syms:
            q = data_agent.get_live_quote(sym)
            if q:
                live_prices[sym] = q["last_price"]

        broker = PaperBroker(portfolio)
        closed = broker.close_intraday_eod(live_prices, stale_only=stale_only)
        if closed:
            notif = NotificationAgent()
            for trade in closed:
                emoji = "✅" if (trade.get("pnl") or 0) >= 0 else "❌"
                label = "EOD Close (missed — closed on restart)" if stale_only else "EOD Close"
                notif.send_alert(
                    f"{emoji} {label}: {trade['symbol']} @ ₹{trade['exit_price']:.2f} | P&L ₹{(trade.get('pnl') or 0):.2f}"
                )
            _logger.info(f"EOD closed {len(closed)} intraday position(s)")
        return closed

    def _run_intraday_eod_close(self):
        if not is_trading_day():
            return
        if not self._auto_close_eod_enabled():
            return
        _logger.info("=== INTRADAY EOD CLOSE ===")
        self._close_intraday_positions()

    def _recover_missed_eod_close(self):
        """Close intraday positions whose 3:20 PM EOD close never fired.

        The cron job is skipped when the app shuts down before 15:20 IST and
        restarts after the misfire grace window, leaving intraday positions open
        overnight. Runs once at startup; deliberately NOT gated on is_trading_day()
        so a stale Friday position is still closed on a weekend restart."""
        if not self._auto_close_eod_enabled():
            return
        closed = self._close_intraday_positions(stale_only=True)
        if closed:
            _logger.warning(
                f"Startup recovery: closed {len(closed)} intraday position(s) "
                "left open by a missed EOD-close job"
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
            "approved_trades": [], "sentiment": {}, "composite_ratings": {},
            "perf_context": "", "strategy_stats": {}, "tuning_actions": [],
            "portfolio_stats": {}, "closed_today": [], "errors": [],
            "regime": "TRENDING",
            "earnings_blackout_symbols": [],
            "momentum_ranks": {},
            "expiry_context": {"expiry_risk": "NONE"},
            "active_strategies": [],
        }
        try:
            self._graph.invoke(initial_state)
        except Exception as e:
            _logger.error(f"Workflow error: {e}")

    def run_now(self):
        """Run the full workflow immediately (for testing or manual trigger)."""
        self._run_workflow()

    def start(self):
        # Recover from an EOD close missed while the app was down — never block startup
        try:
            self._recover_missed_eod_close()
        except Exception as e:
            _logger.error(f"Startup EOD-close recovery failed: {e}")

        # Morning scan at 9:15 AM IST — allow up to 10 min late, never overlap
        self._scheduler.add_job(self._run_morning_scan, "cron",
                                 hour=9, minute=15, day_of_week="mon-fri",
                                 max_instances=1, coalesce=True,
                                 misfire_grace_time=600)
        # Intraday monitor every 15 min — if missed, run once (don't stack)
        self._scheduler.add_job(self._run_intraday_monitor, "cron",
                                 hour="9-15", minute="*/15", day_of_week="mon-fri",
                                 max_instances=1, coalesce=True,
                                 misfire_grace_time=None)
        # Intraday EOD auto-close at 3:20 PM (after last SL/TP check at 3:15).
        # Grace of 1h: a restart shortly after 3:20 still fires the job; anything
        # later is handled by _recover_missed_eod_close() on the next startup.
        self._scheduler.add_job(self._run_intraday_eod_close, "cron",
                                 hour=15, minute=20, day_of_week="mon-fri",
                                 max_instances=1, coalesce=True,
                                 misfire_grace_time=3600)
        # EOD report at 3:45 PM
        self._scheduler.add_job(self._run_eod_report, "cron",
                                 hour=15, minute=45, day_of_week="mon-fri",
                                 max_instances=1, coalesce=True,
                                 misfire_grace_time=600)
        self._scheduler.start()
        _logger.info("Trading scheduler started (IST timezone)")

    def stop(self):
        self._scheduler.shutdown()
        _logger.info("Trading scheduler stopped")
