"""
Trading Agent — Entry Point
Usage:
  python main.py                    # Start full scheduled trading system
  python main.py --scan             # Run one signal scan immediately
  python main.py --dashboard        # Launch Streamlit dashboard
  python main.py --backtest SYMBOL  # Backtest a symbol (all strategies)
  python main.py --portfolio        # Show portfolio stats
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.logger import get_logger
from monitoring.health import current_ist_time, is_market_open, is_trading_day

_logger = get_logger("Main")


def run_dashboard():
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    _logger.info("Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)], check=True)


def run_single_scan():
    from agents.data_agent import DataAgent
    from orchestrator.workflow import TradingScheduler

    _logger.info(f"Running signal scan at {current_ist_time()}")
    data_agent = DataAgent()
    symbols = data_agent.get_watchlist("nifty50")

    scheduler = TradingScheduler(symbols)
    scheduler.run_now()
    _logger.info("Signal scan complete.")


def run_backtest(symbol: str, strategy: str = "swing", days: int = 365):
    from backtesting.engine import BacktestEngine

    _logger.info(f"Backtesting {symbol} | {strategy} | {days} days")
    engine = BacktestEngine()
    results = engine.run([symbol.upper()], strategy=strategy, days=days)

    if not results:
        _logger.info("No backtest results — check symbol and data availability")
        return

    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=f"Backtest Results: {symbol} ({strategy})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    r = results[0].summary()
    for k, v in r.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


def show_portfolio():
    from agents.portfolio_agent import PortfolioAgent

    portfolio = PortfolioAgent()
    stats = portfolio.get_performance_stats()
    daily = portfolio.get_daily_stats()
    positions = portfolio.get_open_positions()

    from rich.console import Console
    from rich.table import Table
    console = Console()

    table = Table(title="Portfolio Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    if positions:
        pos_table = Table(title=f"Open Positions ({len(positions)})")
        for col in ["symbol", "strategy", "entry_price", "stop_loss", "target_1", "target_2", "quantity"]:
            pos_table.add_column(col.replace("_", " ").title())
        for p in positions:
            pos_table.add_row(*[str(p.get(c, "")) for c in ["symbol", "strategy", "entry_price", "stop_loss", "target_1", "target_2", "quantity"]])
        console.print(pos_table)


def run_scheduler():
    from agents.data_agent import DataAgent
    from orchestrator.workflow import TradingScheduler

    _logger.info("=" * 60)
    _logger.info("Trading Agent Starting")
    _logger.info(f"Time: {current_ist_time()}")
    _logger.info(f"Paper Trading: {os.getenv('PAPER_TRADING', 'true')}")
    _logger.info(f"Trading Enabled: {os.getenv('TRADING_ENABLED', 'true')}")
    _logger.info("=" * 60)

    data_agent = DataAgent()
    symbols = data_agent.get_watchlist("nifty50")
    _logger.info(f"Watching {len(symbols)} Nifty 50 stocks")

    scheduler = TradingScheduler(symbols)
    scheduler.start()

    if is_market_open():
        _logger.info("Market is currently OPEN — running immediate scan")
        scheduler.run_now()

    _logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        _logger.info("Shutting down...")
        scheduler.stop()


def main():
    parser = argparse.ArgumentParser(description="NSE/BSE Trading Agent")
    parser.add_argument("--scan", action="store_true", help="Run one signal scan immediately")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--backtest", metavar="SYMBOL", help="Backtest a symbol")
    parser.add_argument("--strategy", default="swing", help="Strategy for backtest (default: swing)")
    parser.add_argument("--days", type=int, default=365, help="Backtest lookback days (default: 365)")
    parser.add_argument("--portfolio", action="store_true", help="Show portfolio stats")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    elif args.scan:
        run_single_scan()
    elif args.backtest:
        run_backtest(args.backtest, args.strategy, args.days)
    elif args.portfolio:
        show_portfolio()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
