"""
Tracks paper trading positions, P&L, and portfolio statistics.
Reads/writes from SQLite via SQLAlchemy.
"""

from __future__ import annotations

import os
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent

_IST = pytz.timezone("Asia/Kolkata")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    strategy TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_1 REAL NOT NULL,
    target_2 REAL NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT DEFAULT 'OPEN',
    exit_price REAL,
    pnl REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_reason TEXT,
    confidence REAL,
    claude_verdict TEXT
);

CREATE TABLE IF NOT EXISTS signals_log (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    strategy TEXT,
    signal_type TEXT,
    confidence REAL,
    claude_verdict TEXT,
    claude_reasoning TEXT,
    timestamp TEXT,
    status TEXT,
    tv_rating TEXT
);
"""


class PortfolioAgent(BaseAgent):
    def __init__(self):
        super().__init__("PortfolioAgent")
        os.makedirs("data", exist_ok=True)
        db_url = os.getenv("DATABASE_URL", "sqlite:///data/trading.db")
        self._engine = create_engine(db_url, echo=False)
        self._init_db()

    def run(self) -> dict:
        return self.get_performance_stats()

    def _init_db(self):
        with self._engine.connect() as conn:
            for stmt in _SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            # Migrate existing DB: add tv_rating column if absent
            try:
                conn.execute(text("ALTER TABLE signals_log ADD COLUMN tv_rating TEXT"))
            except Exception:
                pass  # Column already exists
            conn.commit()

    # ── Position management ────────────────────────────────────────────────

    def open_position(self, trade_id: str, symbol: str, signal_type: str, strategy: str,
                      entry_price: float, stop_loss: float, target_1: float, target_2: float,
                      quantity: int, confidence: float = 0.0, claude_verdict: str = "") -> bool:
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trades (id, symbol, signal_type, strategy, entry_price, stop_loss,
                    target_1, target_2, quantity, status, entry_time, confidence, claude_verdict)
                VALUES (:id, :sym, :stype, :strat, :ep, :sl, :t1, :t2, :qty, 'OPEN', :et, :conf, :cv)
            """), {
                "id": trade_id, "sym": symbol, "stype": signal_type, "strat": strategy,
                "ep": entry_price, "sl": stop_loss, "t1": target_1, "t2": target_2,
                "qty": quantity, "et": datetime.now(_IST).isoformat(),
                "conf": confidence, "cv": claude_verdict,
            })
            conn.commit()
        self.log_info(f"Opened {signal_type} position: {symbol} x{quantity} @ ₹{entry_price}", symbol=symbol)
        return True

    def close_position(self, trade_id: str, exit_price: float, reason: str = "manual") -> Optional[float]:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM trades WHERE id = :id AND status = 'OPEN'"),
                               {"id": trade_id}).fetchone()
            if not row:
                return None
            pnl = (exit_price - row.entry_price) * row.quantity
            if row.signal_type == "SELL":
                pnl = -pnl
            conn.execute(text("""
                UPDATE trades SET status='CLOSED', exit_price=:ep, pnl=:pnl,
                    exit_time=:et, exit_reason=:reason WHERE id=:id
            """), {
                "ep": exit_price, "pnl": pnl,
                "et": datetime.now(_IST).isoformat(),
                "reason": reason, "id": trade_id,
            })
            conn.commit()
        self.log_info(f"Closed {trade_id}: P&L ₹{pnl:.2f} ({reason})", trade_id=trade_id)
        return pnl

    def update_stop_loss(self, trade_id: str, new_sl: float):
        with self._engine.connect() as conn:
            conn.execute(text("UPDATE trades SET stop_loss=:sl WHERE id=:id"),
                         {"sl": new_sl, "id": trade_id})
            conn.commit()

    # ── Position queries ───────────────────────────────────────────────────

    def get_open_positions(self) -> List[dict]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM trades WHERE status='OPEN'")).fetchall()
        return [dict(r._mapping) for r in rows]

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM trades WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT :lim"
            ), {"lim": limit}).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r._mapping) for r in rows])

    # ── P&L and statistics ─────────────────────────────────────────────────

    def get_daily_stats(self) -> dict:
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT pnl FROM trades WHERE status='CLOSED' AND exit_time LIKE :today"
            ), {"today": f"{today}%"}).fetchall()
        daily_pnl = sum(r.pnl for r in rows if r.pnl is not None)
        initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        return {
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round((daily_pnl / initial) * 100, 2),
            "trades_today": len(rows),
        }

    def get_performance_stats(self) -> dict:
        df = self.get_trade_history(limit=500)
        initial_capital = float(os.getenv("ACCOUNT_SIZE", "100000"))

        if df.empty:
            return {
                "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "current_equity": initial_capital, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            }

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        total_pnl = df["pnl"].sum()
        current_equity = initial_capital + total_pnl

        # Running equity for drawdown calculation
        df_sorted = df.sort_values("exit_time")
        equity_curve = initial_capital + df_sorted["pnl"].cumsum()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Sharpe (annualized per day, not per trade)
        df_sorted["exit_date"] = pd.to_datetime(df_sorted["exit_time"]).dt.date
        daily_pnl_series = df_sorted.groupby("exit_date")["pnl"].sum() / initial_capital
        sharpe = 0.0
        if len(daily_pnl_series) > 5 and daily_pnl_series.std() > 0:
            sharpe = round((daily_pnl_series.mean() / daily_pnl_series.std()) * (252 ** 0.5), 2)

        return {
            "total_trades": len(df),
            "win_rate": round(len(wins) / len(df) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "current_equity": round(current_equity, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe": sharpe,
            "avg_win": round(wins["pnl"].mean(), 2) if not wins.empty else 0.0,
            "avg_loss": round(losses["pnl"].mean(), 2) if not losses.empty else 0.0,
        }

    def log_signal(self, signal_dict: dict):
        """Persist every signal to audit log regardless of trade decision."""
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO signals_log
                    (id, symbol, strategy, signal_type, confidence, claude_verdict,
                     claude_reasoning, timestamp, status, tv_rating)
                VALUES (:id, :sym, :strat, :stype, :conf, :cv, :cr, :ts, :st, :tvr)
            """), {
                "id": signal_dict.get("id"),
                "sym": signal_dict.get("symbol"),
                "strat": signal_dict.get("strategy"),
                "stype": signal_dict.get("signal_type"),
                "conf": signal_dict.get("confidence"),
                "cv": signal_dict.get("claude_verdict"),
                "cr": signal_dict.get("claude_reasoning"),
                "ts": signal_dict.get("timestamp"),
                "st": signal_dict.get("status"),
                "tvr": signal_dict.get("tv_rating", "UNKNOWN"),
            })
            conn.commit()
