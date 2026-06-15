"""
Tracks paper trading positions, P&L, and portfolio statistics.
Reads/writes from SQLite via SQLAlchemy.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent
from brokers.costs import chunk_round_trip_cost

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
    original_quantity INTEGER,
    t1_booked INTEGER DEFAULT 0,
    t2_booked INTEGER DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
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
    rejection_reason TEXT,
    composite_rating TEXT,
    entry_price REAL,
    stop_loss REAL,
    target_1 REAL,
    target_2 REAL,
    risk_reward REAL,
    sl_pct REAL,
    timeframe TEXT
);

CREATE TABLE IF NOT EXISTS partial_exits (
    id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    symbol TEXT,
    quantity INTEGER NOT NULL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT,
    exit_time TEXT NOT NULL
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
            # Rename legacy tv_rating column to composite_rating
            try:
                conn.execute(text("ALTER TABLE signals_log RENAME COLUMN tv_rating TO composite_rating"))
            except Exception:
                pass  # Already renamed or doesn't exist
            # Migrate existing DB — add columns that may be absent in older DBs
            _new_cols = [
                ("rejection_reason", "TEXT"),
                ("composite_rating", "TEXT"),
                ("entry_price", "REAL"),
                ("stop_loss", "REAL"),
                ("target_1", "REAL"),
                ("target_2", "REAL"),
                ("risk_reward", "REAL"),
                ("sl_pct", "REAL"),
                ("timeframe", "TEXT"),
            ]
            for col, col_type in _new_cols:
                try:
                    conn.execute(text(f"ALTER TABLE signals_log ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass  # Column already exists
            # Migrate trades table — partial-exit tracking columns for older DBs
            _trade_cols = [
                ("original_quantity", "INTEGER"),
                ("t1_booked", "INTEGER DEFAULT 0"),
                ("t2_booked", "INTEGER DEFAULT 0"),
                ("realized_pnl", "REAL DEFAULT 0"),
            ]
            for col, col_type in _trade_cols:
                try:
                    conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass  # Column already exists
            # Backfill original_quantity for pre-existing open rows
            try:
                conn.execute(text(
                    "UPDATE trades SET original_quantity = quantity "
                    "WHERE original_quantity IS NULL"
                ))
            except Exception:
                pass
            conn.commit()

    # ── Position management ────────────────────────────────────────────────

    def open_position(self, trade_id: str, symbol: str, signal_type: str, strategy: str,
                      entry_price: float, stop_loss: float, target_1: float, target_2: float,
                      quantity: int, confidence: float = 0.0, claude_verdict: str = "") -> bool:
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trades (id, symbol, signal_type, strategy, entry_price, stop_loss,
                    target_1, target_2, quantity, original_quantity, t1_booked, t2_booked,
                    realized_pnl, status, entry_time, confidence, claude_verdict)
                VALUES (:id, :sym, :stype, :strat, :ep, :sl, :t1, :t2, :qty, :qty, 0, 0,
                    0, 'OPEN', :et, :conf, :cv)
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
            # Deduct realistic round-trip transaction costs on the remaining quantity.
            # The entry leg is pro-rated against the ORIGINAL order size so chunked
            # scale-outs never multiply the per-order brokerage cap.
            pnl -= chunk_round_trip_cost(
                row.entry_price, exit_price, row.quantity,
                row.original_quantity or row.quantity,
                row.signal_type, row.strategy == "intraday",
            )
            # Fold in profit already booked from T1/T2 partial scale-outs so the
            # trade's recorded P&L reflects the entire position, not just the runner.
            pnl += (row.realized_pnl or 0)
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

    def partial_close(self, trade_id: str, exit_qty: int, exit_price: float,
                      reason: str = "PARTIAL", milestone: str = None) -> Optional[float]:
        """
        Scale out part of an open position (T1/T2 profit booking).
        Reduces the open quantity, accrues the booked profit into realized_pnl,
        and optionally flags a milestone ('t1' or 't2') so it isn't re-booked.
        Returns the realized P&L of this partial, or None if no open position.
        """
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM trades WHERE id = :id AND status = 'OPEN'"),
                               {"id": trade_id}).fetchone()
            if not row:
                return None
            exit_qty = min(int(exit_qty), int(row.quantity))
            if exit_qty <= 0:
                # Nothing to book (position too small to split) — but still flag the
                # milestone if asked so the exit state machine doesn't re-trigger.
                if milestone in ("t1", "t2"):
                    col = "t1_booked" if milestone == "t1" else "t2_booked"
                    conn.execute(text(f"UPDATE trades SET {col}=1 WHERE id=:id"), {"id": trade_id})
                    conn.commit()
                return 0.0
            realized = (exit_price - row.entry_price) * exit_qty
            if row.signal_type == "SELL":
                realized = -realized
            # Cost this chunk's round trip up front: its share of the original entry
            # order plus its own exit order (per-order caps applied correctly).
            realized -= chunk_round_trip_cost(
                row.entry_price, exit_price, exit_qty,
                row.original_quantity or row.quantity,
                row.signal_type, row.strategy == "intraday",
            )
            new_qty = row.quantity - exit_qty
            new_realized = (row.realized_pnl or 0) + realized

            set_clauses = "quantity=:q, realized_pnl=:rp"
            params = {"q": new_qty, "rp": new_realized, "id": trade_id}
            if milestone == "t1":
                set_clauses += ", t1_booked=1"
            elif milestone == "t2":
                set_clauses += ", t2_booked=1"
            conn.execute(text(f"UPDATE trades SET {set_clauses} WHERE id=:id"), params)
            # Ledger the chunk with its own timestamp so daily P&L attributes each
            # booking to the day it actually happened (not the runner's close day).
            conn.execute(text("""
                INSERT INTO partial_exits (id, trade_id, symbol, quantity, exit_price,
                    pnl, reason, exit_time)
                VALUES (:id, :tid, :sym, :qty, :px, :pnl, :reason, :et)
            """), {
                "id": str(uuid.uuid4())[:8], "tid": trade_id, "sym": row.symbol,
                "qty": exit_qty, "px": exit_price, "pnl": realized, "reason": reason,
                "et": datetime.now(_IST).isoformat(),
            })
            conn.commit()
        self.log_info(
            f"Partial exit {trade_id}: {exit_qty} @ ₹{exit_price:.2f} | "
            f"booked ₹{realized:.2f} ({reason}); {new_qty} left",
            trade_id=trade_id,
        )
        return realized

    def update_stop_loss(self, trade_id: str, new_sl: float):
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT signal_type, stop_loss FROM trades WHERE id=:id AND status='OPEN'"),
                {"id": trade_id},
            ).fetchone()
            if not row:
                return
            # Never widen the stop — BUY SL can only move up, SELL SL can only move down
            if row.signal_type == "BUY" and new_sl < row.stop_loss:
                self.log_warning(f"Rejected SL update for {trade_id}: would widen BUY stop {row.stop_loss} → {new_sl}")
                return
            if row.signal_type == "SELL" and new_sl > row.stop_loss:
                self.log_warning(f"Rejected SL update for {trade_id}: would widen SELL stop {row.stop_loss} → {new_sl}")
                return
            conn.execute(text("UPDATE trades SET stop_loss=:sl WHERE id=:id"),
                         {"sl": new_sl, "id": trade_id})
            conn.commit()

    # ── Position queries ───────────────────────────────────────────────────

    def get_open_positions(self) -> List[dict]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM trades WHERE status='OPEN'")).fetchall()
        return [dict(r._mapping) for r in rows]

    def _get_open_realized_pnl(self) -> float:
        """Sum of realized_pnl already booked (T1/T2 partials) on OPEN trades."""
        try:
            with self._engine.connect() as conn:
                val = conn.execute(text(
                    "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE status='OPEN'"
                )).scalar()
            return float(val or 0.0)
        except Exception:
            return 0.0

    def get_unrealized_pnl(self, live_prices: Dict[str, float]) -> float:
        """Mark open positions to market. Positions without a quoted price are skipped."""
        total = 0.0
        for p in self.get_open_positions():
            price = live_prices.get(p["symbol"])
            if price is None:
                continue
            pnl = (price - p["entry_price"]) * p["quantity"]
            if p["signal_type"] == "SELL":
                pnl = -pnl
            total += pnl
        return round(total, 2)

    def get_available_cash(self) -> float:
        """Equity minus capital tied up in open positions (unrealized exposure deducted)."""
        initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        stats = self.get_performance_stats()
        equity = stats.get("current_equity", initial)
        open_positions = self.get_open_positions()
        invested = sum(p["entry_price"] * p["quantity"] for p in open_positions)
        return max(equity - invested, 0.0)

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
        """Today's realized P&L, attributing every fill to the day it happened:
        T1/T2 partial bookings count on their booking day (partial_exits ledger);
        a trade's final close counts only its runner P&L (pnl minus the partials
        already attributed), so nothing is double-counted or shifted to close day."""
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        with self._engine.connect() as conn:
            # Subtract only LEDGERED partials from each closing trade's pnl — partials
            # booked before the ledger existed have no rows and stay attributed to the
            # close day (legacy behaviour) instead of disappearing from daily stats.
            rows = conn.execute(text("""
                SELECT t.pnl,
                       COALESCE((SELECT SUM(p.pnl) FROM partial_exits p
                                 WHERE p.trade_id = t.id), 0) AS ledgered
                FROM trades t
                WHERE t.status='CLOSED' AND t.exit_time LIKE :today
            """), {"today": f"{today}%"}).fetchall()
            partial_pnl = conn.execute(text(
                "SELECT COALESCE(SUM(pnl), 0) FROM partial_exits WHERE exit_time LIKE :today"
            ), {"today": f"{today}%"}).scalar() or 0.0
        runner_pnl = sum(
            (r.pnl or 0) - (r.ledgered or 0) for r in rows if r.pnl is not None
        )
        daily_pnl = runner_pnl + partial_pnl
        initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        return {
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round((daily_pnl / initial) * 100, 2),
            "trades_today": len(rows),
        }

    def get_performance_stats(self) -> dict:
        df = self.get_trade_history(limit=500)
        initial_capital = float(os.getenv("ACCOUNT_SIZE", "100000"))
        # Profit already banked by T1/T2 scale-outs on still-open trades is real
        # cash — equity, available-cash, and live-drawdown checks must see it.
        open_realized = self._get_open_realized_pnl()

        if df.empty:
            return {
                "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "current_equity": round(initial_capital + open_realized, 2),
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "peak_equity": initial_capital,
            }

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        total_pnl = df["pnl"].sum()
        current_equity = initial_capital + total_pnl + open_realized

        # Running equity for drawdown calculation
        df_sorted = df.sort_values("exit_time")
        equity_curve = initial_capital + df_sorted["pnl"].cumsum()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Sharpe (annualized per day, not per trade)
        df_sorted["exit_date"] = df_sorted["exit_time"].apply(
            lambda x: datetime.fromisoformat(x).date() if isinstance(x, str) and x else None
        )
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
            "peak_equity": round(float(peak.max()), 2),
        }

    def log_signal(self, signal_dict: dict):
        """Persist every signal to audit log regardless of trade decision."""
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO signals_log
                    (id, symbol, strategy, signal_type, confidence, claude_verdict,
                     claude_reasoning, timestamp, status, rejection_reason, composite_rating,
                     entry_price, stop_loss, target_1, target_2,
                     risk_reward, sl_pct, timeframe)
                VALUES (:id, :sym, :strat, :stype, :conf, :cv, :cr, :ts, :st, :rj, :tvr,
                        :ep, :sl, :t1, :t2, :rr, :slp, :tf)
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
                "rj": signal_dict.get("rejection_reason"),
                "tvr": signal_dict.get("composite_rating", "UNKNOWN"),
                "ep": signal_dict.get("entry_price"),
                "sl": signal_dict.get("stop_loss"),
                "t1": signal_dict.get("target_1"),
                "t2": signal_dict.get("target_2"),
                "rr": signal_dict.get("risk_reward"),
                "slp": signal_dict.get("sl_pct"),
                "tf": signal_dict.get("timeframe"),
            })
            conn.commit()
