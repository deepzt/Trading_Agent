"""
PositionReviewAgent — daily EOD conviction review of OPEN holdings.

The mechanical exit path (brokers/paper_broker.check_exits) only ever fires on
hard price levels: stop-loss, T1/T2, an indicator flip, or the max-hold time-stop.
A position that drifts sideways while slightly green — "stale / dead money" — slips
through every one of those until its (often distant) time-stop. This agent fills
that gap: once a day it re-scores each open holding for staleness / conviction
decay and recommends HOLD / TRIM / EXIT.

Scoring is LOCAL first (free, no API): time pressure (held a long fraction of the
max-hold window with little progress to T1), momentum decay (composite TA score
weak / rolled over for the position's direction), and dead money (flat P&L while
the clock runs down). Only BORDERLINE cases are escalated to Claude/GPT for a
tie-break, reusing the same provider plumbing as ClaudeValidationAgent.

Action: clear EXIT cases are auto-closed via PortfolioAgent.close_position (paper
trades only, same path every other exit uses) ONLY when
  - position_review.auto_exit is true, AND
  - exit_conviction >= auto_exit_min_confidence (0-10), AND
  - the account has >= min_closed_trades_for_auto_exit closed trades (don't act on
    immature statistics).
Everything else is surfaced as a recommendation (dashboard + Telegram) and left
for the human to act on. Latest results are persisted to data/position_reviews.json
for the dashboard.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytz
import yaml

from agents.base_agent import BaseAgent
from agents.claude_validation_agent import _resolve_provider

_IST = pytz.timezone("Asia/Kolkata")
_CFG_PATH = Path(__file__).parent.parent / "config" / "trading_config.yaml"
_REVIEWS_PATH = Path(__file__).parent.parent / "data" / "position_reviews.json"

_REVIEW_SYSTEM_PROMPT = (
    "You are a disciplined portfolio risk reviewer for an Indian-equities (NSE) paper-"
    "trading system. You are given ONE currently-open position and its current state. "
    "Decide whether it is still worth holding capital, or whether the capital is stale / "
    "dead money that should be freed. Be conservative: do not recommend EXIT on normal "
    "noise or a position that simply hasn't moved yet — only when momentum has genuinely "
    "decayed or the holding has stalled for a meaningful fraction of its intended window.\n\n"
    "Respond with JSON ONLY in this exact shape:\n"
    '{"verdict": "HOLD" | "TRIM" | "EXIT", "confidence": <0-10 conviction that the '
    'position should be EXITED>, "reasoning": "<one concise sentence>"}'
)


class PositionReviewAgent(BaseAgent):
    """Re-evaluates open holdings for staleness / conviction decay once per day."""

    def __init__(self):
        super().__init__("PositionReviewAgent")
        cfg = self._load_cfg()
        pr = cfg.get("position_review", {}) or {}

        self.enabled: bool = bool(pr.get("enabled", True))
        self.auto_exit: bool = bool(pr.get("auto_exit", True))
        self.gate: float = float(pr.get("auto_exit_min_confidence", 8.0))
        self.min_closed_trades: int = int(pr.get("min_closed_trades_for_auto_exit", 10))
        self.borderline_band: float = float(pr.get("borderline_band", 1.5))
        self.min_hold_bars: int = int(pr.get("min_hold_bars_before_review", 2))
        self.use_claude: bool = bool(pr.get("use_claude_on_borderline", True))

        # Per-strategy max-hold windows (calendar days), same source the time-stop uses.
        self._max_hold = self._load_max_hold(cfg)

        # LLM client is created lazily on the first borderline case.
        provider_pref = cfg.get("validation", {}).get("provider", "auto")
        self._provider = _resolve_provider(provider_pref) if self.use_claude else "none"
        self._claude_cfg = cfg.get("claude", {})
        self._openai_cfg = cfg.get("openai", {})
        self._client = None  # built on demand

    # ── config loading ──────────────────────────────────────────────────────
    @staticmethod
    def _load_cfg() -> dict:
        with open(_CFG_PATH) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_max_hold(cfg: dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for name, scfg in (cfg.get("strategies", {}) or {}).items():
            if isinstance(scfg, dict) and scfg.get("max_hold_days"):
                out[name] = int(scfg["max_hold_days"])
        return out

    # ── public entry point ──────────────────────────────────────────────────
    def run(self, dry_run: bool = False) -> dict:
        """Review every open holding. When dry_run is False, auto-close clear EXIT
        cases (subject to the config gates). Returns a summary dict and persists the
        per-position reviews to data/position_reviews.json for the dashboard."""
        from agents.data_agent import DataAgent
        from agents.portfolio_agent import PortfolioAgent
        from agents.technical_analysis_agent import TechnicalAnalysisAgent
        from agents.tradingview_ta_agent import CompositeTAAgent

        if not self.enabled:
            self.log_info("Position review disabled in config — skipping")
            return {"reviews": [], "skipped": True}

        portfolio = PortfolioAgent()
        positions = portfolio.get_open_positions()
        if not positions:
            self._persist([], closed_count=0)
            self.log_info("Position review: no open positions")
            return {"reviews": [], "acted": 0}

        symbols = sorted({p["symbol"] for p in positions})
        data_agent = DataAgent()

        live_prices: Dict[str, float] = {}
        for sym in symbols:
            q = data_agent.get_live_quote(sym)
            if q:
                live_prices[sym] = q["last_price"]

        # Fresh TA for the held symbols only (few positions → cheap). 260 daily bars
        # so EMA200-dependent composite components are valid.
        ta_data: Dict = {}
        comp_scores: Dict[str, Optional[float]] = {}
        comp_labels: Dict[str, str] = {}
        try:
            raw = data_agent.run(symbols, timeframe="1d", days=260)
            ta_data = TechnicalAnalysisAgent().run(raw)
            cta = CompositeTAAgent()
            comp_labels = cta.run(ta_data)
            comp_scores = cta.score(ta_data)
        except Exception as e:
            self.log_warning(f"TA fetch failed — reviewing on price/time only: {e}")

        closed_count = self._closed_trade_count(portfolio)
        can_auto = self.auto_exit and closed_count >= self.min_closed_trades

        reviews: List[dict] = []
        for pos in positions:
            sym = pos["symbol"]
            r = self._review_position(
                pos,
                price=live_prices.get(sym),
                df=ta_data.get(sym),
                comp_score=comp_scores.get(sym),
                comp_label=comp_labels.get(sym, "UNKNOWN"),
            )
            reviews.append(r)

        acted = self._apply_actions(reviews, live_prices, portfolio, can_auto, dry_run)

        self._persist(reviews, closed_count=closed_count, can_auto=can_auto, acted=acted)
        self.log_info(
            f"Position review: {len(reviews)} holdings | "
            f"{sum(1 for r in reviews if r['verdict'] == 'EXIT')} exit | "
            f"{acted} auto-closed (auto={'on' if can_auto else 'off'}, closed_trades={closed_count})"
        )
        return {"reviews": reviews, "acted": acted, "closed_count": closed_count, "can_auto": can_auto}

    # ── per-position scoring ────────────────────────────────────────────────
    def _review_position(self, pos: dict, price: Optional[float], df,
                         comp_score: Optional[float], comp_label: str) -> dict:
        sym = pos["symbol"]
        strategy = pos.get("strategy") or "swing"
        entry = float(pos["entry_price"])
        t1 = float(pos["target_1"])
        side = pos.get("signal_type", "BUY")
        bars_held = self._bars_held(pos.get("entry_time"))
        max_days = self._max_hold.get(strategy)

        unreal_pct = None
        t1_progress = None
        if price is not None and entry:
            move = (price - entry) / entry * 100.0
            unreal_pct = round(move if side == "BUY" else -move, 2)
            denom = (t1 - entry) if side == "BUY" else (entry - t1)
            num = (price - entry) if side == "BUY" else (entry - price)
            if denom:
                t1_progress = round(max(min(num / denom, 1.5), -1.5), 2)

        conviction = 0.0
        reasons: List[str] = []

        # 1) Time pressure — long fraction of the window elapsed with little progress to T1.
        hold_frac = (bars_held / max_days) if max_days else None
        if hold_frac is not None and hold_frac >= 0.5:
            prog = t1_progress if t1_progress is not None else 0.0
            stalled = max(0.0, 1.0 - max(prog, 0.0))          # 1.0 = no progress
            pressure = min((hold_frac - 0.5) / 0.5, 1.0)        # ramps 0→1 over the back half
            add = 4.0 * pressure * stalled
            if add > 0.3:
                conviction += add
                reasons.append(
                    f"held {bars_held}/{max_days}d ({hold_frac*100:.0f}% of window), "
                    f"only {max(prog,0.0)*100:.0f}% to T1"
                )

        # 2) Momentum decay — composite TA weak for the position's DIRECTION.
        if comp_score is not None:
            pos_momentum = comp_score if side == "BUY" else (10.0 - comp_score)
            if pos_momentum < 5.0:
                add = 3.0 * min((5.0 - pos_momentum) / 5.0, 1.0)
                conviction += add
                reasons.append(f"composite TA {comp_score:.1f}/10 ({comp_label}) — momentum fading")
        if side == "BUY" and comp_label in ("SELL", "STRONG_SELL"):
            conviction += 1.0
            if "composite" not in " ".join(reasons):
                reasons.append(f"composite rating turned {comp_label}")
        # EMA/MACD rollover straight from the indicator frame.
        if df is not None and len(df) >= 1:
            row = df.iloc[-1]
            ema_bearish = (not row.get("ema_bullish_align", True)) if side == "BUY" \
                else bool(row.get("ema_bullish_align", False))
            macdh = row.get("MACDh_12_26_9")
            macd_against = (macdh is not None and ((macdh < 0) if side == "BUY" else (macdh > 0)))
            if ema_bearish and macd_against:
                conviction += 1.0
                reasons.append("EMA alignment lost + MACD against the position")

        # 3) Dead money — flat P&L while a big chunk of the window has burned.
        if hold_frac is not None and hold_frac >= 0.5 and unreal_pct is not None \
                and abs(unreal_pct) < 1.0:
            conviction += 1.5
            reasons.append(f"flat P&L ({unreal_pct:+.1f}%) — dead money")

        conviction = round(min(conviction, 10.0), 1)

        # ── verdict ──
        too_new = bars_held < self.min_hold_bars
        source = "rules"
        claude_reasoning = ""

        if too_new:
            verdict = "HOLD"
            reasons.append(f"only {bars_held}d held — too new to review")
        elif conviction >= self.gate:
            verdict = "EXIT"
        elif conviction >= (self.gate - self.borderline_band):
            # Borderline — let the LLM break the tie if enabled.
            if self.use_claude and self._provider != "none":
                cl = self._ask_llm(pos, price, unreal_pct, t1_progress, bars_held,
                                   max_days, comp_score, comp_label, reasons)
                if cl:
                    source = "claude"
                    verdict = cl.get("verdict", "TRIM")
                    claude_reasoning = cl.get("reasoning", "")
                    # Blend the LLM's exit conviction with the local score (both 0-10).
                    cconf = cl.get("confidence")
                    if isinstance(cconf, (int, float)):
                        conviction = round(0.5 * conviction + 0.5 * float(cconf), 1)
                else:
                    verdict = "TRIM"
            else:
                verdict = "TRIM"
        else:
            verdict = "HOLD"

        # A clean HOLD with no adverse flags would otherwise have an empty "why" —
        # spell out the affirmative reason so every position carries an explanation.
        if not reasons:
            bits = []
            if comp_label and comp_label != "UNKNOWN":
                bits.append(f"composite {comp_label}"
                            + (f" {comp_score:.1f}/10" if comp_score is not None else ""))
            if unreal_pct is not None:
                bits.append(f"{unreal_pct:+.1f}% P&L")
            if t1_progress is not None:
                bits.append(f"{max(t1_progress, 0.0) * 100:.0f}% to T1")
            reasons.append(
                ("healthy — " + ", ".join(bits) + "; no staleness flags") if bits
                else "no staleness flags — momentum intact"
            )

        return {
            "symbol": sym,
            "strategy": strategy,
            "side": side,
            "verdict": verdict,
            "exit_conviction": conviction,
            "bars_held": bars_held,
            "max_days": max_days,
            "t1_progress_pct": round(t1_progress * 100, 1) if t1_progress is not None else None,
            "unrealized_pct": unreal_pct,
            "composite_score": comp_score,
            "composite_label": comp_label,
            "reasons": reasons,
            "source": source,
            "claude_reasoning": claude_reasoning,
            "action": None,  # filled by _apply_actions
            "realized_pnl": None,
        }

    # ── acting on reviews ───────────────────────────────────────────────────
    def _apply_actions(self, reviews: List[dict], live_prices: Dict[str, float],
                       portfolio, can_auto: bool, dry_run: bool) -> int:
        acted = 0
        for r in reviews:
            if r["verdict"] != "EXIT":
                r["action"] = "HOLD" if r["verdict"] == "HOLD" else "RECOMMEND_TRIM"
                continue

            price = live_prices.get(r["symbol"])
            eligible = (can_auto and not dry_run
                        and r["exit_conviction"] >= self.gate
                        and price is not None)
            if not eligible:
                r["action"] = "RECOMMEND_EXIT"
                continue

            # Auto-close via the standard paper-trade exit path.
            pos = next((p for p in portfolio.get_open_positions()
                        if p["symbol"] == r["symbol"]), None)
            if not pos:
                r["action"] = "RECOMMEND_EXIT"
                continue
            pnl = portfolio.close_position(pos["id"], price, "STALE_EXIT")
            if pnl is None:
                r["action"] = "RECOMMEND_EXIT"
            else:
                r["action"] = "AUTO_EXITED"
                r["realized_pnl"] = round(pnl, 2)
                acted += 1
        return acted

    # ── LLM tie-break ───────────────────────────────────────────────────────
    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            if self._provider == "anthropic":
                import anthropic
                self._model = self._claude_cfg.get("model")
                self._max_tokens = int(self._claude_cfg.get("max_tokens", 512))
                self._client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=30.0)
            elif self._provider == "openai":
                import openai
                self._model = self._openai_cfg.get("model")
                self._max_tokens = int(self._openai_cfg.get("max_tokens", 512))
                self._client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"), timeout=30.0)
            else:
                return False
        except Exception as e:
            self.log_warning(f"LLM client init failed — borderline cases default to TRIM: {e}")
            return False
        return True

    def _ask_llm(self, pos, price, unreal_pct, t1_progress, bars_held, max_days,
                 comp_score, comp_label, reasons) -> Optional[dict]:
        if not self._ensure_client():
            return None
        prompt = (
            f"Open position review:\n"
            f"Symbol: {pos['symbol']}\n"
            f"Direction: {pos.get('signal_type', 'BUY')} | Strategy: {pos.get('strategy')}\n"
            f"Entry: ₹{pos['entry_price']} | Current: ₹{price}\n"
            f"Unrealized: {unreal_pct}% | Progress to Target-1: "
            f"{round(t1_progress*100,1) if t1_progress is not None else 'n/a'}%\n"
            f"Held: {bars_held} of {max_days} max-hold days\n"
            f"Composite TA now: {comp_score}/10 ({comp_label})\n"
            f"Local staleness flags: {'; '.join(reasons) or 'none'}\n\n"
            f"Should this capital keep being held? Respond with JSON only."
        )
        try:
            if self._provider == "anthropic":
                resp = self._client.messages.create(
                    model=self._model, max_tokens=self._max_tokens,
                    system=[{"type": "text", "text": _REVIEW_SYSTEM_PROMPT,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": prompt}],
                )
                return json.loads(resp.content[0].text.strip())
            resp = self._client.chat.completions.create(
                model=self._model, max_tokens=self._max_tokens, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            self.log_warning(f"{pos['symbol']}: borderline LLM review failed — {e}")
            return None

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bars_held(entry_time: Optional[str]) -> int:
        if not entry_time:
            return 0
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            if entry_dt.tzinfo is None:
                entry_dt = _IST.localize(entry_dt)
            return max((datetime.now(_IST) - entry_dt).days, 0)
        except Exception:
            return 0

    @staticmethod
    def _closed_trade_count(portfolio) -> int:
        try:
            df = portfolio.get_trade_history(limit=100000)
            return 0 if df is None or df.empty else len(df)
        except Exception:
            return 0

    def _persist(self, reviews: List[dict], closed_count: int,
                 can_auto: bool = False, acted: int = 0) -> None:
        try:
            _REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(_IST).isoformat(),
                "closed_trades": closed_count,
                "auto_exit_active": can_auto,
                "auto_exit_min_confidence": self.gate,
                "min_closed_trades_for_auto_exit": self.min_closed_trades,
                "auto_closed": acted,
                "reviews": reviews,
            }
            with open(_REVIEWS_PATH, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            self.log_warning(f"Could not persist position reviews: {e}")
