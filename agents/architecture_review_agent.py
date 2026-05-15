"""
Evaluates proposed features against the current system architecture.
Uses AI (Anthropic or OpenAI) to analyse fit, complexity, and risk,
then generates a concrete implementation spec — per-agent change instructions
— so approved features can be built without ambiguity.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from agents.base_agent import BaseAgent

# ── Architecture snapshot ──────────────────────────────────────────────────

_ARCHITECTURE = """
## System Overview
NSE/BSE multi-agent algorithmic trading system — Python 3.11+, Streamlit dashboard,
LangGraph StateGraph orchestration, SQLite storage, paper trading only (no live broker yet).

## Agent Pipeline (sequential LangGraph nodes)
fetch_data → run_ta → fetch_sentiment → fetch_composite_ratings
→ generate_signals → validate_with_claude → check_risk
→ execute_paper_trades → send_notifications

## Agents (agents/)
DataAgent           data_agent.py              Fetches OHLCV via yfinance; manages Nifty 50 watchlist
TechnicalAnalysisAgent technical_analysis_agent.py Computes 30+ indicators: RSI, MACD, Bollinger Bands, EMA(9/21/50/200), ATR, ADX, OBV
SentimentAgent      sentiment_agent.py         Scrapes NSE RSS + Google Finance; per-symbol sentiment label + headline
CompositeTAAgent    tradingview_ta_agent.py    Computes in-house composite TA ratings (STRONG_BUY…STRONG_SELL) from enriched indicators; no external dep.
SignalAgent         signal_agent.py            Generates BUY/SELL signals via swing/intraday/positional strategies; applies composite TA score adj
ClaudeValidationAgent claude_validation_agent.py Validates signals via Anthropic Claude or OpenAI GPT-4o; APPROVE/REJECT/MODIFY + reasoning
RiskAgent           risk_agent.py              Position sizing, max exposure, daily loss limit, portfolio-level risk controls
PortfolioAgent      portfolio_agent.py         Paper trades in SQLite; tracks open/closed positions, equity, P&L, win rate
NotificationAgent   notification_agent.py      Telegram alerts for approved signals; multi-recipient; EOD P&L report

## Orchestration (orchestrator/workflow.py)
LangGraph StateGraph; TradingState TypedDict shared across all nodes.
APScheduler: 9:15 AM IST morning scan | 15-min intraday monitor 9:30–3:15 PM | 3:45 PM EOD report.
All jobs check is_market_open() / is_trading_day() before executing.

## Database (SQLite via SQLAlchemy, data/trading.db)
signals_log  : timestamp, symbol, strategy, timeframe, signal_type, entry_price, stop_loss,
               target_1, target_2, confidence, claude_verdict, claude_reasoning, composite_rating, status
portfolio    : symbol, strategy, entry_price, quantity, stop_loss, target_1, target_2, entry_time, current_price
trade_history: symbol, strategy, entry_price, exit_price, quantity, pnl, entry_time, exit_time, reason, signal_id

## Configuration (config/trading_config.yaml)
Watchlist, strategy params, confidence thresholds (min 6.0 for AI validation, 7.0 to trade),
risk controls, AI provider (auto/anthropic/openai), market timing.

## Dashboard (dashboard/app.py — Streamlit, 7 tabs)
Market Pulse | Portfolio | Technical Chart | Live Signals | Backtesting | Watchlist | Settings
Auto-refresh 30 s during market hours. Sidebar: account summary, AI provider, quick actions.

## External Integrations
yfinance (market data) | in-house composite TA (no external dep.) | Anthropic / OpenAI API (validation) | Telegram Bot API (notifications)

## Known Gaps / Improvement Opportunities
- No feedback loop: trade outcomes never read back into signal generation or AI prompts
- Static confidence thresholds: not adapted based on historical strategy performance
- AI validation is stateless: no history of past signal accuracy fed to model
- No live broker: Zerodha Kite / Angel One hooks stubbed in .env, not wired
- No options data: strategy config exists but the agent is disabled
- No scan summary notification: Telegram only fires when signals are APPROVED; silent scans give no feedback
"""

_SYSTEM_PROMPT = (
    "You are a senior software architect reviewing proposed features for an algorithmic trading system.\n\n"
    "CURRENT SYSTEM ARCHITECTURE:\n"
    + _ARCHITECTURE
    + "\n\nYour job: evaluate whether a proposed feature should be built, analyse its fit with the existing "
    "architecture, estimate complexity and risk, identify exactly which components need to change, and produce "
    "a concrete implementation spec that another developer can act on immediately.\n\n"
    "Guidelines:\n"
    "- Be decisive. Reject features that add complexity without clear trading value.\n"
    "- Prefer changes that reuse existing agents and infrastructure over new abstractions.\n"
    "- Flag anything that risks breaking the real-time pipeline or introducing latency.\n"
    "- If the feature is good but the request is vague or over-engineered, return MODIFY with a tighter scope.\n"
    "- Respond ONLY with valid JSON — no markdown fences, no extra text.\n\n"
    "Response schema (all fields required):\n"
    "{\n"
    '  "verdict": "APPROVE" | "REJECT" | "MODIFY",\n'
    '  "summary": "<one-sentence verdict>",\n'
    '  "rationale": "<2-3 paragraphs of reasoning>",\n'
    '  "value_score": <integer 1-10>,\n'
    '  "risk_score": <integer 1-10>,\n'
    '  "complexity": "LOW" | "MEDIUM" | "HIGH",\n'
    '  "estimated_effort": "<e.g. 3-5 hours>",\n'
    '  "affected_components": [\n'
    '    {\n'
    '      "file": "<relative file path>",\n'
    '      "agent": "<class or module name>",\n'
    '      "current_role": "<what it does now>",\n'
    '      "changes": ["<specific change instruction 1>", "<specific change instruction 2>"],\n'
    '      "priority": "must" | "nice-to-have"\n'
    '    }\n'
    '  ],\n'
    '  "new_files": [\n'
    '    { "file": "<relative path>", "purpose": "<what this new file should do>" }\n'
    '  ],\n'
    '  "implementation_plan": ["<ordered step 1>", "<ordered step 2>"],\n'
    '  "risks": ["<risk 1>", "<risk 2>"],\n'
    '  "dependencies": ["<new pip package if any>"],\n'
    '  "modified_request": null,\n'
    '  "alternative": null\n'
    "}\n"
    "Set modified_request to a refined feature description string if verdict is MODIFY (else null).\n"
    "Set alternative to a simpler approach string if the full request is overkill (else null)."
)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ComponentSpec:
    file: str
    agent: str
    current_role: str
    changes: List[str]
    priority: str  # "must" | "nice-to-have"


@dataclass
class ArchitectureReview:
    verdict: str
    summary: str
    rationale: str
    value_score: int
    risk_score: int
    complexity: str
    estimated_effort: str
    affected_components: List[ComponentSpec]
    new_files: List[dict]
    implementation_plan: List[str]
    risks: List[str]
    dependencies: List[str]
    modified_request: Optional[str]
    alternative: Optional[str]
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureReview":
        components = [
            ComponentSpec(
                file=c.get("file", ""),
                agent=c.get("agent", ""),
                current_role=c.get("current_role", ""),
                changes=c.get("changes", []),
                priority=c.get("priority", "must"),
            )
            for c in d.get("affected_components", [])
        ]
        return cls(
            verdict=d.get("verdict", "REJECT"),
            summary=d.get("summary", ""),
            rationale=d.get("rationale", ""),
            value_score=int(d.get("value_score", 5)),
            risk_score=int(d.get("risk_score", 5)),
            complexity=d.get("complexity", "MEDIUM"),
            estimated_effort=d.get("estimated_effort", "unknown"),
            affected_components=components,
            new_files=d.get("new_files", []),
            implementation_plan=d.get("implementation_plan", []),
            risks=d.get("risks", []),
            dependencies=d.get("dependencies", []),
            modified_request=d.get("modified_request"),
            alternative=d.get("alternative"),
            raw=d,
        )

    def to_dict(self) -> dict:
        return self.raw or {
            "verdict": self.verdict,
            "summary": self.summary,
            "value_score": self.value_score,
            "risk_score": self.risk_score,
            "complexity": self.complexity,
        }


# ── Agent ──────────────────────────────────────────────────────────────────

class ArchitectureReviewAgent(BaseAgent):
    """Analyses feature proposals and produces implementation specs for approved features."""

    def __init__(self):
        super().__init__("ArchReview")
        cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        preferred = cfg.get("validation", {}).get("provider", "auto")
        self._provider = self._resolve_provider(preferred)
        self._max_tokens = 2048

        if self._provider == "anthropic":
            import anthropic as _anthropic
            self._model = cfg["claude"]["model"]
            self._client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=60.0)
            self.log_info("Architecture review provider: Anthropic Claude")

        elif self._provider == "openai":
            import openai as _openai
            self._model = cfg["openai"]["model"]
            self._client = _openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=60.0)
            self.log_info("Architecture review provider: OpenAI GPT")

        else:
            self._model = None
            self._client = None
            self.log_warning("No AI API key — architecture review unavailable")

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, feature_request: str) -> Optional[ArchitectureReview]:
        """Evaluate a feature proposal. Returns ArchitectureReview or None if no AI available."""
        if not feature_request.strip():
            self.log_warning("Empty feature request")
            return None

        self.log_info(f"Reviewing feature: {feature_request[:80]}...")
        raw = self._call_ai(feature_request)
        if raw is None:
            return None

        review = ArchitectureReview.from_dict(raw)
        self.log_info(
            f"Review complete — verdict={review.verdict} "
            f"value={review.value_score}/10 risk={review.risk_score}/10 "
            f"complexity={review.complexity}"
        )
        return review

    def print_review(self, review: ArchitectureReview) -> None:
        """Print a formatted review to the console using Rich."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        verdict_color = {"APPROVE": "green", "REJECT": "red", "MODIFY": "yellow"}.get(review.verdict, "white")

        console.print()
        console.print(Panel(
            f"[bold {verdict_color}]{review.verdict}[/] — {review.summary}",
            title="[bold]Architecture Review[/]",
            border_style=verdict_color,
        ))

        # Scores
        score_table = Table(show_header=False, box=None, padding=(0, 2))
        score_table.add_column("Label", style="dim")
        score_table.add_column("Value", style="bold")
        score_table.add_row("Value Score", f"{review.value_score}/10")
        score_table.add_row("Risk Score", f"{review.risk_score}/10")
        score_table.add_row("Complexity", review.complexity)
        score_table.add_row("Estimated Effort", review.estimated_effort)
        console.print(score_table)

        # Rationale
        console.print()
        console.print("[bold cyan]Rationale[/]")
        console.print(review.rationale)

        # Modified / Alternative
        if review.modified_request:
            console.print()
            console.print(Panel(review.modified_request, title="[yellow]Refined Request[/]", border_style="yellow"))
        if review.alternative:
            console.print()
            console.print(Panel(review.alternative, title="[dim]Simpler Alternative[/]", border_style="dim"))

        # Affected components
        if review.affected_components:
            console.print()
            console.print("[bold cyan]Affected Components[/]")
            for comp in review.affected_components:
                tag = "[green]MUST[/]" if comp.priority == "must" else "[dim]OPTIONAL[/]"
                console.print(f"  {tag} [bold]{comp.agent}[/] ({comp.file})")
                for change in comp.changes:
                    console.print(f"    · {change}")

        # New files
        if review.new_files:
            console.print()
            console.print("[bold cyan]New Files[/]")
            for nf in review.new_files:
                console.print(f"  [bold]{nf.get('file')}[/] — {nf.get('purpose')}")

        # Implementation plan
        if review.implementation_plan:
            console.print()
            console.print("[bold cyan]Implementation Plan[/]")
            for i, step in enumerate(review.implementation_plan, 1):
                console.print(f"  {i}. {step}")

        # Risks
        if review.risks:
            console.print()
            console.print("[bold yellow]Risks[/]")
            for risk in review.risks:
                console.print(f"  [!] {risk}")

        # Dependencies
        if review.dependencies:
            console.print()
            console.print("[bold cyan]New Dependencies[/]")
            for dep in review.dependencies:
                console.print(f"  pip install {dep}")

        console.print()

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_provider(preferred: str) -> str:
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if preferred == "anthropic":
            return "anthropic" if has_anthropic else "none"
        if preferred == "openai":
            return "openai" if has_openai else "none"
        if has_anthropic:
            return "anthropic"
        if has_openai:
            return "openai"
        return "none"

    def _call_ai(self, feature_request: str) -> Optional[dict]:
        user_msg = (
            f"Evaluate this proposed feature for the trading system:\n\n"
            f"{feature_request}\n\n"
            f"Respond with JSON only."
        )
        if self._provider == "anthropic":
            return self._call_anthropic(user_msg)
        if self._provider == "openai":
            return self._call_openai(user_msg)
        self.log_error("No AI provider configured — cannot run architecture review")
        return None

    def _call_anthropic(self, user_msg: str) -> Optional[dict]:
        import anthropic as _anthropic
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
            )
            return json.loads(response.content[0].text.strip())
        except json.JSONDecodeError:
            self.log_error("Anthropic returned invalid JSON")
            return None
        except _anthropic.APIError as e:
            self.log_error(f"Anthropic API error: {e}")
            return None

    def _call_openai(self, user_msg: str) -> Optional[dict]:
        import openai as _openai
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            return json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            self.log_error("OpenAI returned invalid JSON")
            return None
        except _openai.APIError as e:
            self.log_error(f"OpenAI API error: {e}")
            return None
