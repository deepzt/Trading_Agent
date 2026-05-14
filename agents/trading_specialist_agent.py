"""
Trading Specialist Agent — reviews this system's strategies, config, and live performance
against academic research literature and outputs prioritised improvement suggestions.

Run standalone:  python -m agents.trading_specialist_agent
Or via main.py:  python main.py --research-review
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import yaml

from agents.base_agent import BaseAgent

_ROOT = Path(__file__).parent.parent

_SYSTEM_PROMPT = """You are a quantitative trading research specialist with encyclopedic knowledge of academic finance literature. You are reviewing a live Python algorithmic paper trading system for Indian NSE/BSE markets and will suggest concrete, prioritised improvements backed by research evidence.

═══════════════════════════════════════════════════════
KNOWLEDGE BASE — KEY RESEARCH
═══════════════════════════════════════════════════════

── MOMENTUM FACTOR ────────────────────────────────────
• Jegadeesh & Titman (1993) — 3-12 month price momentum earns ~1%/month alpha. Skip last month.
• Carhart (1997) — Momentum is the 4th factor; cannot be explained by 3-factor model.
• Asness, Moskowitz & Pedersen (2013) — Momentum works across 8 asset classes. Combined
  with value it is most robust.
• Moskowitz, Ooi & Pedersen (2012) — Time-series momentum (own-asset 12M) is distinct from
  cross-sectional momentum. Both work.
• Daniel & Moskowitz (2016) — Momentum crashes during sharp market rebounds; volatility-
  scaling (Barroso & Santa-Clara 2015) removes most crash risk.
• Barroso & Santa-Clara (2015) — Scale momentum signal inversely by its recent volatility.

── VALUE FACTOR ───────────────────────────────────────
• Fama & French (1992, 1993) — HML (high book-to-market) earns ~4.9%/year premium.
• Lakonishok, Shleifer & Vishny (1994) — Value outperformance is behavioural overreaction,
  not pure risk compensation.
• Piotroski (2000) — F-Score (9 financial signals) sharply improves value stock selection.

── QUALITY / PROFITABILITY ────────────────────────────
• Novy-Marx (2013) — Gross profitability (GP/Assets) is a strong return predictor.
• Fama & French (2015) — 5-factor model: adds RMW (profitability) and CMA (investment).
• Asness et al. (2019) "Quality Minus Junk" — Profitable, growing, safe firms earn
  ~4%/year alpha. Quality screens reduce false positives in momentum strategies.

── LOW VOLATILITY ─────────────────────────────────────
• Frazzini & Pedersen (2014) "Betting Against Beta" — Low-beta stocks outperform on
  risk-adjusted basis. Security Market Line is flatter than CAPM predicts.
• Baker, Bradley & Wurgler (2011) — Institutional benchmark mandates cause low-vol anomaly.
• Blitz & van Vliet (2007) — Lowest-vol decile outperforms highest-vol by ~12%/year.

── TECHNICAL ANALYSIS (EMPIRICAL) ──────────────────────
• Brock, Lakonishok & LeBaron (1992) — Moving average and trading range breakout rules
  generate significant returns on DJIA data 1897-1986.
• Lo, Mamaysky & Wang (2000) — Chart patterns have measurable statistical content.
• Faber (2007) — 10-month SMA timing: in market when price > SMA, cash otherwise.
  Sharpe improves, max drawdown halves vs buy-and-hold.

── VOLUME & LIQUIDITY ─────────────────────────────────
• Lee & Swaminathan (2000) — Volume predicts momentum persistence: low-volume winners
  have longer-lasting momentum; high-volume winners reverse sooner.
• Amihud (2002) — Illiquidity premium: stocks with high price impact earn more long-run.
• Gervais, Kaniel & Mingelgrin (2001) — Unusual volume surge predicts positive 1-month return.

── EARNINGS / EVENTS ─────────────────────────────────
• Ball & Brown (1968); Bernard & Thomas (1989) — Post-earnings announcement drift (PEAD):
  prices drift in direction of earnings surprise for 60+ trading days.
• Chan, Jegadeesh & Lakonishok (1996) — Price momentum + earnings momentum together are
  stronger than either alone. Revision in analyst forecasts is additive signal.

── MEAN REVERSION ────────────────────────────────────
• De Bondt & Thaler (1985) — 3-5 year losers beat winners by 25%: long-horizon reversion.
• Gatev, Goetzmann & Rouwenhorst (2006) — Pairs trading earns ~11%/year market-neutral.

── BEHAVIORAL FINANCE ────────────────────────────────
• Kahneman & Tversky (1979) — Loss aversion (~2x): disposition effect causes investors
  to sell winners too early, hold losers too long. Strict stop-losses counteract this.
• Daniel, Hirshleifer & Subrahmanyam (1998) — Overconfidence drives momentum; self-
  attribution bias amplifies it, causing eventual reversal.
• Hong & Stein (1999) — Gradual information diffusion creates momentum; newsWatcher +
  momentum trader interaction causes sequential over/underreaction.

── RISK MANAGEMENT ───────────────────────────────────
• Kelly (1956) — Optimal bet = edge/odds. Half-Kelly used in practice due to estimation error.
• Tharp (1998) — Risk 1-2% per trade; position sizing is the most critical variable.
• Schwager (1989, 1992) — Market Wizards: common thread is cutting losses small, riding trends.
• Van Tharp — R-multiples: expectancy = win_rate × avg_win_R − loss_rate × 1.
• Maximum Adverse Excursion (MAE) — measure how far losses move before exit to calibrate stops.

── MARKET REGIMES ────────────────────────────────────
• Lo (2004) "Adaptive Markets Hypothesis" — Strategies work in some regimes, fail in others.
• Ilmanen (2011) "Expected Returns" — Comprehensive reference across asset classes and regimes.
• VIX regime classification: <14 low/trending; 14-20 normal; 20-25 elevated; >25 crisis.

── TREND FOLLOWING ───────────────────────────────────
• Hurst, Ooi & Pedersen (2013) — Trend following works across 100+ years; provides crisis alpha.
• Moskowitz, Ooi & Pedersen (2012) — TSMOM 12M signal works across 58 instruments globally.

── INDIA-SPECIFIC ────────────────────────────────────
• NSE F&O expiry: Monthly expiry (last Thursday) sees elevated volatility; Bank Nifty most affected.
• SGX/GIFT Nifty: Offshore futures price in global sentiment before NSE opens.
• FII vs DII flows: FII net buying/selling drives macro direction; DII provides counter-cyclical support.
• India VIX mean ~16; historically >20 = elevated; >25 = crisis/event risk.
• RBI MPC: bi-monthly policy; surprises move Bank Nifty ±2-4% intraday.
• Sector rotation: IT outperforms on USD strength; FMCG defensive; Banking leads bull markets.

═══════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════
You will receive:
1. The system's current strategy config (YAML)
2. Risk config (YAML)
3. Recent live performance stats
4. A summary of current implementation features

Produce a structured improvement report with these sections:

## Signal Generation Improvements
## Risk Management Improvements
## Filter / Quality Improvements
## Regime Adaptation Improvements
## India-Market-Specific Improvements

For each suggestion:
- State the specific change (concrete, implementable)
- Cite the research paper(s) that support it
- Estimate the expected impact (High / Medium / Low)
- Flag if it conflicts with any existing feature

Prioritise by impact. Be direct — no filler. Target 8-12 total suggestions across all sections.
"""


class TradingSpecialistAgent(BaseAgent):
    """Reviews the trading system against research literature and outputs improvement suggestions."""

    def __init__(self):
        super().__init__("TradingSpecialist")
        self._client = None
        self._provider = self._init_client()

    def _init_client(self) -> str:
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                return "anthropic"
            except Exception:
                pass
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai
                self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                return "openai"
            except Exception:
                pass
        return "none"

    def run(self) -> str:
        """Read system state, call AI, return improvement report as markdown string."""
        if self._provider == "none":
            return "No AI provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."

        context = self._gather_system_context()
        prompt = self._build_prompt(context)
        self.log_info("Requesting research-backed improvement suggestions…")

        try:
            if self._provider == "anthropic":
                resp = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=3000,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            else:
                resp = self._client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=3000,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content
        except Exception as e:
            self.log_error(f"Research review failed: {e}")
            return f"Error: {e}"

    # ── Context gathering ──────────────────────────────────────────────────

    def _gather_system_context(self) -> dict:
        ctx: dict = {}

        # Strategy + risk configs
        try:
            ctx["trading_config"] = (_ROOT / "config" / "trading_config.yaml").read_text()
        except Exception:
            ctx["trading_config"] = "unavailable"
        try:
            ctx["risk_config"] = (_ROOT / "config" / "risk_config.yaml").read_text()
        except Exception:
            ctx["risk_config"] = "unavailable"

        # Live performance stats
        try:
            from agents.portfolio_agent import PortfolioAgent
            from agents.performance_tracker import PerformanceTracker
            portfolio = PortfolioAgent()
            tracker = PerformanceTracker()
            df = portfolio.get_trade_history(limit=500)
            stats = portfolio.get_performance_stats()
            strat_stats = tracker.compute_strategy_stats(df)
            ctx["performance_stats"] = json.dumps(stats, default=str, indent=2)
            ctx["strategy_stats"] = json.dumps(strat_stats, default=str, indent=2)
            ctx["total_closed_trades"] = len(df)
        except Exception:
            ctx["performance_stats"] = "unavailable"
            ctx["strategy_stats"] = "unavailable"
            ctx["total_closed_trades"] = 0

        # Tuning state
        try:
            tuning_path = _ROOT / "data" / "tuning_state.json"
            ctx["tuning_state"] = tuning_path.read_text() if tuning_path.exists() else "{}"
        except Exception:
            ctx["tuning_state"] = "{}"

        # Feature summary (what is already implemented)
        ctx["implemented_features"] = """
- EMA 9/21 crossover swing strategy with RSI 40-65 filter and ATR stops
- Intraday breakout above prior-day high with 1.5x volume surge requirement
- Positional weekly trend following with EMA 200, RSI 45-70, 12-1M momentum rank ≥60th pct
- TradingView TA cross-validation (STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL)
- Claude/GPT AI signal validation with APPROVE/REJECT/MODIFY verdicts
- Market regime detection: TRENDING / VOLATILE (VIX≥18) / CRISIS (VIX≥24)
- ATR stop widening 33% in VOLATILE regime; no intraday signals in CRISIS
- Earnings blackout: blocks swing/positional signals within 3 trading days of board meeting
- Sector-neutral 12-1 month momentum ranking; positional requires ≥60th percentile rank
- Auto-tuner: adjusts confidence thresholds based on strategy win-rates (min 10 trades)
- 2% account risk per trade, ATR-based stop placement, max position size cap
- Daily loss limit, max drawdown limit, max open positions limit
- Duplicate position guard: one open position per symbol at a time
"""
        return ctx

    def _build_prompt(self, ctx: dict) -> str:
        return f"""
Please review this trading system and suggest research-backed improvements.

## Trading Config
```yaml
{ctx['trading_config']}
```

## Risk Config
```yaml
{ctx['risk_config']}
```

## Live Performance ({ctx['total_closed_trades']} closed trades)
```json
{ctx['performance_stats']}
```

## Per-Strategy Stats
```json
{ctx['strategy_stats']}
```

## Already Implemented Features
{ctx['implemented_features']}

## Auto-Tuner State
```json
{ctx['tuning_state']}
```

Based on all of the above, produce your improvement report now.
"""


# ── Standalone entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = TradingSpecialistAgent()
    report = agent.run()
    print("\n" + "=" * 70)
    print("TRADING RESEARCH IMPROVEMENT REPORT")
    print("=" * 70 + "\n")
    print(report)
