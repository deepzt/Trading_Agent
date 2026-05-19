# Feature Implementation To-Do List

Work on these **outside market hours** (before 9:00 AM or after 4:00 PM IST, Mon–Fri; or weekends).
Test each feature with `python main.py --scan` before the next 9:15 AM scheduled run.

---

## Pre-Requisite (Do This First)

### [ ] 0. RiskAgent context threading refactor
**Effort:** 2–3h  
**File:** `agents/risk_agent.py`  
**What:** Add `context: dict = None` parameter to `RiskAgent.run()` and `_check_signal()`.  
Also update the call site in `orchestrator/workflow.py:119`.  
**Why first:** Features 2, 4, and 7 all inject context into the risk check. Without this, all three require incompatible separate patches to the same method.

---

## Phase 1 — High Value, Low Risk (~24h total)

### [ ] 1. Earnings Blackout Window + Compliance Audit Trail
**Effort:** 12–16h | **Research:** SEBI PIT Regulations 2015; Ball & Brown 1968  
**New file:** `agents/earnings_calendar_agent.py`  
**Modified files:** `agents/risk_agent.py`, `orchestrator/workflow.py`, `dashboard/app.py`

- [ ] Fetch NSE corporate calendar from `https://www.nseindia.com/api/event-calendar` (reuse SentimentAgent session/header pattern)
- [ ] Determine if any watchlist symbol has a board meeting within next 3 **trading days** (not calendar days — needs holiday list)
- [ ] Source NSE holiday calendar as a static YAML (e.g., `config/nse_holidays.yaml`) for correct trading-day counting
- [ ] Add `earnings_blackout_symbols: set()` to `TradingState` and `_run_workflow()` initial state
- [ ] Add new node to workflow between `fetch_tv_ratings` and `compute_performance`
- [ ] Hard block in `RiskAgent._check_signal()`: reject positional/swing signals for blackout symbols
- [ ] Log blocked signals with `claude_verdict = "EARNINGS_BLACKOUT"` in signals_log
- [ ] Add `EARNINGS BLACKOUT` column/highlight in Streamlit Live Signals tab
- [ ] Full try/except with fallback (empty set) on NSE API failure — must not crash pipeline

---

### [x] 2. Cross-Sectional Sector-Neutral Momentum Ranking ✓ DONE
**Effort:** 8–10h | **Research:** Jegadeesh & Titman 1993; Asness et al. 2013  
**New method:** `compute_momentum_rank()` in or alongside `agents/performance_tracker.py`  
**Modified files:** `agents/signal_agent.py`, `orchestrator/workflow.py`

- [x] Compute 12-1 month return per symbol from already-fetched OHLCV (days -252 to -21)
- [x] Rank within each sector using existing `SECTORS` dict in `strategies/positional.py`
- [x] Symbols not in SECTORS go into a synthetic "OTHER" group (do not discard them)
- [x] Add `momentum_ranks: {}` to `TradingState` and `_run_workflow()` initial state
- [x] Compute ranks in `compute_performance` node (data already available there) and store in state
- [x] Pass `momentum_ranks` to `SignalAgent.run()` in `generate_signals` node
- [x] Add condition in `SignalAgent._positional_signal()`: `sector_momentum_rank >= 0.6`
- [x] Fixed `fetch_data` days 365→400 to ensure 252+ trading rows for valid ranking
- [x] Expanded SECTORS to 16 sectors covering all 100 Nifty 100 symbols (0 in OTHER)

---

## Phase 2 — Market Context Enrichment (~35h total)

### [ ] 3. Market Regime Detection with India VIX Gating
**Effort:** 10–14h | **Research:** Ang et al. 2006; Banerjee & Sahadeb 2015  
**New file:** `agents/regime_detector.py`  
**Modified files:** `agents/sentiment_agent.py`, `agents/signal_agent.py`, `agents/risk_agent.py`, `orchestrator/workflow.py`

- [ ] Extend VIX history fetch in `SentimentAgent._get_market_context()` from 5 days to 30 days (needed for 20-day EMA)
- [ ] Add VIX thresholds to `config/trading_config.yaml` as config values (NOT hardcoded):
  - `vix_volatile_threshold: 18`
  - `vix_crisis_threshold: 24`
- [ ] Calibrate thresholds against 2 years of India VIX data from yfinance `^INDIAVIX` before going live
- [ ] `RegimeDetector.classify(vix, nifty_close, nifty_ema20)` → returns `"TRENDING"` / `"VOLATILE"` / `"CRISIS"`
- [ ] Add `regime: "TRENDING"` to `TradingState` and `_run_workflow()` initial state
- [ ] Add new node between `fetch_sentiment` and `fetch_tv_ratings` in the graph
- [ ] In `SignalAgent._intraday_signal()`: skip signal if regime == CRISIS
- [ ] In `RiskAgent`: raise ATR stop multiplier 1.5x → 2.0x when regime == VOLATILE (via context, not hardcode)
- [ ] Add regime badge to Market Pulse dashboard tab

---

### [ ] 4. FII/DII Flow Integration (Soft Advisory)
**Effort:** 12–15h | **Research:** Chakrabarti 2001; Bose & Reid 2011  
**Modified files:** `agents/sentiment_agent.py`, `agents/claude_validation_agent.py`  
**Note:** Approved as **soft advisory only** — inject into Claude prompt, NOT a hard RiskAgent block

- [ ] Fetch FII/DII data from `https://www.nseindia.com/api/fiidiiTradeReact` (reuse SentimentAgent session)
- [ ] Parse last 5 trading days of FII net buy/sell values
- [ ] Compute `fii_net_5d` (5-day rolling net in crores) and `dii_net_5d`
- [ ] Classify `fii_flow_regime`: `"INFLOW"` (avg > +500 Cr/day), `"NEUTRAL"`, `"OUTFLOW"` (avg < -500 Cr/day)
- [ ] Add `fii_flow_regime` to the `_market` dict in `SentimentAgent._get_market_context()`
- [ ] Inject it into Claude validation prompt in `ClaudeValidationAgent._build_prompt()`
- [ ] Cache TTL: 24h max (refresh each morning scan) — do NOT use the 7-day DataAgent cache
- [ ] Full try/except with fallback (`fii_flow_regime: "NEUTRAL"`) on API failure

---

### [x] 5. F&O Expiry Calendar Awareness ✓ DONE
**Effort:** 16–20h | **Research:** Bansal & Connolly 2019; Saini & Sehgal 2020  
**New file:** `agents/fno_calendar_agent.py`  
**Modified files:** `agents/risk_agent.py`, `orchestrator/workflow.py`  
**Note:** v1 uses datetime-only gating — drop NSE OI fetch until v2

- [x] Implement weekly expiry: every Thursday
- [x] Implement monthly expiry: last Thursday of each month
- [x] Handle Thursday holiday rollover: if Thursday is NSE holiday → expiry moves to Wednesday
- [x] Created `config/nse_holidays.yaml` with 2025–2026 NSE holiday list
- [x] Classify `expiry_risk`: `"NONE"` / `"EXPIRY_WEEK"` (within 48h) / `"EXPIRY_DAY"` (same day)
- [x] Add `expiry_context: {}` to `TradingState` and `_run_workflow()` initial state
- [x] Add new node `check_fno_expiry` to workflow (after check_earnings, before compute_performance)
- [x] In `RiskAgent._check_signal()` (via context): reject new positional signals within 48h of expiry
- [x] On expiry day: cap intraday positions at 1

---

## Deferred — Do Not Start Yet

### [ ] 6. Walk-Forward Parameter Adaptation
**Effort:** 35–50h  
**Blocked by:** Backtest engine uses fixed 3% SL/6% TP; live uses ATR-based stops. Optimization output won't transfer to live until these are aligned. Align backtest/live mechanics first (separate ~20–30h project), then revisit.

---

### [ ] 7. Intraday OFI (Order Flow Imbalance) Detector
**Effort:** 30–40h (full redesign)  
**Blocked by:** Current intraday signals are generated at 9:15 AM from daily data. OFI requires 5-minute data fetched during market hours. Needs a separate real-time intraday signal loop in `_run_intraday_monitor` — scope as a new architectural feature, not a bolt-on.

---

## Implementation Rules (follow every time)

1. **Always use venv:** `.\venv\Scripts\pip install` / `.\venv\Scripts\python`
2. **Add every new `TradingState` key** to both the `TypedDict` (workflow.py:35) and the `_run_workflow()` initial dict (workflow.py:258) — forgetting this crashes the workflow on first run
3. **Wrap all new NSE API calls** in try/except with a safe fallback return — a failed fetch must never crash the pipeline node
4. **Test with `python main.py --scan`** after each feature before the next 9:15 AM scheduled run
5. **Do not touch `streamlit==1.43.2` or `python-telegram-bot==13.15`** — both are pinned for compatibility reasons
