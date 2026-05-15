"""
Trading Agent Dashboard — enhanced dark terminal theme.
Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import pytz

from agents.data_agent import DataAgent
from agents.portfolio_agent import PortfolioAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from monitoring.health import current_ist_time, is_market_open, is_trading_day

_IST = pytz.timezone("Asia/Kolkata")

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE/BSE Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark terminal CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide Streamlit toolbar, footer, and all branding */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stSidebarFooter"] { display: none !important; }
.viewerBadge_container__r5tak { display: none !important; }
.viewerBadge_link__qRIco { display: none !important; }
#stDecoration { display: none !important; }
div[style*="position: absolute"][style*="bottom: 20px"] { display: none !important; }

/* Base dark background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Remove default padding */
.block-container { padding: 0.8rem 1.5rem 1rem 1.5rem; max-width: 100%; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.75rem; letter-spacing: 0.06em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.4rem; font-weight: 700; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }

/* Tabs — target by ARIA role (stable across Streamlit versions) */
div[role="tablist"] {
    background-color: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 4px;
    padding: 0 4px;
}
button[role="tab"] {
    color: #c9d1d9 !important;
    background: transparent !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    opacity: 1 !important;
}
button[role="tab"]:hover {
    color: #ffffff !important;
    background: #21262d !important;
}
button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #58a6ff !important;
    background: #21262d !important;
}

/* DataFrames */
div[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }

/* Dividers */
hr { border-color: #21262d; }

/* Buttons */
.stButton > button { background: #21262d; border: 1px solid #30363d; color: #e6edf3; border-radius: 6px; font-size: 0.8rem; }
.stButton > button:hover { background: #30363d; border-color: #58a6ff; }

/* Select/Input */
div[data-baseweb="select"] { background: #161b22 !important; }
.stSelectbox > div, .stTextInput > div > div { background: #161b22 !important; border-color: #30363d !important; }

/* Headers */
h1 { color: #e6edf3; font-size: 1.2rem !important; font-weight: 700; margin: 0 !important; }
h2 { color: #e6edf3; font-size: 1rem !important; font-weight: 600; margin-bottom: 0.4rem !important; }
h3 { color: #8b949e; font-size: 0.85rem !important; font-weight: 500; margin-bottom: 0.3rem !important; }

/* Expanders */
details { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }

/* Ticker card style */
.ticker-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    font-family: monospace;
}
.ticker-name { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
.ticker-value { color: #e6edf3; font-size: 1.1rem; font-weight: 700; margin: 2px 0; }
.ticker-up { color: #3fb950; font-size: 0.75rem; font-weight: 600; }
.ticker-down { color: #f85149; font-size: 0.75rem; font-weight: 600; }
.ticker-neutral { color: #8b949e; font-size: 0.75rem; }

/* Signal cards */
.signal-buy {
    background: linear-gradient(90deg, #0d2818 0%, #161b22 100%);
    border: 1px solid #238636;
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.signal-sell {
    background: linear-gradient(90deg, #2d0f0f 0%, #161b22 100%);
    border: 1px solid #da3633;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.signal-symbol { color: #e6edf3; font-size: 1rem; font-weight: 700; }
.signal-meta { color: #8b949e; font-size: 0.75rem; margin: 4px 0; }
.signal-levels { display: flex; gap: 16px; margin-top: 8px; }
.level-badge {
    background: #21262d;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.75rem;
    font-family: monospace;
}
.conf-bar-bg { background: #21262d; border-radius: 4px; height: 4px; margin-top: 6px; }
.conf-bar-fill { height: 4px; border-radius: 4px; }

/* Info boxes */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
}
</style>
<script>
(function removeBranding() {
    function sweep() {
        document.querySelectorAll('*').forEach(function(el) {
            var t = el.innerText || '';
            if ((t.includes('Built with Claude') || t.includes('Anthropic')) && el.children.length === 0) {
                var p = el.parentElement;
                while (p && p.children.length <= 2) { p = p.parentElement; }
                if (p) p.style.display = 'none';
            }
        });
    }
    sweep();
    new MutationObserver(sweep).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

_PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", size=11),
    margin=dict(l=40, r=20, t=30, b=30),
)

# ── Cached data fetchers ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_indices():
    tickers = {
        "NIFTY 50":  "^NSEI",
        "SENSEX":    "^BSESN",
        "BANKNIFTY": "^NSEBANK",
        "INDIA VIX": "^INDIAVIX",
        "S&P 500":   "^GSPC",
        "GOLD":      "GC=F",
        "CRUDE OIL": "CL=F",
        "USD/INR":   "USDINR=X",
    }
    result = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d", interval="1d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                last = float(hist["Close"].iloc[-1])
                if prev == 0:
                    result[name] = {"price": last, "change": 0, "change_pct": 0}
                else:
                    chg = last - prev
                    chg_pct = (chg / prev) * 100
                    result[name] = {"price": last, "change": chg, "change_pct": chg_pct}
            elif len(hist) == 1:
                last = float(hist["Close"].iloc[-1])
                result[name] = {"price": last, "change": 0, "change_pct": 0}
        except Exception:
            result[name] = {"price": 0, "change": 0, "change_pct": 0}
    return result


@st.cache_data(ttl=600)
def fetch_sector_performance():
    sector_etfs = {
        "IT":        ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
        "Banking":   ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
        "FMCG":      ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS"],
        "Pharma":    ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS"],
        "Auto":      ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS"],
        "Metals":    ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
        "Energy":    ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"],
        "Realty":    ["ADANIENT.NS", "ADANIPORTS.NS"],
    }
    rows = []
    for sector, stocks in sector_etfs.items():
        changes = []
        for sym in stocks:
            try:
                h = yf.Ticker(sym).history(period="2d")
                if len(h) >= 2:
                    prev_close = h["Close"].iloc[-2]
                    if prev_close != 0:
                        chg = (h["Close"].iloc[-1] - prev_close) / prev_close * 100
                        changes.append(chg)
            except Exception:
                pass
        if changes:
            rows.append({"Sector": sector, "Change %": round(np.mean(changes), 2)})
    if not rows:
        return pd.DataFrame(columns=["Sector", "Change %"])
    return pd.DataFrame(rows).sort_values("Change %", ascending=False)


@st.cache_data(ttl=900)
def fetch_movers(symbols, top_n=5):
    movers = []
    for sym in symbols[:30]:
        try:
            h = yf.Ticker(f"{sym}.NS").history(period="2d")
            if len(h) >= 2:
                chg = (h["Close"].iloc[-1] - h["Close"].iloc[-2]) / h["Close"].iloc[-2] * 100
                movers.append({"symbol": sym, "change_pct": round(chg, 2), "price": round(h["Close"].iloc[-1], 2)})
        except Exception:
            pass
    df = pd.DataFrame(movers)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    gainers = df.nlargest(top_n, "change_pct")
    losers = df.nsmallest(top_n, "change_pct")
    return gainers, losers


@st.cache_data(ttl=900)
def fetch_ohlcv_for_chart(symbol, timeframe, days):
    agent = DataAgent()
    raw = agent.run([symbol], timeframe=timeframe, days=days)
    if not raw:
        return None
    ta = TechnicalAnalysisAgent()
    enriched = ta.run(raw)
    return enriched.get(symbol)


# ── Helper renderers ───────────────────────────────────────────────────────

def render_ticker_strip(indices: dict):
    keys = list(indices.keys())
    if not keys:
        st.warning("Live market data unavailable — Yahoo Finance may be rate-limiting. Try refreshing in a minute.")
        return
    cols = st.columns(len(keys))
    for col, name in zip(cols, keys):
        d = indices[name]
        price = d["price"]
        chg_pct = d["change_pct"]
        color_class = "ticker-up" if chg_pct > 0 else ("ticker-down" if chg_pct < 0 else "ticker-neutral")
        arrow = "▲" if chg_pct > 0 else ("▼" if chg_pct < 0 else "—")
        fmt = f"{price:,.0f}" if price > 100 else f"{price:.2f}"
        col.markdown(f"""
        <div class="ticker-card">
            <div class="ticker-name">{name}</div>
            <div class="ticker-value">{fmt}</div>
            <div class="{color_class}">{arrow} {chg_pct:+.2f}%</div>
        </div>""", unsafe_allow_html=True)


def render_signal_card(sig: dict):
    is_buy = sig.get("signal_type", "BUY") == "BUY"
    cls = "signal-buy" if is_buy else "signal-sell"
    verdict = sig.get("claude_verdict", "")
    verdict_badge = {"APPROVE": "✅ Approved", "REJECT": "❌ Rejected", "SKIP": "⏭ Skipped"}.get(verdict, verdict)
    conf = sig.get("confidence", 0)
    bar_color = "#3fb950" if conf >= 7 else ("#e3b341" if conf >= 5 else "#f85149")
    conf_pct = int(conf * 10)

    entry = sig.get("entry_price", 0)
    sl = sig.get("stop_loss", 0)
    t1 = sig.get("target_1", 0)
    t2 = sig.get("target_2", 0)
    reasoning = sig.get("claude_reasoning") or ""

    composite_rating = sig.get("composite_rating") or ""
    _composite_colors = {"STRONG_BUY": "#3fb950", "BUY": "#58c94b", "SELL": "#f85149", "STRONG_SELL": "#da3633", "NEUTRAL": "#8b949e"}
    composite_color = _composite_colors.get(composite_rating, "")
    composite_badge = (
        f'<span style="background:{composite_color}22; color:{composite_color}; border:1px solid {composite_color}; '
        f'border-radius:3px; padding:1px 6px; font-size:0.65rem; font-weight:600; margin-left:6px;">TA: {composite_rating}</span>'
        if composite_rating and composite_rating not in ("UNKNOWN", "") else ""
    )
    blackout_badge = (
        '<span style="background:#d2992222; color:#d29922; border:1px solid #d29922; '
        'border-radius:3px; padding:1px 6px; font-size:0.65rem; font-weight:600; margin-left:6px;">🔒 EARNINGS BLACKOUT</span>'
        if sig.get("_blackout") else ""
    )

    st.markdown(f"""
    <div class="{cls}">
        <div style="display:flex; justify-content:space-between; align-items:start;">
            <div>
                <span class="signal-symbol">{'🟢' if is_buy else '🔴'} {sig.get('symbol','')}</span>
                <span style="color:#8b949e; font-size:0.75rem; margin-left:8px;">{sig.get('strategy','').title()} · {sig.get('timeframe','')}</span>{composite_badge}{blackout_badge}
            </div>
            <span style="color:#8b949e; font-size:0.72rem;">{str(sig.get('timestamp',''))[:19]}</span>
        </div>
        <div style="display:flex; gap:10px; margin-top:8px; flex-wrap:wrap;">
            <span class="level-badge" style="border-left:3px solid #58a6ff;">Entry ₹{entry:,.2f}</span>
            <span class="level-badge" style="border-left:3px solid #f85149;">SL ₹{sl:,.2f}</span>
            <span class="level-badge" style="border-left:3px solid #e3b341;">T1 ₹{t1:,.2f}</span>
            <span class="level-badge" style="border-left:3px solid #3fb950;">T2 ₹{t2:,.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
            <span style="color:#8b949e; font-size:0.72rem;">{verdict_badge} &nbsp;·&nbsp; Confidence {conf}/10</span>
        </div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%; background:{bar_color};"></div></div>
        {f'<div style="color:#8b949e; font-size:0.72rem; margin-top:6px; font-style:italic; line-height:1.5;">🤖 {reasoning}</div>' if reasoning else ''}
    </div>""", unsafe_allow_html=True)


def candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Full technical chart: candlestick + EMAs + BBands + Volume + RSI + MACD."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("", "Volume", "RSI (14)", "MACD (12,26,9)"),
    )

    # ── Row 1: Candlestick ─────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=symbol, increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
        line_width=1,
    ), row=1, col=1)

    ema_colors = {"EMA_9": "#f0e68c", "EMA_21": "#87ceeb", "EMA_50": "#ffa07a", "EMA_200": "#da70d6"}
    for col_name, color in ema_colors.items():
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=col_name.replace("_", " "),
                line=dict(color=color, width=1.2, dash="solid"),
                opacity=0.85,
            ), row=1, col=1)

    if "BBU_20_2.0" in df.columns and "BBL_20_2.0" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBU_20_2.0"], name="BB Upper",
            line=dict(color="#4a9eff", width=0.8, dash="dash"), opacity=0.5, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBL_20_2.0"], name="BB Lower",
            line=dict(color="#4a9eff", width=0.8, dash="dash"), opacity=0.5,
            fill="tonexty", fillcolor="rgba(74,158,255,0.04)", showlegend=False,
        ), row=1, col=1)

    # ── Row 2: Volume ──────────────────────────────────────────────────────
    close_s = df["close"].fillna(method="ffill").fillna(0)
    open_s  = df["open"].fillna(method="ffill").fillna(0)
    colors = ["#3fb950" if c >= o else "#f85149"
              for c, o in zip(close_s, open_s)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.7, showlegend=False,
    ), row=2, col=1)
    if "volume_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["volume_ma20"], name="Vol MA20",
            line=dict(color="#e3b341", width=1.2), showlegend=False,
        ), row=2, col=1)

    # ── Row 3: RSI ─────────────────────────────────────────────────────────
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"], name="RSI",
            line=dict(color="#58a6ff", width=1.5), showlegend=False,
        ), row=3, col=1)
        for level, color, dash in [(70, "#f85149", "dash"), (30, "#3fb950", "dash"), (50, "#4a4f5a", "dot")]:
            fig.add_hline(y=level, line_color=color, line_dash=dash, line_width=0.8, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="#f85149", opacity=0.06, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="#3fb950", opacity=0.06, row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    # ── Row 4: MACD ────────────────────────────────────────────────────────
    if all(c in df.columns for c in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]):
        hist = df["MACDh_12_26_9"]
        bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist.fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=hist, name="MACD Hist",
            marker_color=bar_colors, opacity=0.8, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_12_26_9"], name="MACD",
            line=dict(color="#58a6ff", width=1.2), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACDs_12_26_9"], name="Signal",
            line=dict(color="#f0e68c", width=1.2), showlegend=False,
        ), row=4, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        **_PLOTLY_THEME,
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="#21262d", showgrid=True, zeroline=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.25], pattern="hour"),  # exclude non-market hours
        ],
    )
    fig.update_yaxes(gridcolor="#21262d", showgrid=True, zeroline=False)
    fig.update_traces(selector=dict(type="candlestick"), xhoverformat="%d %b '%y")
    return fig


def equity_drawdown_chart(history: pd.DataFrame, initial: float) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.65, 0.35],
        subplot_titles=("Portfolio Equity (₹)", "Drawdown (%)"),
    )
    df = history.sort_values("exit_time").copy()
    equity = initial + df["pnl"].cumsum()
    peak = equity.cummax().replace(0, np.nan)
    drawdown = ((equity - peak) / peak) * 100

    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=equity,
        name="Equity", line=dict(color="#3fb950", width=2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
    ), row=1, col=1)
    fig.add_hline(y=initial, line_color="#21262d", line_dash="dot", line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=drawdown,
        name="Drawdown", line=dict(color="#f85149", width=1.5),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(**_PLOTLY_THEME, height=420)
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    return fig


def monthly_pnl_heatmap(history: pd.DataFrame) -> go.Figure:
    df = history.copy()
    df["exit_time"] = pd.to_datetime(
        df["exit_time"].apply(
            lambda x: datetime.fromisoformat(x).replace(tzinfo=None) if isinstance(x, str) and x else None
        )
    )
    df["month"] = df["exit_time"].dt.strftime("%b")
    df["year"] = df["exit_time"].dt.year
    pivot = df.pivot_table(values="pnl", index="year", columns="month", aggfunc="sum")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in months if m in pivot.columns])

    max_abs = pivot.abs().max().max()
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        colorscale=[[0, "#f85149"], [0.5, "#21262d"], [1, "#3fb950"]],
        zmid=0, zmin=-max_abs, zmax=max_abs,
        text=[[f"₹{v:,.0f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{x} %{y}</b><br>P&L: ₹%{z:,.0f}<extra></extra>",
        colorbar=dict(title="P&L (₹)", thickness=12, len=0.8),
    ))
    fig.update_layout(**_PLOTLY_THEME, height=200, title="Monthly P&L Heatmap")
    return fig


def rsi_heatmap(snapshot: list) -> go.Figure:
    if not snapshot:
        return go.Figure()
    df = pd.DataFrame(snapshot)
    df["rsi_val"] = pd.to_numeric(df["rsi"], errors="coerce")
    df = df.dropna(subset=["rsi_val"]).sort_values("rsi_val")

    colors = []
    for r in df["rsi_val"]:
        if r >= 70:
            colors.append("#f85149")
        elif r >= 55:
            colors.append("#3fb950")
        elif r >= 45:
            colors.append("#e3b341")
        else:
            colors.append("#58a6ff")

    fig = go.Figure(go.Bar(
        x=df["symbol"], y=df["rsi_val"],
        marker_color=colors, text=df["rsi_val"].round(1),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>RSI: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=70, line_color="#f85149", line_dash="dash", line_width=1)
    fig.add_hline(y=30, line_color="#3fb950", line_dash="dash", line_width=1)
    fig.add_hline(y=50, line_color="#4a4f5a", line_dash="dot", line_width=1)
    fig.update_layout(
        **_PLOTLY_THEME, height=320,
        title="RSI Heatmap — Watchlist",
        xaxis_title="", yaxis_title="RSI",
        yaxis_range=[0, 100],
        bargap=0.2,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding
    st.markdown(
        "<div style='padding: 8px 0 12px 0;'>"
        "<div style='font-size:1rem; font-weight:800; color:#e6edf3; letter-spacing:0.04em;'>TRADING TERMINAL</div>"
        "<div style='font-size:0.7rem; color:#8b949e; margin-top:2px;'>NSE · BSE · Paper Mode</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Live portfolio snapshot
    try:
        _sb_portfolio = PortfolioAgent()
        _sb_stats = _sb_portfolio.get_performance_stats()
        _sb_daily = _sb_portfolio.get_daily_stats()
        _sb_initial = float(os.getenv("ACCOUNT_SIZE", "100000"))
        _sb_equity = _sb_stats.get("current_equity", _sb_initial)
        _sb_pnl = _sb_stats.get("total_pnl", 0)
        _sb_daily_pnl = _sb_daily.get("daily_pnl", 0)
        _sb_open = len(_sb_portfolio.get_open_positions())

        pnl_color = "#3fb950" if _sb_pnl >= 0 else "#f85149"
        daily_color = "#3fb950" if _sb_daily_pnl >= 0 else "#f85149"

        st.markdown(
            f"<div style='font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>Portfolio</div>"
            f"<div style='font-size:1.15rem; font-weight:700; color:#e6edf3; font-family:monospace;'>₹{_sb_equity:,.0f}</div>"
            f"<div style='font-size:0.78rem; color:{pnl_color}; margin-top:2px;'>{'+' if _sb_pnl>=0 else ''}₹{_sb_pnl:,.0f} total</div>"
            f"<div style='font-size:0.72rem; color:{daily_color};'>{'+' if _sb_daily_pnl>=0 else ''}₹{_sb_daily_pnl:,.0f} today</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Win Rate", f"{_sb_stats.get('win_rate', 0)}%")
        col_b.metric("Open", f"{_sb_open} pos")
    except Exception:
        pass

    st.divider()

    # Actions
    import time as _time
    if "last_scan_time" not in st.session_state:
        st.session_state.last_scan_time = 0
    _cooldown = 300
    _since_last = _time.time() - st.session_state.last_scan_time
    if st.button("▶  Run Signal Scan", use_container_width=True):
        if _since_last < _cooldown:
            st.warning(f"Scan cooldown: wait {int(_cooldown - _since_last)}s before next scan.")
        else:
            st.session_state.last_scan_time = _time.time()
            with st.spinner("Scanning market..."):
                try:
                    from orchestrator.workflow import TradingScheduler
                    da = DataAgent()
                    symbols = da.get_watchlist("nifty50")[:20]
                    TradingScheduler(symbols).run_now()
                    st.success("Done — check Live Signals tab.")
                except Exception as e:
                    st.error("Signal scan failed. Check server logs for details.")

    if st.button("↺  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # System status
    market_open = is_market_open()
    trading_on = os.getenv("TRADING_ENABLED", "true").lower() != "false"
    st.markdown(
        f"<div style='font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>System</div>"
        f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
        f"<span style='font-size:0.75rem; color:#8b949e;'>Market</span>"
        f"<span style='font-size:0.75rem; font-weight:600; color:{'#3fb950' if market_open else '#f85149'};'>"
        f"{'OPEN' if market_open else 'CLOSED'}</span></div>"
        f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
        f"<span style='font-size:0.75rem; color:#8b949e;'>Mode</span>"
        f"<span style='font-size:0.75rem; font-weight:600; color:#e3b341;'>PAPER</span></div>"
        f"<div style='display:flex; justify-content:space-between;'>"
        f"<span style='font-size:0.75rem; color:#8b949e;'>Trading</span>"
        f"<span style='font-size:0.75rem; font-weight:600; color:{'#3fb950' if trading_on else '#f85149'};'>"
        f"{'ON' if trading_on else 'OFF'}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='position:absolute; bottom:20px; font-size:0.65rem; color:#484f58;'>"
        "Built with Claude · Anthropic</div>",
        unsafe_allow_html=True,
    )


# ── Main tabs ──────────────────────────────────────────────────────────────

TABS = ["📡 Market Pulse", "💼 Portfolio", "📊 Technical Chart",
        "🔔 Live Signals", "🔬 Backtesting", "📋 Watchlist", "⚙️ Settings", "📈 System Health"]
tabs = st.tabs(TABS)

# ── Tab 1: Market Pulse ────────────────────────────────────────────────────

with tabs[0]:
    st.markdown("#### Live Market Overview")

    with st.spinner("Fetching live data..."):
        indices = fetch_indices()

    # Regime badge derived from live VIX
    _vix_val = indices.get("INDIA VIX", {}).get("price", 0) or 0
    try:
        _vix_float = float(_vix_val)
    except Exception:
        _vix_float = 0.0
    if _vix_float >= 24:
        _regime_label, _regime_color = "CRISIS", "#f85149"
    elif _vix_float >= 18:
        _regime_label, _regime_color = "VOLATILE", "#d29922"
    else:
        _regime_label, _regime_color = "TRENDING", "#3fb950"
    _regime_icon = {"CRISIS": "🔴", "VOLATILE": "🟡", "TRENDING": "🟢"}[_regime_label]
    st.markdown(
        f'<div style="display:inline-block;background:{_regime_color}22;border:1px solid {_regime_color};'
        f'border-radius:6px;padding:3px 12px;font-size:0.85rem;font-weight:600;color:{_regime_color};margin-bottom:8px;">'
        f'{_regime_icon} Market Regime: {_regime_label}'
        + (f' &nbsp;|&nbsp; VIX {_vix_float:.1f}' if _vix_float else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    render_ticker_strip(indices)
    st.divider()

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("#### Sector Performance (1D)")
        with st.spinner("Loading sectors..."):
            sector_df = fetch_sector_performance()
        if not sector_df.empty:
            colors = ["#3fb950" if v >= 0 else "#f85149" for v in sector_df["Change %"]]
            fig = go.Figure(go.Bar(
                x=sector_df["Change %"], y=sector_df["Sector"],
                orientation="h", marker_color=colors,
                text=[f"{v:+.2f}%" for v in sector_df["Change %"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
            ))
            fig.update_layout(**_PLOTLY_THEME, height=300, xaxis_title="% Change",
                              xaxis_zeroline=True, xaxis_zerolinecolor="#21262d",
                              xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector data loading...")

    with col_right:
        st.markdown("#### Top Movers (Nifty 50)")
        try:
            da = DataAgent()
            nifty = da.get_watchlist("nifty50")
            gainers, losers = fetch_movers(nifty, top_n=5)

            if not gainers.empty:
                st.markdown("<div style='color:#3fb950; font-size:0.78rem; font-weight:600;'>▲ TOP GAINERS</div>", unsafe_allow_html=True)
                for _, row in gainers.iterrows():
                    st.markdown(
                        f"<div class='info-box' style='display:flex; justify-content:space-between;'>"
                        f"<span style='font-weight:600; font-size:0.85rem;'>{row['symbol']}</span>"
                        f"<span style='font-family:monospace; font-size:0.85rem;'>₹{row['price']:,.2f}"
                        f" <span style='color:#3fb950;'>+{row['change_pct']:.2f}%</span></span>"
                        f"</div>", unsafe_allow_html=True
                    )
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            if not losers.empty:
                st.markdown("<div style='color:#f85149; font-size:0.78rem; font-weight:600;'>▼ TOP LOSERS</div>", unsafe_allow_html=True)
                for _, row in losers.iterrows():
                    st.markdown(
                        f"<div class='info-box' style='display:flex; justify-content:space-between;'>"
                        f"<span style='font-weight:600; font-size:0.85rem;'>{row['symbol']}</span>"
                        f"<span style='font-family:monospace; font-size:0.85rem;'>₹{row['price']:,.2f}"
                        f" <span style='color:#f85149;'>{row['change_pct']:.2f}%</span></span>"
                        f"</div>", unsafe_allow_html=True
                    )
        except Exception as e:
            st.warning(f"Movers unavailable: {e}")

    st.divider()

    # Market breadth
    st.markdown("#### Market Breadth — Nifty 50 vs 200 EMA")
    with st.spinner("Computing breadth..."):
        try:
            da = DataAgent()
            ta = TechnicalAnalysisAgent()
            nifty_syms = da.get_watchlist("nifty50")[:25]
            raw = da.run(nifty_syms, timeframe="1d", days=220)
            enriched = ta.run(raw)
            above, below, total = 0, 0, 0
            for sym, df in enriched.items():
                if "above_ema200" in df.columns and not df.empty:
                    total += 1
                    if bool(df["above_ema200"].iloc[-1]):
                        above += 1
                    else:
                        below += 1
            if total > 0:
                breadth_pct = round((above / total) * 100, 1)
                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                bc1.metric("Above 200 EMA", f"{above}/{total}", f"{breadth_pct}%")
                bc2.metric("Breadth Score", f"{breadth_pct}%",
                           "Bullish" if breadth_pct > 60 else ("Bearish" if breadth_pct < 40 else "Neutral"))

                fig_breadth = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=breadth_pct,
                    title=dict(text="Market Breadth %", font=dict(color="#8b949e", size=12)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#8b949e"),
                        bar=dict(color="#58a6ff"),
                        bgcolor="#161b22",
                        steps=[
                            dict(range=[0, 40], color="#f85149"),
                            dict(range=[40, 60], color="#e3b341"),
                            dict(range=[60, 100], color="#3fb950"),
                        ],
                        threshold=dict(line=dict(color="#e6edf3", width=2), thickness=0.75, value=breadth_pct),
                    ),
                    number=dict(suffix="%", font=dict(color="#e6edf3", size=28)),
                ))
                fig_breadth.update_layout(**_PLOTLY_THEME, height=220)
                bc3.plotly_chart(fig_breadth, use_container_width=True)
        except Exception as e:
            st.info(f"Breadth calculation: {e}")


# ── Tab 2: Portfolio ───────────────────────────────────────────────────────

with tabs[1]:
    portfolio = PortfolioAgent()
    stats = portfolio.get_performance_stats()
    daily = portfolio.get_daily_stats()
    initial_capital = float(os.getenv("ACCOUNT_SIZE", "100000"))

    st.markdown("#### Portfolio Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_pnl = stats["total_pnl"]
    daily_pnl = daily["daily_pnl"]
    sharpe = stats["sharpe"]
    drawdown = stats["max_drawdown_pct"]

    c1.metric("Portfolio Equity", f"₹{stats['current_equity']:,.0f}",
              f"{total_pnl:+,.0f}",
              delta_color="normal")
    c2.metric("Today's P&L", f"₹{daily_pnl:+,.0f}",
              f"{daily['daily_pnl_pct']:+.2f}",
              delta_color="normal")
    c3.metric("Win Rate", f"{stats['win_rate']}%",
              f"{stats['total_trades']} trades",
              delta_color="off")
    sharpe_label = "Excellent" if sharpe > 1.5 else ("Good" if sharpe > 1 else ("Low" if sharpe >= 0 else "Negative"))
    c4.metric("Sharpe Ratio", f"{sharpe:.2f}",
              sharpe_label,
              delta_color="normal" if sharpe >= 0 else "inverse")
    c5.metric("Max Drawdown", f"{drawdown:.2f}%",
              delta_color="off")
    c6.metric("Avg Win / Loss", f"₹{stats['avg_win']:,.0f} / ₹{stats['avg_loss']:,.0f}",
              delta_color="off")

    history = portfolio.get_trade_history(limit=500)

    if not history.empty:
        st.plotly_chart(equity_drawdown_chart(history, initial_capital), use_container_width=True)
        st.plotly_chart(monthly_pnl_heatmap(history), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### P&L by Strategy")
            strat_pnl = history.groupby("strategy")["pnl"].agg(["sum", "count", "mean"]).reset_index()
            strat_pnl.columns = ["Strategy", "Total P&L", "Trades", "Avg P&L"]
            colors = ["#3fb950" if v >= 0 else "#f85149" for v in strat_pnl["Total P&L"]]
            fig = go.Figure(go.Bar(
                x=strat_pnl["Strategy"], y=strat_pnl["Total P&L"],
                marker_color=colors,
                text=[f"₹{v:,.0f}" for v in strat_pnl["Total P&L"]],
                textposition="outside",
            ))
            fig.update_layout(**_PLOTLY_THEME, height=260, showlegend=False,
                              yaxis=dict(title="P&L (₹)", gridcolor="#21262d"),
                              xaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Trade P&L Distribution")
            wins = history[history["pnl"] > 0]["pnl"]
            losses = history[history["pnl"] <= 0]["pnl"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=wins, name="Wins", marker_color="#3fb950", opacity=0.75, nbinsx=20))
            fig.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#f85149", opacity=0.75, nbinsx=20))
            fig.update_layout(**_PLOTLY_THEME, height=260, barmode="overlay",
                              xaxis=dict(title="P&L (₹)", gridcolor="#21262d"),
                              yaxis=dict(title="Count", gridcolor="#21262d"),
                              legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

        # Fetch live prices once — reused by both breakdown and open positions table
        positions = portfolio.get_open_positions()
        live_data = {}
        for p in positions:
            sym = p["symbol"]
            if sym not in live_data:
                try:
                    t = yf.Ticker(f"{sym}.NS")
                    h = t.history(period="1d")
                    if not h.empty:
                        live_data[sym] = float(h["Close"].iloc[-1])
                except Exception:
                    pass

        # Stock breakdown — investment + consolidated P&L per symbol
        all_syms = sorted(set(
            [p["symbol"] for p in positions] +
            (list(history["symbol"].unique()) if not history.empty else [])
        ))
        breakdown_rows = []
        for sym in all_syms:
            open_sym = [p for p in positions if p["symbol"] == sym]
            invested = sum(p["entry_price"] * p["quantity"] for p in open_sym)
            unrealized = sum(
                (live_data.get(sym, p["entry_price"]) - p["entry_price"]) * p["quantity"]
                for p in open_sym
            )
            closed_sym = history[history["symbol"] == sym] if not history.empty else pd.DataFrame()
            realized = round(closed_sym["pnl"].sum(), 2) if not closed_sym.empty else 0.0
            closed_count = len(closed_sym)
            wins = len(closed_sym[closed_sym["pnl"] > 0]) if not closed_sym.empty else 0
            win_rate = round(wins / closed_count * 100, 1) if closed_count > 0 else None
            total_pnl = round(realized + unrealized, 2)
            breakdown_rows.append({
                "Symbol": sym,
                "Open Trades": len(open_sym),
                "Invested": f"₹{invested:,.0f}" if invested > 0 else "—",
                "Unrealized P&L": f"₹{unrealized:+,.0f}" if invested > 0 else "—",
                "Realized P&L": f"₹{realized:+,.0f}" if closed_count > 0 else "—",
                "Total P&L": f"₹{total_pnl:+,.0f}",
                "Closed Trades": closed_count,
                "Win Rate": f"{win_rate}%" if win_rate is not None else "—",
            })
        with st.expander("📊 Stock Breakdown — Investment & P&L per Symbol", expanded=False):
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

        # Open positions
        st.divider()
        st.markdown(f"#### Open Positions ({len(positions)})")
        if positions:
            pos_rows = []
            for p in positions:
                lp = live_data.get(p["symbol"], p["entry_price"])
                entry = p["entry_price"] or 0
                unreal = (lp - entry) * p["quantity"]
                unreal_pct = ((lp - entry) / entry * 100) if entry != 0 else 0.0
                pos_rows.append({
                    "Symbol": p["symbol"],
                    "Strategy": p["strategy"],
                    "Type": p["signal_type"],
                    "Entry ₹": p["entry_price"],
                    "Live ₹": round(lp, 2),
                    "SL ₹": p["stop_loss"],
                    "T1 ₹": p["target_1"],
                    "T2 ₹": p["target_2"],
                    "Qty": p["quantity"],
                    "Unreal P&L": f"₹{unreal:+,.0f} ({unreal_pct:+.1f}%)",
                    "Conf": p.get("confidence", 0),
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

        # Recent trades table
        st.markdown("#### Recent Trades")
        display = history.head(30).copy()
        display["P&L"] = display["pnl"].apply(lambda x: f"₹{x:+,.2f}")
        display["Result"] = display["pnl"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss")
        cols_show = [c for c in ["symbol","strategy","signal_type","entry_price","exit_price","P&L","Result","exit_reason","exit_time"] if c in display.columns]
        st.dataframe(display[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("No trade history yet. Run a signal scan to start paper trading.")


# ── Tab 3: Technical Chart ─────────────────────────────────────────────────

with tabs[2]:
    st.markdown("#### Technical Analysis Chart")
    col_sym, col_tf, col_days, col_btn = st.columns([2, 1.5, 1.5, 1])
    with col_sym:
        chart_symbol = st.text_input("Symbol (NSE)", "RELIANCE", label_visibility="collapsed",
                                     placeholder="e.g. RELIANCE, TCS, INFY")
    with col_tf:
        chart_tf = st.selectbox("Timeframe", ["1d", "1wk", "15m", "1h"], label_visibility="collapsed")
    with col_days:
        chart_days = st.selectbox("Period", [90, 180, 365, 730], format_func=lambda x: f"{x}d",
                                  label_visibility="collapsed")
    with col_btn:
        load_chart = st.button("Load Chart", use_container_width=True)

    if load_chart or "chart_loaded" not in st.session_state:
        st.session_state["chart_loaded"] = True
        with st.spinner(f"Loading {chart_symbol.upper()}..."):
            chart_df = fetch_ohlcv_for_chart(chart_symbol.upper(), chart_tf, chart_days)

        if chart_df is not None and not chart_df.empty:
            # Quick stats strip
            last = chart_df.iloc[-1]
            prev = chart_df.iloc[-2] if len(chart_df) > 1 else last
            day_chg = float(last["close"]) - float(prev["close"])
            day_chg_pct = (day_chg / float(prev["close"])) * 100

            s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
            s1.metric("Close", f"₹{last['close']:,.2f}", f"{day_chg:+.2f} ({day_chg_pct:+.2f}%)")
            s2.metric("Open", f"₹{last['open']:,.2f}")
            s3.metric("High", f"₹{last['high']:,.2f}")
            s4.metric("Low", f"₹{last['low']:,.2f}")
            def _safe_float(series_row, col):
                try:
                    v = float(series_row[col]) if col in series_row.index else None
                    return None if (v is None or not np.isfinite(v)) else v
                except (TypeError, ValueError):
                    return None

            rsi_val = _safe_float(last, "RSI_14")
            s5.metric("RSI (14)", f"{rsi_val:.1f}" if rsi_val is not None else "—",
                      "Overbought" if rsi_val is not None and rsi_val > 70 else ("Oversold" if rsi_val is not None and rsi_val < 30 else "Normal"))
            adx_val = _safe_float(last, "ADX_14")
            s6.metric("ADX (14)", f"{adx_val:.1f}" if adx_val is not None else "—",
                      "Strong trend" if adx_val is not None and adx_val > 25 else "Weak trend")
            atr_val = _safe_float(last, "ATRr_14")
            s7.metric("ATR (14)", f"₹{atr_val:.2f}" if atr_val is not None else "—")

            st.plotly_chart(candlestick_chart(chart_df, chart_symbol.upper()), use_container_width=True)

            # Indicator summary table
            with st.expander("Indicator Summary (Latest Bar)"):
                ta_agent = TechnicalAnalysisAgent()
                summary = ta_agent.get_summary(chart_df)
                sum_df = pd.DataFrame(
                    [{"Indicator": k, "Value": str(v)} for k, v in summary.items()]
                )
                st.dataframe(sum_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No data for {chart_symbol.upper()}. Check the symbol and try again.")


# ── Tab 4: Live Signals ────────────────────────────────────────────────────

with tabs[3]:
    st.markdown("#### Signal Log")
    portfolio = PortfolioAgent()

    @st.cache_data(ttl=300)
    def _fetch_blackout_symbols():
        try:
            from agents.earnings_calendar_agent import EarningsCalendarAgent as _ECA
            from agents.data_agent import DataAgent as _DA
            syms = _DA().get_watchlist("nifty50") + _DA().get_watchlist("nifty200")
            return _ECA().run(list(set(syms)))
        except Exception:
            return []

    _blackout_syms = _fetch_blackout_symbols()

    try:
        from sqlalchemy import text as sqlt
        with portfolio._engine.connect() as conn:
            rows = conn.execute(sqlt(
                "SELECT * FROM signals_log ORDER BY timestamp DESC LIMIT 100"
            )).fetchall()
        signals_df = pd.DataFrame([dict(r._mapping) for r in rows]) if rows else pd.DataFrame()
    except Exception:
        signals_df = pd.DataFrame()

    if not signals_df.empty:
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            verdict_filter = st.multiselect("Verdict", ["APPROVE", "REJECT", "SKIP"],
                                            default=["APPROVE", "REJECT", "SKIP"])
        with col_f2:
            strat_options = ["All"] + list(signals_df["strategy"].dropna().unique())
            strat_filter = st.selectbox("Strategy", strat_options)
        with col_f3:
            min_conf = st.slider("Min Confidence", 0.0, 10.0, 0.0, 0.5)

        filtered = signals_df.copy()
        if verdict_filter:
            filtered = filtered[filtered["claude_verdict"].isin(verdict_filter)]
        if strat_filter != "All":
            filtered = filtered[filtered["strategy"] == strat_filter]
        filtered = filtered[filtered["confidence"].fillna(0) >= min_conf]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", len(filtered))
        m2.metric("Approved", len(filtered[filtered["claude_verdict"] == "APPROVE"]))
        m3.metric("Rejected", len(filtered[filtered["claude_verdict"] == "REJECT"]))
        avg_conf = filtered['confidence'].mean() if not filtered.empty else None
        m4.metric("Avg Confidence", f"{avg_conf:.1f}/10" if avg_conf is not None and pd.notna(avg_conf) else "—")

        st.divider()

        for _, row in filtered.head(20).iterrows():
            sig_dict = row.to_dict()
            sig_dict["_blackout"] = sig_dict.get("symbol", "") in _blackout_syms
            render_signal_card(sig_dict)
    else:
        st.info("No signals yet. Click 'Run Signal Scan' in the sidebar.")


# ── Tab 5: Backtesting ─────────────────────────────────────────────────────

with tabs[4]:
    st.markdown("#### Strategy Backtesting")

    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
    with col1:
        bt_symbol = st.text_input("Symbol", "RELIANCE", key="bt_sym", label_visibility="collapsed")
    with col2:
        bt_strategy = st.selectbox("Strategy", ["swing", "momentum", "mean_reversion", "positional"], key="bt_strat", label_visibility="collapsed")
    with col3:
        bt_days = st.selectbox("Period", [365, 730, 1095, 1825],
                               format_func=lambda x: f"{x//365}yr{'s' if x//365>1 else ''}", key="bt_days", label_visibility="collapsed")
    with col4:
        run_bt = st.button("Run Backtest", use_container_width=True)

    if run_bt:
        with st.spinner(f"Backtesting {bt_strategy} on {bt_symbol.upper()} ({bt_days}d)..."):
            try:
                from backtesting.engine import BacktestEngine
                engine = BacktestEngine(initial_capital=float(os.getenv("ACCOUNT_SIZE", "100000")))
                results = engine.run([bt_symbol.upper()], strategy=bt_strategy, days=bt_days)

                if results:
                    r = results[0]
                    s = r.summary()
                    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                    bc1.metric("Total Return", f"{s['total_return_pct']}%")
                    bc2.metric("Sharpe Ratio", s["sharpe"],
                               "Excellent" if s["sharpe"] > 1.5 else ("Good" if s["sharpe"] > 1 else "Poor"))
                    bc3.metric("Max Drawdown", f"{s['max_drawdown_pct']}%")
                    bc4.metric("Win Rate", f"{s['win_rate']}%")
                    bc5.metric("Total Trades", s["total_trades"])

                    equity = r.equity_curve()
                    peak = equity.cummax().replace(0, np.nan)
                    dd = (equity - peak) / peak * 100

                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.04, row_heights=[0.65, 0.35],
                                        subplot_titles=("Equity Curve (₹)", "Drawdown (%)"))
                    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity",
                                            line=dict(color="#3fb950", width=2),
                                            fill="tozeroy", fillcolor="rgba(63,185,80,0.08)"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                            line=dict(color="#f85149", width=1.5),
                                            fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
                                            showlegend=False), row=2, col=1)
                    fig.update_layout(**_PLOTLY_THEME, height=420)
                    fig.update_xaxes(gridcolor="#21262d")
                    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Walk-Forward Validation"):
                        wf = engine.walk_forward(bt_symbol.upper(), bt_strategy, total_days=bt_days)
                        if wf:
                            wf_df = pd.DataFrame(wf)
                            wf_df["color"] = wf_df["return_pct"].apply(lambda x: "#3fb950" if x >= 0 else "#f85149")
                            fig_wf = go.Figure(go.Bar(
                                x=wf_df["period_start"], y=wf_df["return_pct"],
                                marker_color=wf_df["color"],
                                text=[f"{v:+.1f}%" for v in wf_df["return_pct"]],
                                textposition="outside",
                            ))
                            fig_wf.update_layout(**_PLOTLY_THEME, height=260,
                                                 title="Walk-Forward Period Returns",
                                                 yaxis=dict(title="Return %", gridcolor="#21262d"),
                                                 xaxis=dict(gridcolor="#21262d"))
                            st.plotly_chart(fig_wf, use_container_width=True)
                        else:
                            st.info("Not enough data for walk-forward.")
                else:
                    st.warning("No signals generated. Try a different symbol or period.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    st.divider()
    st.markdown("#### Strategy Comparison")
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        cmp_symbol = st.text_input("Symbol for comparison", "TCS", key="cmp_sym", label_visibility="collapsed")
    with cc2:
        run_cmp = st.button("Compare All", use_container_width=True)

    if run_cmp:
        with st.spinner("Running all strategies..."):
            try:
                from backtesting.engine import BacktestEngine
                engine = BacktestEngine()
                comp = engine.compare_strategies(cmp_symbol.upper(), days=365)
                if not comp.empty:
                    fig_cmp = go.Figure()
                    metrics = ["total_return_pct", "sharpe", "win_rate"]
                    colors_list = ["#58a6ff", "#3fb950", "#e3b341"]
                    for i, metric in enumerate(metrics):
                        if metric in comp.columns:
                            fig_cmp.add_trace(go.Bar(
                                name=metric.replace("_", " ").title(),
                                x=comp["strategy"], y=comp[metric],
                                marker_color=colors_list[i], opacity=0.85,
                            ))
                    fig_cmp.update_layout(**_PLOTLY_THEME, height=300, barmode="group",
                                         yaxis=dict(gridcolor="#21262d"),
                                         xaxis=dict(gridcolor="#21262d"),
                                         legend=dict(font=dict(size=10)))
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    st.dataframe(comp, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Comparison failed: {e}")


# ── Tab 6: Watchlist ───────────────────────────────────────────────────────

with tabs[5]:
    st.markdown("#### Watchlist Analytics")
    cfg_path = Path(__file__).parent.parent / "config" / "watchlist.yaml"
    try:
        with open(cfg_path) as f:
            watchlist = yaml.safe_load(f) or {}
    except Exception as e:
        st.error(f"Could not load watchlist config: {e}")
        st.stop()

    wl_group = st.radio("Group", ["nifty50", "midcap_focus"], horizontal=True)
    symbols_list = watchlist.get(wl_group, [])

    col_load, col_n = st.columns([1, 3])
    with col_load:
        load_wl = st.button("Load Indicators", use_container_width=True)
    with col_n:
        max_n = max(10, min(40, len(symbols_list)))
        top_n = st.slider("Stocks to load", 1, max_n, min(20, max_n))

    if load_wl:
        with st.spinner("Fetching data..."):
            da = DataAgent()
            ta = TechnicalAnalysisAgent()
            raw = da.run(symbols_list[:top_n], timeframe="1d", days=220)
            enriched = ta.run(raw)

            snapshot = []
            for sym, df in enriched.items():
                s = ta.get_summary(df)
                s["symbol"] = sym
                row = df.iloc[-1]
                s["price"] = round(float(row["close"]), 2)
                atr_v = row.get("ATRr_14")
                close_v = float(row["close"]) if pd.notna(row["close"]) else 0
                s["atr_pct"] = round(float(atr_v) / close_v * 100, 2) if (atr_v is not None and pd.notna(atr_v) and close_v != 0) else 0
                snapshot.append(s)

        if snapshot:
            # RSI heatmap
            st.plotly_chart(rsi_heatmap(snapshot), use_container_width=True)

            # Full indicator table
            st.markdown("#### Indicator Snapshot")
            snap_df = pd.DataFrame(snapshot)
            cols_order = ["symbol", "price", "rsi", "macd", "ema_9", "ema_21", "ema_50",
                          "ema_200", "adx", "atr", "volume_ratio", "above_ema200", "ema_cross_bull"]
            cols_exist = [c for c in cols_order if c in snap_df.columns]
            snap_display = snap_df[cols_exist].copy()
            snap_display.columns = [c.replace("_", " ").title() for c in snap_display.columns]
            st.dataframe(snap_display, use_container_width=True, hide_index=True)

            # Momentum ranking
            st.markdown("#### Momentum Ranking (by RSI)")
            snap_df["rsi_num"] = pd.to_numeric(snap_df["rsi"], errors="coerce")
            momentum = snap_df[["symbol", "price", "rsi_num", "adx", "volume_ratio", "above_ema200"]].dropna(subset=["rsi_num"])
            momentum = momentum.sort_values("rsi_num", ascending=False)
            momentum.columns = ["Symbol", "Price ₹", "RSI", "ADX", "Vol Ratio", "Above 200 EMA"]
            st.dataframe(momentum, use_container_width=True, hide_index=True)


# ── Tab 7: Settings ────────────────────────────────────────────────────────

with tabs[6]:
    st.markdown("#### System Configuration")

    c1, c2, c3, c4 = st.columns(4)
    trading_on = os.getenv("TRADING_ENABLED", "true").lower() != "false"
    paper = os.getenv("PAPER_TRADING", "true").lower() != "false"
    account = float(os.getenv("ACCOUNT_SIZE", "100000"))
    c1.metric("Trading", "Enabled" if trading_on else "DISABLED")
    c2.metric("Mode", "Paper" if paper else "🔴 LIVE")
    c3.metric("Account Size", f"₹{account:,.0f}")
    _ai_provider = "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else ("OpenAI" if os.getenv("OPENAI_API_KEY") else "None")
    c4.metric("AI Validator", _ai_provider)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### API Keys")
        for key, label in [("ANTHROPIC_API_KEY", "Anthropic API"),
                           ("OPENAI_API_KEY", "OpenAI API"),
                           ("TELEGRAM_BOT_TOKEN", "Telegram Bot"),
                           ("TELEGRAM_CHAT_ID", "Telegram Chat ID")]:
            val = os.getenv(key, "")
            status = "✅ Configured" if val else "❌ Not configured"
            st.markdown(f"<div class='info-box'><span style='color:#8b949e; font-size:0.75rem;'>{label}</span><br>{status}</div>",
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Test Notifications")
    if st.button("📨 Send Test Telegram Alert"):
        from agents.notification_agent import NotificationAgent
        try:
            notif = NotificationAgent()
            ok = notif.send_alert("✅ Trading system test alert — Telegram is working!")
            if ok:
                st.success("Test alert sent! Check your Telegram.")
            else:
                st.warning("Not sent — check that TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env, then restart the dashboard.")
        except Exception as e:
            st.error(f"Error: {e}")

    with col_b:
        st.markdown("#### Risk Config")
        risk_path = Path(__file__).parent.parent / "config" / "risk_config.yaml"
        try:
            with open(risk_path) as f:
                risk_cfg = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Could not load risk config: {e}")
            risk_cfg = {}
        pos = risk_cfg.get("position_sizing", {})
        lim = risk_cfg.get("portfolio_limits", {})
        dd = risk_cfg.get("daily_limits", {})
        for label, val in [
            ("Risk per trade", f"{pos.get('risk_per_trade_pct', 2)}%"),
            ("Max positions", lim.get("max_open_positions", 5)),
            ("Max daily loss", f"{dd.get('max_daily_loss_pct', 5)}%"),
            ("Max drawdown", f"{risk_cfg.get('drawdown', {}).get('max_drawdown_pct', 15)}%"),
        ]:
            st.markdown(f"<div class='info-box' style='display:flex; justify-content:space-between;'>"
                        f"<span style='color:#8b949e; font-size:0.78rem;'>{label}</span>"
                        f"<span style='font-weight:600; font-size:0.85rem; font-family:monospace;'>{val}</span>"
                        f"</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Quick Commands")
    st.code("""
# Start everything (dashboard + scheduler) — recommended
python main.py

# Scheduler only, no UI (server / background use)
python main.py --headless

# Dashboard only, no scheduler
python main.py --dashboard

# One-shot signal scan and exit
python main.py --scan

# Backtest a symbol
python main.py --backtest RELIANCE --strategy swing --days 365

# View portfolio in terminal
python main.py --portfolio
""", language="bash")

# ── Tab 8: System Health ───────────────────────────────────────────────────

with tabs[7]:
    st.markdown("#### System Health & Auto-Improvement")
    st.markdown(
        "<div class='info-box' style='margin-bottom:1rem;'>The system automatically tracks per-strategy "
        "performance, injects that context into AI validation, and adjusts confidence thresholds to "
        "improve signal quality over time. This tab shows what the system is learning and doing.</div>",
        unsafe_allow_html=True,
    )

    try:
        from agents.performance_tracker import PerformanceTracker
        from agents.portfolio_agent import PortfolioAgent as _PA
        import json as _json
        from pathlib import Path as _Path

        _portfolio = _PA()
        _df = _portfolio.get_trade_history(limit=500)
        _tracker = PerformanceTracker()
        _strat_stats = _tracker.compute_strategy_stats(_df)
        _sym_stats = _tracker.compute_symbol_stats(_df)
        _tuning = _tracker.get_tuning_state()

        _cfg_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
        with open(_cfg_path) as _f:
            _cfg_yaml = yaml.safe_load(_f)
        _default_threshold = float(_cfg_yaml["signals"]["min_confidence_for_claude"])

        # ── Section 1: Strategy Performance ───────────────────────────────
        st.markdown("#### Strategy Performance")
        if _strat_stats:
            _strat_cols = st.columns(len(_strat_stats))
            for _i, (_strat, _s) in enumerate(sorted(_strat_stats.items())):
                _wr = _s["win_rate"]
                _wr_color = "#3fb950" if _wr >= 55 else ("#e3b341" if _wr >= 40 else "#f85149")
                _strat_cols[_i].markdown(
                    f"<div class='info-box' style='text-align:center;'>"
                    f"<div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em;'>"
                    f"{_strat.title()}</div>"
                    f"<div style='font-size:1.8rem; font-weight:700; color:{_wr_color};'>{_wr}%</div>"
                    f"<div style='font-size:0.75rem; color:#8b949e;'>Win Rate</div>"
                    f"<div style='font-size:0.85rem; margin-top:0.4rem;'>{_s['total_trades']} trades "
                    f"· avg {_s['avg_r']}R</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No closed trades yet — strategy performance will appear after trades complete.")

        # ── Section 2: Auto-Tuning Status ─────────────────────────────────
        st.divider()
        st.markdown("#### Auto-Tuning Status")
        _overrides = _tuning.get("threshold_overrides", {})
        _last_updated = _tuning.get("last_updated", "Never")

        if _overrides or _strat_stats:
            _all_strategies = set(list(_overrides.keys()) + list(_strat_stats.keys()))
            _all_strategies.update(["swing", "intraday", "positional"])
            _rows_html = ""
            for _strat in sorted(_all_strategies):
                _current = _overrides.get(_strat, _default_threshold)
                _diff = round(_current - _default_threshold, 1)
                _diff_str = f"+{_diff}" if _diff > 0 else str(_diff)
                _diff_color = "#f85149" if _diff > 0 else ("#3fb950" if _diff < 0 else "#8b949e")
                _arrow = "↑ more selective" if _diff > 0 else ("↓ more permissive" if _diff < 0 else "— no change")
                _wr_display = f"{_strat_stats[_strat]['win_rate']}% WR, {_strat_stats[_strat]['total_trades']} trades" if _strat in _strat_stats else "Insufficient data"
                _rows_html += (
                    f"<tr>"
                    f"<td style='padding:0.5rem 0.8rem; font-weight:600;'>{_strat.title()}</td>"
                    f"<td style='padding:0.5rem 0.8rem; color:#8b949e;'>{_default_threshold}</td>"
                    f"<td style='padding:0.5rem 0.8rem; font-weight:700; font-family:monospace;'>{_current}</td>"
                    f"<td style='padding:0.5rem 0.8rem; color:{_diff_color}; font-family:monospace;'>{_diff_str} {_arrow}</td>"
                    f"<td style='padding:0.5rem 0.8rem; color:#8b949e; font-size:0.8rem;'>{_wr_display}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<table style='width:100%; border-collapse:collapse; font-size:0.88rem;'>"
                f"<thead><tr style='border-bottom:1px solid #30363d;'>"
                f"<th style='padding:0.4rem 0.8rem; text-align:left; color:#8b949e;'>Strategy</th>"
                f"<th style='padding:0.4rem 0.8rem; text-align:left; color:#8b949e;'>Default</th>"
                f"<th style='padding:0.4rem 0.8rem; text-align:left; color:#8b949e;'>Current</th>"
                f"<th style='padding:0.4rem 0.8rem; text-align:left; color:#8b949e;'>Adjustment</th>"
                f"<th style='padding:0.4rem 0.8rem; text-align:left; color:#8b949e;'>Reason</th>"
                f"</tr></thead><tbody>{_rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(f"Last updated: {_last_updated}  ·  Thresholds adjust by ±0.5 when win rate exceeds bounds (min 5.0, max 8.5, requires 10+ trades)")
        else:
            st.info("Auto-tuning will activate after 10+ closed trades per strategy.")

        # ── Section 3: AI Context Preview ─────────────────────────────────
        st.divider()
        st.markdown("#### AI Prompt Context")
        _ai_ctx = _tuning.get("ai_context", "") or _tracker.build_ai_context(_strat_stats, _sym_stats)
        if _ai_ctx:
            st.caption("This performance summary is injected into every GPT-4o/Claude validation call:")
            st.code(_ai_ctx, language=None)
        else:
            st.info("Performance context will appear here once enough trade history exists (5+ trades per strategy).")

        # ── Section 4: Symbol Leaderboard ─────────────────────────────────
        if _sym_stats:
            st.divider()
            st.markdown("#### Symbol Leaderboard")
            _sorted_syms = sorted(_sym_stats.items(), key=lambda x: -x[1]["win_rate"])
            _col_best, _col_worst = st.columns(2)
            with _col_best:
                st.markdown("**Top Performers**")
                for _sym, _s in _sorted_syms[:5]:
                    _bar = "█" * int(_s["win_rate"] / 10)
                    st.markdown(
                        f"<div class='info-box' style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span style='font-weight:600;'>{_sym}</span>"
                        f"<span style='color:#3fb950; font-family:monospace;'>{_s['win_rate']}% "
                        f"({_s['wins']}/{_s['total_trades']})</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            with _col_worst:
                st.markdown("**Under-Performers**")
                for _sym, _s in reversed(_sorted_syms[-5:]):
                    st.markdown(
                        f"<div class='info-box' style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span style='font-weight:600;'>{_sym}</span>"
                        f"<span style='color:#f85149; font-family:monospace;'>{_s['win_rate']}% "
                        f"({_s['wins']}/{_s['total_trades']})</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Section 5: Improvement History ────────────────────────────────
        _history = _tuning.get("history", [])
        if _history:
            st.divider()
            st.markdown("#### Auto-Tuning History")
            for _entry in reversed(_history[-20:]):
                _color = "#f85149" if "Raised" in _entry.get("action", "") else "#3fb950"
                st.markdown(
                    f"<div class='info-box' style='border-left:3px solid {_color}; padding-left:0.8rem;'>"
                    f"<span style='font-size:0.75rem; color:#8b949e;'>{_entry.get('date','')}</span><br>"
                    f"<span style='font-weight:600;'>{_entry.get('action','')}</span><br>"
                    f"<span style='font-size:0.8rem; color:#8b949e;'>{_entry.get('reason','')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif not _history and _strat_stats:
            st.divider()
            st.markdown("#### Auto-Tuning History")
            st.info("No threshold adjustments yet — all strategies are within healthy win-rate range or have insufficient data.")

    except Exception as _e:
        st.error(f"Could not load system health data: {_e}")
