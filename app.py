"""
app.py — Stock AI Advisor
Bloomberg-style AI-powered stock recommendation engine.
Supports IDX (Indonesian) and global equities via yfinance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader   import load_stock_data, get_stock_info, get_multiple_stocks, format_number
from utils.indicators    import compute_all_indicators, get_current_signals, get_trend_direction
from utils.scoring       import compute_score, get_score_color, get_recommendation_color
from utils.recommendation import generate_recommendation
from utils.forecasting   import forecast_price
from utils.charts        import (
    create_candlestick_chart,
    create_rsi_chart,
    create_macd_chart,
    create_forecast_chart,
    create_score_gauge,
    create_multi_stock_chart,
    create_sub_scores_bar,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock AI Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0E1117 0%, #1A1A2E 50%, #0E1117 100%);
    border: 1px solid #00FFAA33;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #00FFAA, #00AAFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.header-subtitle {
    color: #8888AA;
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #1E1E2E;
    border: 1px solid #2A2A3E;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00FFAA55; }
.metric-label {
    color: #8888AA;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #FAFAFA;
}
.metric-delta {
    font-size: 12px;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Recommendation badge */
.rec-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2px;
    font-family: 'IBM Plex Mono', monospace;
    border: 2px solid currentColor;
    text-transform: uppercase;
}

/* Signal pill */
.signal-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
    font-weight: 500;
}
.bull-pill { background: #00FF8833; color: #00FF88; border: 1px solid #00FF8866; }
.bear-pill { background: #FF335533; color: #FF3355; border: 1px solid #FF335566; }
.neutral-pill { background: #FFDD0033; color: #FFDD00; border: 1px solid #FFDD0066; }

/* Section header */
.section-header {
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: #00FFAA;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #2A2A3E;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Summary box */
.summary-box {
    background: #1A1A2E;
    border-left: 3px solid #00FFAA;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #CCCCDD;
    font-style: italic;
}

/* Trade table */
.trade-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #2A2A3E;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.trade-label { color: #8888AA; }
.trade-value { color: #FAFAFA; font-weight: 500; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A0A1A !important;
    border-right: 1px solid #1E1E2E;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background: #1E1E2E;
    border-radius: 8px;
    padding: 4px;
}
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* Plotly config button */
.js-plotly-plot .plotly .modebar { background: #1E1E2E !important; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Stock AI Advisor")
    st.markdown("---")
    
    # Stock selection
    st.markdown("**Stock Symbol**")
    popular_stocks = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK",
                      "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
    
    selected_preset = st.selectbox("Quick select", ["Custom"] + popular_stocks)
    
    if selected_preset == "Custom":
        ticker = st.text_input("Enter ticker", value="BBCA.JK",
                               placeholder="e.g. BBCA.JK / AAPL").upper().strip()
    else:
        ticker = selected_preset
    
    st.markdown("---")
    
    # Timeframe & Period
    st.markdown("**Analysis Settings**")
    
    period_map = {
        "1 Month":  "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year":   "1y",
        "2 Years":  "2y",
        "5 Years":  "5y",
    }
    period_label = st.selectbox("Data Period", list(period_map.keys()), index=3)
    period = period_map[period_label]
    
    interval_map = {
        "Daily":   "1d",
        "Weekly":  "1wk",
        "Monthly": "1mo",
    }
    interval_label = st.selectbox("Interval", list(interval_map.keys()), index=0)
    interval = interval_map[interval_label]
    
    risk_profile = st.select_slider(
        "Risk Profile",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate",
    )
    
    st.markdown("---")
    
    # Portfolio settings
    st.markdown("**Portfolio Mode**")
    enable_portfolio = st.checkbox("Enable Position Sizing", value=False)
    
    if enable_portfolio:
        capital = st.number_input("Capital (IDR)", min_value=1_000_000,
                                  value=10_000_000, step=1_000_000,
                                  format="%d")
        risk_pct = st.slider("Max Risk per Trade (%)", min_value=0.5, max_value=5.0,
                             value=2.0, step=0.5)
    else:
        capital = None
        risk_pct = 2.0
    
    st.markdown("---")
    
    # Forecast settings
    st.markdown("**AI Forecast**")
    enable_forecast = st.checkbox("Enable Price Forecast", value=True)
    if enable_forecast:
        forecast_days = st.select_slider("Forecast Days", options=[7, 14, 30, 60], value=30)
    
    # Multi-stock comparison
    st.markdown("---")
    st.markdown("**Compare Stocks**")
    enable_compare = st.checkbox("Multi-Stock Comparison", value=False)
    if enable_compare:
        compare_input = st.text_area(
            "Tickers (one per line)",
            value="BBCA.JK\nBBRI.JK\nTLKM.JK",
            height=100,
        )
        compare_tickers = [t.strip().upper() for t in compare_input.split("\n") if t.strip()]
    
    st.markdown("---")
    st.caption("Data from Yahoo Finance · Not financial advice")


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <div>
        <div class="header-title">📈 STOCK AI ADVISOR</div>
        <div class="header-subtitle">AI-Powered Technical & Fundamental Analysis Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Load data
if not ticker:
    st.warning("Please enter a ticker symbol.")
    st.stop()

with st.spinner(f"Loading {ticker} data..."):
    df_raw = load_stock_data(ticker, period=period, interval=interval)

if df_raw.empty:
    st.error(f"❌ Could not load data for **{ticker}**. Please check the ticker symbol and try again.")
    st.info("💡 For Indonesian stocks, append `.JK` (e.g., `BBCA.JK`). For US stocks, use the plain symbol (e.g., `AAPL`).")
    st.stop()

# Compute indicators
df = compute_all_indicators(df_raw)
signals = get_current_signals(df)
score = compute_score(signals, risk_profile)
rec = generate_recommendation(
    df, signals, score, risk_profile,
    capital=capital if enable_portfolio else None,
    risk_pct=risk_pct if enable_portfolio else 2.0,
)
info = get_stock_info(ticker)
trend = get_trend_direction(signals)


# ── TOP KPI STRIP ─────────────────────────────────────────────────────────────

close_price  = signals.get("close", 0)
prev_close   = df["Close"].iloc[-2] if len(df) > 1 else close_price
price_change = close_price - prev_close
price_chg_pct = (price_change / prev_close * 100) if prev_close else 0
is_positive   = price_change >= 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Price</div>
        <div class="metric-value" style="color:{'#00FF88' if is_positive else '#FF3355'}">
            {close_price:,.0f}
        </div>
        <div class="metric-delta" style="color:{'#00FF88' if is_positive else '#FF3355'}">
            {'▲' if is_positive else '▼'} {abs(price_chg_pct):.2f}%
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    rec_color = get_recommendation_color(score["category"])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recommendation</div>
        <div class="metric-value" style="font-size:16px; color:{rec_color}">
            {score["category"].upper()}
        </div>
        <div class="metric-delta" style="color:{rec_color}">AI Score: {score["total"]:.0f}/100</div>
    </div>""", unsafe_allow_html=True)

with col3:
    rsi = signals.get("rsi", 0)
    rsi_color = "#FF3355" if rsi > 70 else ("#00FF88" if rsi < 30 else "#FFDD00")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI (14)</div>
        <div class="metric-value" style="color:{rsi_color}">{rsi:.1f}</div>
        <div class="metric-delta" style="color:{rsi_color}">
            {"Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")}
        </div>
    </div>""", unsafe_allow_html=True)

with col4:
    vol_ratio = signals.get("volume_ratio", 1)
    vol_color = "#00FF88" if vol_ratio > 1.3 else ("#FF3355" if vol_ratio < 0.7 else "#FFDD00")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Volume Ratio</div>
        <div class="metric-value" style="color:{vol_color}">{vol_ratio:.2f}x</div>
        <div class="metric-delta" style="color:{vol_color}">
            {'High Vol' if vol_ratio > 1.3 else ('Low Vol' if vol_ratio < 0.7 else 'Normal Vol')}
        </div>
    </div>""", unsafe_allow_html=True)

with col5:
    trend_color = "#00FF88" if trend == "UPTREND" else ("#FF3355" if trend == "DOWNTREND" else "#FFDD00")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Trend</div>
        <div class="metric-value" style="color:{trend_color}; font-size:16px">{trend}</div>
        <div class="metric-delta" style="color:#8888AA">
            {info.get("sector", "N/A")}
        </div>
    </div>""", unsafe_allow_html=True)

with col6:
    bb_pct = signals.get("bb_pct", 0.5)
    bb_color = "#FF3355" if bb_pct > 0.85 else ("#00FF88" if bb_pct < 0.15 else "#FFDD00")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BB Position</div>
        <div class="metric-value" style="color:{bb_color}">{bb_pct*100:.0f}%</div>
        <div class="metric-delta" style="color:{bb_color}">
            {'Near Upper' if bb_pct > 0.8 else ('Near Lower' if bb_pct < 0.2 else 'Mid Band')}
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Charts",
    "🤖 AI Analysis",
    "💼 Trade Setup",
    "📈 Forecast",
    "🏦 Fundamentals",
    "🔀 Compare",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    show_ma = st.toggle("Show Moving Averages", value=True)
    
    st.markdown(f"<div class='section-header'>{ticker} — Price Chart · {period_label}</div>",
                unsafe_allow_html=True)
    
    fig_main = create_candlestick_chart(df, signals, show_ma=show_ma)
    st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": True})
    
    col_rsi, col_macd = st.columns(2)
    
    with col_rsi:
        st.markdown("<div class='section-header'>RSI (14)</div>", unsafe_allow_html=True)
        st.plotly_chart(create_rsi_chart(df), use_container_width=True,
                        config={"displayModeBar": False})
    
    with col_macd:
        st.markdown("<div class='section-header'>MACD (12/26/9)</div>", unsafe_allow_html=True)
        st.plotly_chart(create_macd_chart(df), use_container_width=True,
                        config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    col_gauge, col_score = st.columns([1, 2])
    
    with col_gauge:
        st.markdown("<div class='section-header'>AI Score</div>", unsafe_allow_html=True)
        st.plotly_chart(create_score_gauge(score["total"]), use_container_width=True,
                        config={"displayModeBar": False})
        
        rec_color = get_recommendation_color(score["category"])
        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px">
            <span class="rec-badge" style="color:{rec_color}; border-color:{rec_color}aa">
                {score["category"]}
            </span>
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <br>
        <div style="text-align:center">
            <div style="font-size:12px; color:#8888AA; font-family:IBM Plex Mono">Bullish Probability</div>
            <div style="font-size:26px; font-weight:700; color:#00FFAA">{score['bullish_pct']:.0f}%</div>
            <div style="font-size:12px; color:#FF3355; font-family:IBM Plex Mono">
                Bearish: {100 - score['bullish_pct']:.0f}%
            </div>
        </div>""", unsafe_allow_html=True)
    
    with col_score:
        st.markdown("<div class='section-header'>Factor Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(create_sub_scores_bar(score), use_container_width=True,
                        config={"displayModeBar": False})
        
        # Active signals
        st.markdown("<div class='section-header' style='margin-top:16px'>Active Signals</div>",
                    unsafe_allow_html=True)
        
        bull_signals = []
        bear_signals = []
        
        if signals.get("rsi_oversold"):       bull_signals.append("RSI Oversold")
        if signals.get("macd_bullish_cross"):  bull_signals.append("MACD Bullish X")
        if signals.get("golden_cross"):        bull_signals.append("Golden Cross")
        if signals.get("price_above_sma50"):   bull_signals.append("Above SMA50")
        if signals.get("price_above_sma200"):  bull_signals.append("Above SMA200")
        if signals.get("high_volume"):         bull_signals.append("High Volume")
        if signals.get("bb_breakout_up"):      bull_signals.append("BB Breakout Up")
        
        if signals.get("rsi_overbought"):      bear_signals.append("RSI Overbought")
        if signals.get("macd_bearish_cross"):  bear_signals.append("MACD Bearish X")
        if signals.get("death_cross"):         bear_signals.append("Death Cross")
        if not signals.get("price_above_sma50"):  bear_signals.append("Below SMA50")
        if not signals.get("price_above_sma200"): bear_signals.append("Below SMA200")
        if signals.get("bb_breakout_down"):    bear_signals.append("BB Breakdown")
        
        if bull_signals or bear_signals:
            pills_html = ""
            for s in bull_signals:
                pills_html += f'<span class="signal-pill bull-pill">▲ {s}</span>'
            for s in bear_signals:
                pills_html += f'<span class="signal-pill bear-pill">▼ {s}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.markdown('<span class="signal-pill neutral-pill">— No strong signals detected</span>',
                        unsafe_allow_html=True)
    
    # AI summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>AI Analyst Summary</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{rec["summary"]}</div>', unsafe_allow_html=True)
    
    # Indicator table
    with st.expander("📋 Full Indicator Values", expanded=False):
        ind_data = {
            "Indicator": [
                "SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20",
                "RSI", "MACD", "MACD Signal", "Stoch %K", "Stoch %D",
                "BB Upper", "BB Middle", "BB Lower", "BB %", "ATR",
                "VWAP", "Volume Ratio",
            ],
            "Value": [
                f"{signals.get('sma_20', 0):,.2f}",
                f"{signals.get('sma_50', 0):,.2f}",
                f"{signals.get('sma_100', 0):,.2f}",
                f"{signals.get('sma_200', 0):,.2f}",
                f"{signals.get('ema_20', 0):,.2f}",
                f"{signals.get('rsi', 0):.2f}",
                f"{signals.get('macd', 0):.4f}",
                f"{signals.get('macd_signal', 0):.4f}",
                f"{signals.get('stoch_k', 0):.2f}",
                f"{signals.get('stoch_d', 0):.2f}",
                f"{signals.get('bb_upper', 0):,.2f}",
                f"{signals.get('bb_middle', 0):,.2f}",
                f"{signals.get('bb_lower', 0):,.2f}",
                f"{signals.get('bb_pct', 0)*100:.1f}%",
                f"{signals.get('atr', 0):,.2f}",
                f"{signals.get('vwap', 0):,.2f}",
                f"{signals.get('volume_ratio', 0):.2f}x",
            ],
        }
        st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE SETUP
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    col_setup, col_rr = st.columns([3, 2])
    
    with col_setup:
        st.markdown("<div class='section-header'>Trade Setup</div>", unsafe_allow_html=True)
        
        entry = rec["entry_price"]
        sl    = rec["stop_loss"]
        tp1   = rec["take_profit_1"]
        tp2   = rec["take_profit_2"]
        tp3   = rec["take_profit_3"]
        
        is_buy = "Buy" in rec["recommendation"] or rec["recommendation"] == "Hold"
        
        rows = [
            ("Recommendation", rec["recommendation"],
             get_recommendation_color(rec["recommendation"])),
            ("Entry Price",  f"{entry:,.2f}", "#FAFAFA"),
            ("Stop Loss",    f"{sl:,.2f}",   "#FF3355"),
            ("Take Profit 1", f"{tp1:,.2f}", "#00FF88"),
            ("Take Profit 2", f"{tp2:,.2f}", "#00CC66"),
            ("Take Profit 3", f"{tp3:,.2f}", "#009944"),
            ("Risk Amount",   f"{rec['risk_amount']:,.2f}", "#FF9966"),
            ("Reward Amount", f"{rec['reward_amount']:,.2f}", "#66FFAA"),
            ("Risk/Reward",   f"1 : {rec['rr_ratio']:.2f}",
             "#00FFAA" if rec["rr_ratio"] >= 2 else "#FFDD00"),
            ("Confidence",   f"{rec['confidence']:.0f}%",
             get_score_color(rec["confidence"])),
        ]
        
        rows_html = ""
        for label, value, color in rows:
            rows_html += f"""
            <div class="trade-row">
                <span class="trade-label">{label}</span>
                <span class="trade-value" style="color:{color}">{value}</span>
            </div>"""
        
        st.markdown(f"""
        <div style="background:#1E1E2E; border:1px solid #2A2A3E;
                    border-radius:10px; padding:16px 20px">
            {rows_html}
        </div>""", unsafe_allow_html=True)
        
        # Stop loss & target pct
        if entry > 0:
            sl_pct  = abs(entry - sl)  / entry * 100
            tp1_pct = abs(tp1 - entry) / entry * 100
            st.markdown(f"""
            <div style="margin-top:12px; padding:12px; background:#11111F; border-radius:8px;
                        font-family:IBM Plex Mono; font-size:12px; color:#8888AA">
                📉 Stop Loss distance: <span style="color:#FF3355">{sl_pct:.1f}%</span> &nbsp;|&nbsp;
                📈 TP1 distance: <span style="color:#00FF88">{tp1_pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    
    with col_rr:
        st.markdown("<div class='section-header'>Confidence Meter</div>", unsafe_allow_html=True)
        st.plotly_chart(create_score_gauge(rec["confidence"]), use_container_width=True,
                        config={"displayModeBar": False})
        
        # ATR info
        atr = signals.get("atr", 0)
        st.markdown(f"""
        <div style="background:#1E1E2E; border:1px solid #2A2A3E; border-radius:8px;
                    padding:14px 18px; margin-top:8px">
            <div style="font-family:IBM Plex Mono; font-size:11px; color:#8888AA;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:8px">
                Volatility (ATR-14)
            </div>
            <div style="font-size:22px; font-weight:700; color:#FAFAFA">{atr:,.2f}</div>
            <div style="font-size:11px; color:#8888AA; font-family:IBM Plex Mono; margin-top:4px">
                {atr/close_price*100:.2f}% of price
            </div>
        </div>""", unsafe_allow_html=True)
    
    # Portfolio / Position sizing
    if enable_portfolio and rec.get("position"):
        pos = rec["position"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Position Sizing</div>", unsafe_allow_html=True)
        
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.metric("Recommended Lots", f"{pos['lots']:,}")
        with pc2:
            st.metric("Total Shares",     f"{pos['shares']:,}")
        with pc3:
            st.metric("Capital Required", f"Rp {pos['capital_used']:,.0f}")
        with pc4:
            st.metric("Max Risk (IDR)",   f"Rp {pos['max_risk_idr']:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    if not enable_forecast:
        st.info("Enable **AI Forecast** in the sidebar to see price predictions.")
    else:
        with st.spinner("Running AI forecast..."):
            try:
                forecast_df = forecast_price(df, days_ahead=forecast_days)
            except Exception as e:
                forecast_df = pd.DataFrame()
                st.warning(f"Forecast error: {e}")
        
        st.markdown("<div class='section-header'>AI Price Forecast</div>",
                    unsafe_allow_html=True)
        
        if forecast_df is not None and not forecast_df.empty:
            last_price   = df["Close"].iloc[-1]
            forecast_end = forecast_df["Forecast"].iloc[-1]
            forecast_chg = (forecast_end - last_price) / last_price * 100
            
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.metric("Current Price",      f"{last_price:,.2f}")
            with fc2:
                st.metric(f"{forecast_days}d Forecast", f"{forecast_end:,.2f}",
                          delta=f"{forecast_chg:+.2f}%")
            with fc3:
                method = forecast_df["Method"].iloc[0]
                st.metric("Model", method)
            
            st.plotly_chart(create_forecast_chart(df, forecast_df),
                            use_container_width=True, config={"displayModeBar": False})
            
            with st.expander("Forecast Data Table"):
                st.dataframe(forecast_df[["Forecast", "Upper", "Lower"]].round(2),
                             use_container_width=True)
        else:
            st.warning("Insufficient data for forecast. Try a longer data period.")
        
        st.caption(
            "⚠️ Forecasts are based on statistical models and historical patterns. "
            "They are NOT investment advice. Past performance does not guarantee future results."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("<div class='section-header'>Company Overview</div>",
                unsafe_allow_html=True)
    
    fc1, fc2 = st.columns([2, 1])
    
    with fc1:
        st.markdown(f"""
        <div style="background:#1E1E2E; border-radius:10px; padding:18px 22px;
                    border:1px solid #2A2A3E">
            <div style="font-size:20px; font-weight:700; color:#FAFAFA; margin-bottom:4px">
                {info.get("name", ticker)}
            </div>
            <div style="font-size:12px; color:#8888AA; font-family:IBM Plex Mono; margin-bottom:14px">
                {info.get("sector", "N/A")} · {info.get("industry", "N/A")} · 
                {info.get("exchange", "N/A")}
            </div>
            <div style="font-size:13px; color:#CCCCDD; line-height:1.6">
                {info.get("description", "No description available.")[:500]}...
            </div>
        </div>""", unsafe_allow_html=True)
    
    with fc2:
        curr = info.get("currency", "IDR")
        fundamental_rows = [
            ("Market Cap",       format_number(info.get("market_cap", 0), curr)),
            ("P/E Ratio",        f"{info.get('pe_ratio', 0):.2f}x" if info.get("pe_ratio") else "N/A"),
            ("P/B Ratio",        f"{info.get('pb_ratio', 0):.2f}x" if info.get("pb_ratio") else "N/A"),
            ("Dividend Yield",   f"{info.get('dividend_yield', 0)*100:.2f}%" if info.get("dividend_yield") else "N/A"),
            ("Beta",             f"{info.get('beta', 0):.2f}"),
            ("52W High",         f"{info.get('52w_high', 0):,.0f}"),
            ("52W Low",          f"{info.get('52w_low', 0):,.0f}"),
            ("ROE",              f"{info.get('roe', 0)*100:.1f}%" if info.get("roe") else "N/A"),
            ("Profit Margin",    f"{info.get('profit_margins', 0)*100:.1f}%" if info.get("profit_margins") else "N/A"),
            ("Revenue Growth",   f"{info.get('revenue_growth', 0)*100:.1f}%" if info.get("revenue_growth") else "N/A"),
            ("Debt/Equity",      f"{info.get('debt_to_equity', 0):.2f}x" if info.get("debt_to_equity") else "N/A"),
            ("Current Ratio",    f"{info.get('current_ratio', 0):.2f}" if info.get("current_ratio") else "N/A"),
        ]
        
        rows_html = ""
        for label, value in fundamental_rows:
            rows_html += f"""
            <div class="trade-row">
                <span class="trade-label">{label}</span>
                <span class="trade-value">{value}</span>
            </div>"""
        
        st.markdown(f"""
        <div style="background:#1E1E2E; border:1px solid #2A2A3E;
                    border-radius:10px; padding:16px 20px">
            {rows_html}
        </div>""", unsafe_allow_html=True)
    
    # 52-week range bar
    st.markdown("<br>", unsafe_allow_html=True)
    w52_high = info.get("52w_high", 0)
    w52_low  = info.get("52w_low",  0)
    
    if w52_high > 0 and w52_low > 0 and w52_high > w52_low:
        pct_in_range = (close_price - w52_low) / (w52_high - w52_low) * 100
        st.markdown(f"""
        <div style="background:#1E1E2E; border:1px solid #2A2A3E; border-radius:10px;
                    padding:16px 20px">
            <div style="font-family:IBM Plex Mono; font-size:11px; color:#8888AA;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:10px">
                52-Week Range
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;
                        font-family:IBM Plex Mono; font-size:12px">
                <span style="color:#FF3355">{w52_low:,.0f}</span>
                <span style="color:#FAFAFA">Current: {close_price:,.0f}</span>
                <span style="color:#00FF88">{w52_high:,.0f}</span>
            </div>
            <div style="background:#2A2A3E; border-radius:4px; height:8px; position:relative">
                <div style="background:linear-gradient(90deg,#FF3355,#FFDD00,#00FF88);
                            width:{pct_in_range:.0f}%; height:100%; border-radius:4px"></div>
            </div>
            <div style="font-family:IBM Plex Mono; font-size:11px; color:#8888AA;
                        margin-top:6px; text-align:right">
                {pct_in_range:.1f}% from 52W Low
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MULTI-STOCK COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    if not enable_compare:
        st.info("Enable **Multi-Stock Comparison** in the sidebar to compare stocks.")
    else:
        with st.spinner("Loading comparison data..."):
            comp_data = get_multiple_stocks(compare_tickers, period=period)
        
        if not comp_data:
            st.error("Could not load any comparison data.")
        else:
            st.markdown("<div class='section-header'>Normalized Return Comparison</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(create_multi_stock_chart(comp_data),
                            use_container_width=True, config={"displayModeBar": False})
            
            # Comparison table
            comp_rows = []
            for tick, d in comp_data.items():
                if d.empty:
                    continue
                with st.spinner(f"Computing {tick}..."):
                    d_ind = compute_all_indicators(d)
                    sig   = get_current_signals(d_ind)
                    sc    = compute_score(sig, risk_profile)
                
                ret_1m  = (d["Close"].iloc[-1] / d["Close"].iloc[-22] - 1) * 100 if len(d) >= 22 else 0
                ret_3m  = (d["Close"].iloc[-1] / d["Close"].iloc[-66] - 1) * 100 if len(d) >= 66 else 0
                
                comp_rows.append({
                    "Ticker":   tick,
                    "Price":    f"{d['Close'].iloc[-1]:,.2f}",
                    "1M Ret %": f"{ret_1m:+.1f}%",
                    "3M Ret %": f"{ret_3m:+.1f}%",
                    "RSI":      f"{sig.get('rsi', 0):.1f}",
                    "AI Score": f"{sc['total']:.0f}",
                    "Signal":   sc["category"],
                })
            
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding:16px; text-align:center;
            border-top:1px solid #2A2A3E; color:#444455;
            font-family:IBM Plex Mono; font-size:11px">
    Stock AI Advisor · Data from Yahoo Finance · For educational purposes only ·
    Not financial advice · Always DYOR
</div>
""", unsafe_allow_html=True)
