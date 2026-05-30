"""
charts.py - Stock AI Advisor
All Plotly chart functions. Each update_layout call is fully explicit
(no **dict unpacking) to avoid key-conflict TypeError in Plotly 5.x+.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── SOLID COLORS ──────────────────────────────────────────────────────────────
BG        = "#0E1117"
BG_SEC    = "#1E1E2E"
BG_LEG    = "#1A1A2E"
GRID      = "#2A2A3E"
TEXT      = "#FAFAFA"
TEXT_DIM  = "#888899"
ACCENT    = "#00FFAA"
ACCENT2   = "#00AAFF"
ACCENT3   = "#FF6B9D"
BULL      = "#00FF88"
BEAR      = "#FF3355"
NEUTRAL   = "#FFDD00"
SMA20     = "#FF9F1C"
SMA50     = "#2EC4B6"
SMA200    = "#FF6B9D"
EMA20     = "#FFBF69"
VOL_POS   = "#00CC77"
VOL_NEG   = "#CC3344"

# ── RGBA COLORS (transparent) ─────────────────────────────────────────────────
BB_LINE       = "rgba(123,104,238,0.55)"
BB_FILL       = "rgba(123,104,238,0.07)"
BULL_FILL     = "rgba(0,255,136,0.55)"
BEAR_FILL     = "rgba(255,51,85,0.45)"
ACCENT2_FILL  = "rgba(0,170,255,0.09)"
FORECAST_BAND = "rgba(0,170,255,0.14)"
OVERBOUGHT    = "rgba(255,51,85,0.13)"
OVERSOLD      = "rgba(0,255,136,0.13)"
NEUTRAL_FILL  = "rgba(255,221,0,0.06)"
HOVER_BG      = "rgba(26,26,46,1)"
STEP1         = "rgba(255,51,85,0.20)"
STEP2         = "rgba(255,107,157,0.20)"
STEP3         = "rgba(255,221,0,0.20)"
STEP4         = "rgba(0,255,136,0.20)"
STEP5         = "rgba(0,255,136,0.40)"
TRANSPARENT   = "rgba(0,0,0,0)"

# ── SHARED AXIS CONFIG ────────────────────────────────────────────────────────
def _axis():
    return dict(
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, linecolor=GRID,
        tickfont=dict(color=TEXT_DIM, size=10),
    )

def _legend():
    return dict(bgcolor=BG_LEG, bordercolor=GRID, borderwidth=1, font=dict(size=10))

def _font():
    return dict(family="IBM Plex Mono, monospace", color=TEXT, size=11)


# ── 1. CANDLESTICK + VOLUME ───────────────────────────────────────────────────
def create_candlestick_chart(df: pd.DataFrame, signals: dict, show_ma: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25],
        shared_xaxes=True, vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color=BULL, decreasing_line_color=BEAR,
        increasing_fillcolor=BULL_FILL, decreasing_fillcolor=BEAR_FILL,
        line=dict(width=1),
    ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns and df["BB_Upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color=BB_LINE, width=1, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Band",
            line=dict(color=BB_LINE, width=1, dash="dot"),
            fill="tonexty", fillcolor=BB_FILL,
        ), row=1, col=1)

    # Moving Averages
    if show_ma:
        for col, color, name, width in [
            ("SMA_20",  SMA20,  "SMA 20",  1.2),
            ("SMA_50",  SMA50,  "SMA 50",  1.5),
            ("SMA_200", SMA200, "SMA 200", 2.0),
            ("EMA_20",  EMA20,  "EMA 20",  1.0),
        ]:
            if col in df.columns and df[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=name,
                    line=dict(color=color, width=width), opacity=0.85,
                ), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns and df["VWAP"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color=ACCENT, width=1.5, dash="dash"), opacity=0.7,
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        VOL_POS if float(c) >= float(o) else VOL_NEG
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7, showlegend=False,
    ), row=2, col=1)

    if "Volume_SMA" in df.columns and df["Volume_SMA"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA"], name="Vol SMA",
            line=dict(color=ACCENT2, width=1.2), showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        legend=_legend(), height=520,
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        hoverlabel=dict(bgcolor=HOVER_BG, font_color=TEXT),
    )
    for i in (1, 2):
        fig.update_xaxes(**_axis(), row=i, col=1)
        fig.update_yaxes(**_axis(), row=i, col=1)
    fig.update_yaxes(title_text="Price",  title_font=dict(color=TEXT_DIM), row=1, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(color=TEXT_DIM), row=2, col=1)
    return fig


# ── 2. RSI ────────────────────────────────────────────────────────────────────
def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI" not in df.columns or df["RSI"].isna().all():
        return fig

    fig.add_hrect(y0=70, y1=100, fillcolor=OVERBOUGHT, line_width=0,
                  annotation_text="Overbought", annotation_position="right")
    fig.add_hrect(y0=0,  y1=30,  fillcolor=OVERSOLD,   line_width=0,
                  annotation_text="Oversold",   annotation_position="right")
    fig.add_hrect(y0=30, y1=70,  fillcolor=NEUTRAL_FILL, line_width=0)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color=ACCENT2, width=2),
        fill="tozeroy", fillcolor=ACCENT2_FILL,
    ))
    for level, color in [(70, BEAR), (50, NEUTRAL), (30, BULL)]:
        fig.add_hline(y=level, line_color=color, line_width=1, line_dash="dot", opacity=0.6)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        height=200, showlegend=False,
        margin=dict(l=60, r=20, t=20, b=40),
        yaxis=dict(**_axis(), range=[0, 100], title_text="RSI"),
        xaxis=dict(**_axis()),
    )
    return fig


# ── 3. MACD ───────────────────────────────────────────────────────────────────
def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"]):
        return fig
    if df["MACD"].isna().all():
        return fig

    hist_colors = [
        BULL if float(v) >= 0 else BEAR
        for v in df["MACD_Hist"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"], name="Histogram",
        marker_color=hist_colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color=ACCENT2, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color=ACCENT3, width=1.5),
    ))
    fig.add_hline(y=0, line_color=GRID, line_width=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        legend=_legend(), height=200,
        margin=dict(l=60, r=20, t=20, b=40),
        yaxis=dict(**_axis(), title_text="MACD"),
        xaxis=dict(**_axis()),
    )
    return fig


# ── 4. FORECAST ───────────────────────────────────────────────────────────────
def create_forecast_chart(df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    hist = df["Close"].iloc[-90:]

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist, name="Historical",
        line=dict(color=ACCENT, width=2),
    ))

    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df["Upper"].tolist() + forecast_df["Lower"].tolist()[::-1],
            fill="toself", fillcolor=FORECAST_BAND,
            line=dict(color=TRANSPARENT), name="Confidence Interval",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Forecast"],
            name=f"AI Forecast ({forecast_df['Method'].iloc[0]})",
            line=dict(color=ACCENT2, width=2, dash="dash"),
        ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        legend=_legend(), height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(**_axis(), title_text="Price"),
        xaxis=dict(**_axis()),
    )
    return fig


# ── 5. SCORE GAUGE ────────────────────────────────────────────────────────────
def create_score_gauge(score: float) -> go.Figure:
    color = BULL if score >= 65 else (NEUTRAL if score >= 45 else BEAR)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "AI Score", "font": {"color": TEXT, "size": 14}},
        number={"font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_DIM},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": BG_SEC, "borderwidth": 1, "bordercolor": GRID,
            "steps": [
                {"range": [0,  25], "color": STEP1},
                {"range": [25, 45], "color": STEP2},
                {"range": [45, 65], "color": STEP3},
                {"range": [65, 80], "color": STEP4},
                {"range": [80,100], "color": STEP5},
            ],
            "threshold": {
                "line": {"color": ACCENT, "width": 3},
                "thickness": 0.8, "value": score,
            },
        },
    ))
    # NOTE: Indicator figures must NOT use plot_bgcolor
    fig.update_layout(
        paper_bgcolor=BG,
        font=_font(),
        height=230,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


# ── 6. MULTI-STOCK COMPARISON ─────────────────────────────────────────────────
def create_multi_stock_chart(stocks_data: dict) -> go.Figure:
    fig = go.Figure()
    palette = [ACCENT, ACCENT2, ACCENT3, SMA20, SMA50, SMA200]

    for i, (ticker, df) in enumerate(stocks_data.items()):
        if df.empty:
            continue
        normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=normalized, name=ticker,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    fig.add_hline(y=0, line_color=GRID, line_width=1, line_dash="dot")
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        legend=_legend(), height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(**_axis(), title_text="Return (%)"),
        xaxis=dict(**_axis()),
    )
    return fig


# ── 7. SUB-SCORES BAR ─────────────────────────────────────────────────────────
def create_sub_scores_bar(score_dict: dict) -> go.Figure:
    categories = ["Trend", "Momentum", "Volume", "Volatility", "Sentiment"]
    values = [
        float(score_dict.get("trend",      50)),
        float(score_dict.get("momentum",   50)),
        float(score_dict.get("volume",     50)),
        float(score_dict.get("volatility", 50)),
        float(score_dict.get("sentiment",  50)),
    ]
    bar_colors = [
        BULL if v >= 65 else (NEUTRAL if v >= 45 else BEAR)
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=categories, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0f}" for v in values],
        textposition="auto",
    ))
    fig.add_vline(x=50, line_color=GRID, line_width=1, line_dash="dot")

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG_SEC, font=_font(),
        showlegend=False, height=210,
        margin=dict(l=90, r=20, t=20, b=30),
        xaxis=dict(**_axis(), range=[0, 100], title_text="Score"),
        yaxis=dict(**_axis()),
    )
    return fig
