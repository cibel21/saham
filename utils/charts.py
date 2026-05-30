"""
charts.py
All Plotly chart creation functions for the Stock AI Advisor.
Bloomberg terminal-inspired dark UI.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":         "#0E1117",
    "bg_sec":     "#1E1E2E",
    "grid":       "#2A2A3E",
    "text":       "#FAFAFA",
    "text_dim":   "#888899",
    "accent":     "#00FFAA",
    "accent2":    "#00AAFF",
    "accent3":    "#FF6B9D",
    "bull":       "#00FF88",
    "bear":       "#FF3355",
    "neutral":    "#FFDD00",
    "sma_20":     "#FF9F1C",
    "sma_50":     "#2EC4B6",
    "sma_100":    "#CBF3F0",
    "sma_200":    "#FF6B9D",
    "ema_20":     "#FFBF69",
    "bb":         "#7B68EE",
    "volume":     "#4466AA",
    "volume_pos": "#00CC77",
    "volume_neg": "#CC3344",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg_sec"],
    font=dict(family="IBM Plex Mono, monospace", color=COLORS["text"], size=11),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(
        bgcolor="#1A1A2E",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        font=dict(size=10),
    ),
)

AXIS_DEFAULTS = dict(
    showgrid=True,
    gridcolor=COLORS["grid"],
    gridwidth=0.5,
    zeroline=False,
    linecolor=COLORS["grid"],
    tickfont=dict(color=COLORS["text_dim"], size=10),
)


def create_candlestick_chart(df: pd.DataFrame, signals: dict, show_ma: bool = True) -> go.Figure:
    """
    Full main chart: candlestick + moving averages + Bollinger Bands + volume.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )
    
    # ── CANDLESTICK ───────────────────────────────────────────────────────────


    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color=COLORS["bull"],
        decreasing_line_color=COLORS["bear"],
    ), row=1, col=1)
    
    # ── BOLLINGER BANDS ───────────────────────────────────────────────────────
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            name="BB Upper",
            line=dict(color=COLORS["bb"] + "88", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            name="BB Band",
            line=dict(color=COLORS["bb"] + "88", width=1, dash="dot"),
            fill="tonexty",
            fillcolor=COLORS["bb"] + "11",
        ), row=1, col=1)
    
    # ── MOVING AVERAGES ───────────────────────────────────────────────────────
    if show_ma:
        ma_config = [
            ("SMA_20",  COLORS["sma_20"],  "SMA 20",  1.2),
            ("SMA_50",  COLORS["sma_50"],  "SMA 50",  1.5),
            ("SMA_200", COLORS["sma_200"], "SMA 200", 2.0),
            ("EMA_20",  COLORS["ema_20"],  "EMA 20",  1.0),
        ]
        for col, color, name, width in ma_config:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    name=name,
                    line=dict(color=color, width=width),
                    opacity=0.85,
                ), row=1, col=1)
    
    # ── VWAP ──────────────────────────────────────────────────────────────────
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"],
            name="VWAP",
            line=dict(color=COLORS["accent"], width=1.5, dash="dash"),
            opacity=0.7,
        ), row=1, col=1)
    
    # ── VOLUME BARS ───────────────────────────────────────────────────────────
    colors_vol = [COLORS["volume_pos"] if c >= o else COLORS["volume_neg"]
                  for c, o in zip(df["Close"], df["Open"])]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color=colors_vol,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)
    
    # Volume SMA line
    if "Volume_SMA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA"],
            name="Vol SMA",
            line=dict(color=COLORS["accent2"], width=1.2),
            showlegend=False,
        ), row=2, col=1)
    
    # ── LAYOUT ────────────────────────────────────────────────────────────────
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=520,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A1A2E", font_color=COLORS["text"]),
    )
    
    for i in range(1, 3):
        fig.update_xaxes(**AXIS_DEFAULTS, row=i, col=1)
        fig.update_yaxes(**AXIS_DEFAULTS, row=i, col=1)
    
    fig.update_yaxes(title_text="Price", row=1, col=1,
                     title_font=dict(color=COLORS["text_dim"]))
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     title_font=dict(color=COLORS["text_dim"]))
    
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    fig = go.Figure()
    
    if "RSI" not in df.columns:
        return fig
    
    # Overbought zone
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["bear"] + "22",
                  line_width=0, annotation_text="Overbought", annotation_position="right")
    # Oversold zone
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["bull"] + "22",
                  line_width=0, annotation_text="Oversold", annotation_position="right")
    # Neutral zone
    fig.add_hrect(y0=30, y1=70, fillcolor=COLORS["neutral"] + "08", line_width=0)
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI",
        line=dict(color=COLORS["accent2"], width=2),
        fill="tozeroy",
        fillcolor=COLORS["accent2"] + "15",
    ))
    
    # Reference lines
    for level, color in [(70, COLORS["bear"]), (50, COLORS["neutral"]), (30, COLORS["bull"])]:
        fig.add_hline(y=level, line_color=color, line_width=1, line_dash="dot", opacity=0.6)
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=200,
        yaxis=dict(**AXIS_DEFAULTS, range=[0, 100], title_text="RSI"),
        xaxis=dict(**AXIS_DEFAULTS),
        showlegend=False,
    )
    
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD chart with histogram and signal lines."""
    fig = go.Figure()
    
    required = ["MACD", "MACD_Signal", "MACD_Hist"]
    if not all(c in df.columns for c in required):
        return fig
    
    # Histogram
    hist_colors = [COLORS["bull"] if v >= 0 else COLORS["bear"] for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        name="Histogram",
        marker_color=hist_colors,
        opacity=0.7,
    ))
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD",
        line=dict(color=COLORS["accent2"], width=1.5),
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal",
        line=dict(color=COLORS["accent3"], width=1.5),
    ))
    
    fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1)
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=200,
        yaxis=dict(**AXIS_DEFAULTS, title_text="MACD"),
        xaxis=dict(**AXIS_DEFAULTS),
    )
    
    return fig


def create_forecast_chart(df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Forecast chart overlaid on historical data."""
    fig = go.Figure()
    
    # Historical prices (last 90 days for clarity)
    hist = df["Close"].iloc[-90:]
    
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist,
        name="Historical",
        line=dict(color=COLORS["accent"], width=2),
    ))
    
    if forecast_df is not None and not forecast_df.empty:
        # Confidence band
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df["Upper"].tolist() + forecast_df["Lower"].tolist()[::-1],
            fill="toself",
            fillcolor=COLORS["accent2"] + "25",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Forecast"],
            name=f"AI Forecast ({forecast_df['Method'].iloc[0]})",
            line=dict(color=COLORS["accent2"], width=2, dash="dash"),
        ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=350,
        yaxis=dict(**AXIS_DEFAULTS, title_text="Price"),
        xaxis=dict(**AXIS_DEFAULTS),
    )
    
    return fig


def create_score_gauge(score: float) -> go.Figure:
    """Gauge chart for AI score."""
    if score >= 65:
        color = COLORS["bull"]
    elif score >= 45:
        color = COLORS["neutral"]
    else:
        color = COLORS["bear"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "AI Score", "font": {"color": COLORS["text"], "size": 14}},
        number={"font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_dim"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": COLORS["bg_sec"],
            "borderwidth": 1,
            "bordercolor": COLORS["grid"],
            "steps": [
                {"range": [0,  25], "color": COLORS["bear"]    + "33"},
                {"range": [25, 45], "color": COLORS["accent3"] + "33"},
                {"range": [45, 65], "color": COLORS["neutral"] + "33"},
                {"range": [65, 80], "color": COLORS["bull"]    + "33"},
                {"range": [80,100], "color": COLORS["bull"]    + "66"},
            ],
            "threshold": {
                "line": {"color": COLORS["accent"], "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=230,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    
    return fig


def create_multi_stock_chart(stocks_data: dict) -> go.Figure:
    """Normalized comparison chart for multiple stocks."""
    fig = go.Figure()
    
    palette = [COLORS["accent"], COLORS["accent2"], COLORS["accent3"],
               COLORS["sma_20"], COLORS["sma_50"], COLORS["sma_200"]]
    
    for i, (ticker, df) in enumerate(stocks_data.items()):
        if df.empty:
            continue
        normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
        color = palette[i % len(palette)]
        
        fig.add_trace(go.Scatter(
            x=df.index, y=normalized,
            name=ticker,
            line=dict(color=color, width=2),
        ))
    
    fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1, line_dash="dot")
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=350,
        yaxis=dict(**AXIS_DEFAULTS, title_text="Return (%)"),
        xaxis=dict(**AXIS_DEFAULTS),
    )
    
    return fig


def create_sub_scores_bar(score_dict: dict) -> go.Figure:
    """Horizontal bar chart for sub-scores."""
    categories = ["Trend", "Momentum", "Volume", "Volatility", "Sentiment"]
    values = [
        score_dict.get("trend", 50),
        score_dict.get("momentum", 50),
        score_dict.get("volume", 50),
        score_dict.get("volatility", 50),
        score_dict.get("sentiment", 50),
    ]
    
    bar_colors = [
        COLORS["bull"] if v >= 65 else (COLORS["neutral"] if v >= 45 else COLORS["bear"])
        for v in values
    ]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0f}" for v in values],
        textposition="auto",
    ))
    
    fig.add_vline(x=50, line_color=COLORS["grid"], line_width=1, line_dash="dot")
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=210,
        xaxis=dict(**AXIS_DEFAULTS, range=[0, 100], title_text="Score"),
        yaxis=dict(**AXIS_DEFAULTS),
        margin=dict(l=90, r=20, t=20, b=30),
    )
    
    return fig
