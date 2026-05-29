"""
indicators.py
Computes all technical indicators using the 'ta' library and manual calculations.
"""

import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a given OHLCV DataFrame.
    
    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns
    
    Returns:
        DataFrame enriched with all indicator columns
    """
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    
    # ── TREND INDICATORS ──────────────────────────────────────────────────────
    
    # Simple Moving Averages
    df["SMA_20"]  = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"]  = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_100"] = ta.trend.sma_indicator(df["Close"], window=100)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    
    # Exponential Moving Average
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)
    
    # ── MOMENTUM INDICATORS ───────────────────────────────────────────────────
    
    # RSI
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    
    # MACD
    macd_obj = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()
    
    # Stochastic RSI
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    # Williams %R
    df["Williams_R"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=14)
    
    # ── VOLATILITY INDICATORS ─────────────────────────────────────────────────
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()
    df["BB_Pct"]    = bb.bollinger_pband()
    
    # Average True Range
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    
    # ── VOLUME INDICATORS ─────────────────────────────────────────────────────
    
    # On-Balance Volume
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    
    # VWAP (approximate - rolling 20-day)
    df["VWAP"] = _compute_vwap(df)
    
    # Volume SMA
    df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]  # > 1 means above avg volume
    
    # ── SUPPORT / RESISTANCE ──────────────────────────────────────────────────
    
    df["Support"]    = df["Low"].rolling(window=20).min()
    df["Resistance"] = df["High"].rolling(window=20).max()
    
    # ── DERIVED SIGNALS ───────────────────────────────────────────────────────
    
    df = _compute_signals(df)
    
    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute rolling 20-period VWAP."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    return vwap


def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive buy/sell signal flags from computed indicators."""
    
    # RSI signals
    df["RSI_Oversold"]   = df["RSI"] < 30
    df["RSI_Overbought"] = df["RSI"] > 70
    df["RSI_Neutral"]    = (df["RSI"] >= 40) & (df["RSI"] <= 60)
    
    # MACD crossover signals
    df["MACD_Bullish_Cross"] = (
        (df["MACD"] > df["MACD_Signal"]) &
        (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    )
    df["MACD_Bearish_Cross"] = (
        (df["MACD"] < df["MACD_Signal"]) &
        (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
    )
    
    # Moving average trend signals
    df["Price_Above_SMA50"]  = df["Close"] > df["SMA_50"]
    df["Price_Above_SMA200"] = df["Close"] > df["SMA_200"]
    df["Golden_Cross"] = (
        (df["SMA_50"] > df["SMA_200"]) &
        (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    )
    df["Death_Cross"] = (
        (df["SMA_50"] < df["SMA_200"]) &
        (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1))
    )
    
    # Price vs Bollinger Bands
    df["BB_Breakout_Up"]   = df["Close"] > df["BB_Upper"]
    df["BB_Breakout_Down"] = df["Close"] < df["BB_Lower"]
    
    # Volume signals
    df["High_Volume"] = df["Volume_Ratio"] > 1.5  # 50% above average
    
    return df


def get_current_signals(df: pd.DataFrame) -> dict:
    """
    Extract the most recent indicator values and signals.
    
    Args:
        df: DataFrame with computed indicators
    
    Returns:
        Dictionary of current indicator values
    """
    if df.empty:
        return {}
    
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    return {
        # Price
        "close":       last.get("Close", 0),
        "open":        last.get("Open", 0),
        "high":        last.get("High", 0),
        "low":         last.get("Low", 0),
        "volume":      last.get("Volume", 0),
        
        # Trend
        "sma_20":      last.get("SMA_20", np.nan),
        "sma_50":      last.get("SMA_50", np.nan),
        "sma_100":     last.get("SMA_100", np.nan),
        "sma_200":     last.get("SMA_200", np.nan),
        "ema_20":      last.get("EMA_20", np.nan),
        "ema_50":      last.get("EMA_50", np.nan),
        
        # Momentum
        "rsi":         last.get("RSI", 50),
        "macd":        last.get("MACD", 0),
        "macd_signal": last.get("MACD_Signal", 0),
        "macd_hist":   last.get("MACD_Hist", 0),
        "stoch_k":     last.get("Stoch_K", 50),
        "stoch_d":     last.get("Stoch_D", 50),
        "williams_r":  last.get("Williams_R", -50),
        
        # Volatility
        "bb_upper":    last.get("BB_Upper", np.nan),
        "bb_middle":   last.get("BB_Middle", np.nan),
        "bb_lower":    last.get("BB_Lower", np.nan),
        "bb_pct":      last.get("BB_Pct", 0.5),
        "bb_width":    last.get("BB_Width", 0),
        "atr":         last.get("ATR", 0),
        
        # Volume
        "obv":         last.get("OBV", 0),
        "vwap":        last.get("VWAP", np.nan),
        "volume_ratio":last.get("Volume_Ratio", 1),
        "volume_sma":  last.get("Volume_SMA", 0),
        
        # Support / Resistance
        "support":     last.get("Support", np.nan),
        "resistance":  last.get("Resistance", np.nan),
        
        # Signals (boolean)
        "rsi_oversold":        bool(last.get("RSI_Oversold", False)),
        "rsi_overbought":      bool(last.get("RSI_Overbought", False)),
        "macd_bullish_cross":  bool(last.get("MACD_Bullish_Cross", False)),
        "macd_bearish_cross":  bool(last.get("MACD_Bearish_Cross", False)),
        "price_above_sma50":   bool(last.get("Price_Above_SMA50", False)),
        "price_above_sma200":  bool(last.get("Price_Above_SMA200", False)),
        "golden_cross":        bool(last.get("Golden_Cross", False)),
        "death_cross":         bool(last.get("Death_Cross", False)),
        "bb_breakout_up":      bool(last.get("BB_Breakout_Up", False)),
        "bb_breakout_down":    bool(last.get("BB_Breakout_Down", False)),
        "high_volume":         bool(last.get("High_Volume", False)),
        
        # Previous MACD for trend direction
        "prev_macd_hist":      prev.get("MACD_Hist", 0),
    }


def get_trend_direction(signals: dict) -> str:
    """Determine overall trend direction."""
    bullish = 0
    bearish = 0
    
    close = signals.get("close", 0)
    sma_50 = signals.get("sma_50", 0)
    sma_200 = signals.get("sma_200", 0)
    ema_20 = signals.get("ema_20", 0)
    
    if close and sma_50 and close > sma_50:
        bullish += 1
    elif close and sma_50:
        bearish += 1
    
    if close and sma_200 and close > sma_200:
        bullish += 1
    elif close and sma_200:
        bearish += 1
    
    if sma_50 and sma_200 and sma_50 > sma_200:
        bullish += 1
    elif sma_50 and sma_200:
        bearish += 1
    
    if close and ema_20 and close > ema_20:
        bullish += 1
    elif close and ema_20:
        bearish += 1
    
    if bullish > bearish:
        return "UPTREND"
    elif bearish > bullish:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"
