"""
recommendation.py
Generates trading recommendations with entry price, stop loss,
take profit, and risk/reward calculations.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_recommendation(
    df: pd.DataFrame,
    signals: dict,
    score: dict,
    risk_profile: str = "Moderate",
    capital: Optional[float] = None,
    risk_pct: float = 2.0,
) -> dict:
    """
    Generate full trading recommendation.
    
    Args:
        df: OHLCV DataFrame with computed indicators
        signals: Current signals dict from indicators.py
        score: Score dict from scoring.py
        risk_profile: 'Conservative', 'Moderate', 'Aggressive'
        capital: Portfolio capital in IDR (optional)
        risk_pct: Max % of capital to risk per trade
    
    Returns:
        Dictionary with full recommendation details
    """
    
    close    = signals.get("close", 0)
    atr      = signals.get("atr", close * 0.02)  # fallback to 2% of price
    category = score.get("category", "Hold")
    
    if close == 0 or atr == 0:
        return _empty_recommendation()
    
    # ── ENTRY PRICE ───────────────────────────────────────────────────────────
    entry_price = _compute_entry_price(signals, category)
    
    # ── STOP LOSS ─────────────────────────────────────────────────────────────
    stop_loss = _compute_stop_loss(signals, atr, risk_profile, category)
    
    # ── TAKE PROFIT ───────────────────────────────────────────────────────────
    tp1, tp2, tp3 = _compute_take_profits(signals, atr, risk_profile, category, entry_price)
    
    # ── RISK / REWARD ─────────────────────────────────────────────────────────
    risk_amount   = abs(entry_price - stop_loss)
    reward_amount = abs(tp1 - entry_price)
    rr_ratio      = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
    
    # ── POSITION SIZING ───────────────────────────────────────────────────────
    position = _compute_position_size(
        capital, risk_pct, entry_price, stop_loss
    ) if capital else None
    
    # ── CONFIDENCE ────────────────────────────────────────────────────────────
    confidence = _compute_confidence(score, signals)
    
    # ── SUMMARY ───────────────────────────────────────────────────────────────
    summary = generate_analyst_summary(signals, score, df)
    
    return {
        "recommendation": category,
        "entry_price":    round(entry_price, 2),
        "stop_loss":      round(stop_loss, 2),
        "take_profit_1":  round(tp1, 2),
        "take_profit_2":  round(tp2, 2),
        "take_profit_3":  round(tp3, 2),
        "rr_ratio":       rr_ratio,
        "confidence":     confidence,
        "risk_amount":    round(risk_amount, 2),
        "reward_amount":  round(reward_amount, 2),
        "position":       position,
        "summary":        summary,
        "bullish_pct":    score.get("bullish_pct", 50),
        "bearish_pct":    round(100 - score.get("bullish_pct", 50), 1),
    }


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

def _compute_entry_price(signals: dict, category: str) -> float:
    """Suggest ideal entry price."""
    close = signals.get("close", 0)
    
    if "Buy" in category:
        # For buy: slightly below current (wait for a dip to support / VWAP)
        vwap = signals.get("vwap", close)
        support = signals.get("support", close * 0.97)
        
        if vwap and vwap < close:
            return round((close + vwap) / 2, 2)
        elif support and support > close * 0.92:
            return round((close + support) / 2, 2)
        return close
    
    elif "Sell" in category:
        # For sell: slightly above current (short entry near resistance)
        resistance = signals.get("resistance", close * 1.03)
        if resistance and resistance > close:
            return round((close + resistance) / 2, 2)
        return close
    
    return close


def _compute_stop_loss(signals: dict, atr: float, profile: str, category: str) -> float:
    """Compute dynamic ATR-based stop loss."""
    close = signals.get("close", 0)
    
    # ATR multiplier by profile
    multiplier = {"Conservative": 1.5, "Moderate": 2.0, "Aggressive": 2.5}.get(profile, 2.0)
    
    support = signals.get("support", 0)
    
    if "Buy" in category or category == "Hold":
        # Stop below recent support or ATR band
        atr_stop = close - (atr * multiplier)
        if support and support > 0:
            sl = min(atr_stop, support * 0.99)  # just below support
        else:
            sl = atr_stop
        return max(sl, close * 0.85)  # floor at 15% loss
    
    else:  # Sell signal
        # Stop above recent resistance
        return close + (atr * multiplier)


def _compute_take_profits(
    signals: dict,
    atr: float,
    profile: str,
    category: str,
    entry: float
) -> tuple:
    """Compute 3-tier take profit levels."""
    
    multipliers = {
        "Conservative": (1.5, 2.5, 3.5),
        "Moderate":     (2.0, 3.5, 5.0),
        "Aggressive":   (2.5, 4.5, 7.0),
    }.get(profile, (2.0, 3.5, 5.0))
    
    resistance = signals.get("resistance", 0)
    
    if "Buy" in category or category == "Hold":
        tp1 = entry + atr * multipliers[0]
        tp2 = entry + atr * multipliers[1]
        tp3 = entry + atr * multipliers[2]
        
        # Align TP1 with nearest resistance if available
        if resistance and entry < resistance < tp2:
            tp1 = resistance * 0.995
    else:
        # Short trade — targets below entry
        tp1 = entry - atr * multipliers[0]
        tp2 = entry - atr * multipliers[1]
        tp3 = entry - atr * multipliers[2]
    
    return tp1, tp2, tp3


def _compute_position_size(
    capital: Optional[float],
    risk_pct: float,
    entry: float,
    stop_loss: float
) -> Optional[dict]:
    """Compute position sizing based on capital and risk %."""
    if not capital or entry == 0 or stop_loss == 0:
        return None
    
    risk_per_share = abs(entry - stop_loss)
    if risk_per_share == 0:
        return None
    
    max_risk_idr = capital * (risk_pct / 100)
    shares = int(max_risk_idr / risk_per_share)
    
    # Round to nearest lot (100 shares for IDX stocks)
    lots = max(1, shares // 100)
    shares = lots * 100
    
    return {
        "lots":         lots,
        "shares":       shares,
        "capital_used": round(shares * entry, 0),
        "max_risk_idr": round(max_risk_idr, 0),
    }


def _compute_confidence(score: dict, signals: dict) -> float:
    """Compute confidence score 0–100."""
    base = score.get("total", 50)
    
    # Confidence is highest when score is extreme (not near 50)
    # and when volume confirms
    distance_from_neutral = abs(base - 50)
    vol_boost = 5 if signals.get("high_volume") else 0
    
    confidence = min(100, 50 + distance_from_neutral * 0.8 + vol_boost)
    return round(confidence, 1)


def _empty_recommendation() -> dict:
    return {
        "recommendation": "N/A",
        "entry_price":    0,
        "stop_loss":      0,
        "take_profit_1":  0,
        "take_profit_2":  0,
        "take_profit_3":  0,
        "rr_ratio":       0,
        "confidence":     0,
        "risk_amount":    0,
        "reward_amount":  0,
        "position":       None,
        "summary":        "Insufficient data for analysis.",
        "bullish_pct":    50,
        "bearish_pct":    50,
    }


# ── AI SUMMARY GENERATOR ─────────────────────────────────────────────────────

def generate_analyst_summary(
    signals: dict,
    score: dict,
    df: pd.DataFrame
) -> str:
    """
    Generate a hedge-fund-style analyst summary from the computed signals.
    """
    close   = signals.get("close", 0)
    rsi     = signals.get("rsi", 50)
    vol_ratio = signals.get("volume_ratio", 1)
    category  = score.get("category", "Hold")
    bullish   = score.get("bullish_pct", 50)
    
    trend_dir = _get_trend_description(signals)
    rsi_desc  = _get_rsi_description(rsi)
    macd_desc = _get_macd_description(signals)
    vol_desc  = _get_volume_description(vol_ratio)
    ma_desc   = _get_ma_description(signals)
    
    # Price change
    if len(df) >= 20:
        price_20d_ago = df["Close"].iloc[-20]
        chg_20d = ((close - price_20d_ago) / price_20d_ago) * 100
        chg_desc = f"naik {chg_20d:.1f}%" if chg_20d >= 0 else f"turun {abs(chg_20d):.1f}%"
    else:
        chg_desc = "bergerak sideways"
    
    summary = (
        f"Saham ini menunjukkan {trend_dir} dengan skor AI {score['total']:.0f}/100 "
        f"({category.upper()}). "
        f"Dalam 20 hari terakhir, harga telah {chg_desc}. "
        f"{rsi_desc} "
        f"{macd_desc} "
        f"{ma_desc} "
        f"{vol_desc} "
        f"Probabilitas bullish berada di {bullish:.0f}% berdasarkan agregasi sinyal teknikal. "
        f"{_get_risk_note(score)}"
    )
    
    return summary


def _get_trend_description(signals: dict) -> str:
    above_50  = signals.get("price_above_sma50", False)
    above_200 = signals.get("price_above_sma200", False)
    
    if above_50 and above_200:
        return "momentum bullish yang kuat"
    elif above_50:
        return "tren bullish jangka menengah"
    elif above_200:
        return "pemulihan di atas SMA200"
    else:
        return "tekanan jual yang persisten"


def _get_rsi_description(rsi: float) -> str:
    if rsi < 30:
        return f"RSI oversold di {rsi:.0f}, mengindikasikan potensi reversal bullish."
    elif rsi < 40:
        return f"RSI lemah di {rsi:.0f}, namun belum pada level ekstrem."
    elif rsi > 70:
        return f"RSI overbought di {rsi:.0f}, waspadai potensi koreksi."
    elif rsi > 60:
        return f"RSI kuat di {rsi:.0f}, momentum masih terjaga."
    else:
        return f"RSI netral di {rsi:.0f}, tidak ada divergensi signifikan."


def _get_macd_description(signals: dict) -> str:
    hist = signals.get("macd_hist", 0)
    if signals.get("macd_bullish_cross"):
        return "MACD baru saja membentuk bullish crossover — sinyal beli kuat."
    elif signals.get("macd_bearish_cross"):
        return "MACD membentuk bearish crossover — sinyal jual."
    elif hist > 0:
        return "MACD histogram positif, momentum bullish sedang berlangsung."
    else:
        return "MACD histogram negatif, tekanan bearish masih dominan."


def _get_ma_description(signals: dict) -> str:
    if signals.get("golden_cross"):
        return "Golden cross terbentuk — sinyal beli jangka panjang yang kuat."
    elif signals.get("death_cross"):
        return "Death cross terbentuk — sinyal jual jangka panjang."
    elif signals.get("price_above_sma50") and signals.get("price_above_sma200"):
        return "Harga di atas SMA50 dan SMA200, mengkonfirmasi uptrend menengah-panjang."
    elif not signals.get("price_above_sma50") and not signals.get("price_above_sma200"):
        return "Harga di bawah SMA50 dan SMA200 — downtrend dominan."
    else:
        return "Posisi harga terhadap moving average bersifat mixed."


def _get_volume_description(vol_ratio: float) -> str:
    pct = int((vol_ratio - 1) * 100)
    if vol_ratio > 1.5:
        return f"Volume meningkat {pct}% di atas rata-rata 20 hari, mengindikasikan akumulasi institusi."
    elif vol_ratio > 1.2:
        return f"Volume sedikit di atas rata-rata ({pct}%), konfirmasi sedang."
    elif vol_ratio < 0.7:
        return "Volume tipis — pergerakan harga kurang meyakinkan."
    else:
        return "Volume dalam kisaran normal."


def _get_risk_note(score: dict) -> str:
    total = score.get("total", 50)
    if total >= 75:
        return "Setup teknikal sangat favorable. Manajemen risiko tetap diperlukan."
    elif total >= 55:
        return "Setup teknikal cukup favorable dengan konfirmasi sedang."
    elif total >= 45:
        return "Sinyal mixed. Disarankan menunggu konfirmasi lebih lanjut."
    else:
        return "Setup teknikal melemah. Pertimbangkan untuk reduce exposure."
