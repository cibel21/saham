"""
scoring.py
AI scoring system for buy/sell/hold recommendation generation.
Uses weighted multi-factor scoring (0-100 scale).
"""

import numpy as np
from typing import Tuple


# ── WEIGHTS ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "trend":      0.30,   # 30% — trend alignment
    "momentum":   0.25,   # 25% — RSI, MACD, Stochastic
    "volume":     0.20,   # 20% — volume confirmation
    "volatility": 0.15,   # 15% — Bollinger, ATR
    "sentiment":  0.10,   # 10% — news/external sentiment placeholder
}


def compute_score(signals: dict, risk_profile: str = "Moderate") -> dict:
    """
    Compute the AI score (0–100) from technical signals.
    
    Args:
        signals: Dictionary of current indicator values from indicators.py
        risk_profile: 'Conservative', 'Moderate', or 'Aggressive'
    
    Returns:
        Dictionary with score, category, and sub-scores
    """
    
    trend_score      = _score_trend(signals)
    momentum_score   = _score_momentum(signals)
    volume_score     = _score_volume(signals)
    volatility_score = _score_volatility(signals)
    sentiment_score  = _score_sentiment(signals)   # placeholder 50 = neutral
    
    # Weighted composite score
    raw_score = (
        trend_score      * WEIGHTS["trend"]      +
        momentum_score   * WEIGHTS["momentum"]   +
        volume_score     * WEIGHTS["volume"]      +
        volatility_score * WEIGHTS["volatility"]  +
        sentiment_score  * WEIGHTS["sentiment"]
    )
    
    # Apply risk profile adjustment
    score = _apply_risk_profile(raw_score, risk_profile)
    score = max(0, min(100, score))
    
    category = _score_to_category(score)
    
    return {
        "total":      round(score, 1),
        "category":   category,
        "trend":      round(trend_score, 1),
        "momentum":   round(momentum_score, 1),
        "volume":     round(volume_score, 1),
        "volatility": round(volatility_score, 1),
        "sentiment":  round(sentiment_score, 1),
        "bullish_pct": _compute_bullish_probability(signals),
    }


# ── SUB-SCORERS ───────────────────────────────────────────────────────────────

def _score_trend(signals: dict) -> float:
    """Score trend alignment (0–100)."""
    score = 50.0  # neutral baseline
    
    close   = signals.get("close", 0)
    sma_50  = signals.get("sma_50", 0)
    sma_200 = signals.get("sma_200", 0)
    ema_20  = signals.get("ema_20", 0)
    sma_20  = signals.get("sma_20", 0)
    
    if close and sma_50:
        if close > sma_50:
            score += 12
        else:
            score -= 12
    
    if close and sma_200:
        if close > sma_200:
            score += 15
        else:
            score -= 15
    
    if sma_50 and sma_200:
        if sma_50 > sma_200:  # golden cross territory
            score += 10
        else:
            score -= 10
    
    if close and ema_20:
        if close > ema_20:
            score += 8
        else:
            score -= 8
    
    if signals.get("golden_cross"):
        score += 10
    if signals.get("death_cross"):
        score -= 10
    
    return max(0, min(100, score))


def _score_momentum(signals: dict) -> float:
    """Score momentum indicators (0–100)."""
    score = 50.0
    
    rsi = signals.get("rsi", 50)
    
    # RSI scoring
    if rsi < 30:
        score += 20   # oversold = bullish setup
    elif rsi < 40:
        score += 10
    elif rsi > 70:
        score -= 20   # overbought = bearish
    elif rsi > 60:
        score -= 8
    # 40–60 = neutral, no change
    
    # MACD scoring
    macd_hist = signals.get("macd_hist", 0)
    prev_hist = signals.get("prev_macd_hist", 0)
    
    if macd_hist > 0 and prev_hist <= 0:    # fresh bullish crossover
        score += 18
    elif macd_hist > 0 and macd_hist > prev_hist:  # strengthening
        score += 8
    elif macd_hist < 0 and prev_hist >= 0:   # fresh bearish crossover
        score -= 18
    elif macd_hist < 0 and macd_hist < prev_hist:   # weakening
        score -= 8
    
    # Stochastic scoring
    stoch_k = signals.get("stoch_k", 50)
    if stoch_k < 20:
        score += 8
    elif stoch_k > 80:
        score -= 8
    
    # Williams %R
    willi = signals.get("williams_r", -50)
    if willi < -80:
        score += 5
    elif willi > -20:
        score -= 5
    
    return max(0, min(100, score))


def _score_volume(signals: dict) -> float:
    """Score volume signals (0–100)."""
    score = 50.0
    vol_ratio = signals.get("volume_ratio", 1.0)
    
    if vol_ratio > 2.0:
        score += 20   # Very high volume
    elif vol_ratio > 1.5:
        score += 12
    elif vol_ratio > 1.2:
        score += 6
    elif vol_ratio < 0.5:
        score -= 15   # Low volume — weak signal
    elif vol_ratio < 0.8:
        score -= 7
    
    # OBV trend (if OBV is rising along with price, bullish)
    # Simple proxy: high_volume + price_above_sma50
    if signals.get("high_volume") and signals.get("price_above_sma50"):
        score += 10
    elif signals.get("high_volume") and not signals.get("price_above_sma50"):
        score -= 5  # high volume on downward move = bearish distribution
    
    return max(0, min(100, score))


def _score_volatility(signals: dict) -> float:
    """Score volatility / Bollinger signals (0–100)."""
    score = 50.0
    bb_pct = signals.get("bb_pct", 0.5)
    
    # BB position
    if bb_pct < 0.05:
        score += 15   # near lower band = oversold
    elif bb_pct < 0.2:
        score += 8
    elif bb_pct > 0.95:
        score -= 15   # near upper band = overbought
    elif bb_pct > 0.8:
        score -= 8
    
    # BB width (squeeze vs expansion)
    bb_width = signals.get("bb_width", 0)
    if bb_width < 0.05:
        # Bollinger squeeze — neutral but potential breakout
        pass
    
    if signals.get("bb_breakout_up"):
        score += 10
    if signals.get("bb_breakout_down"):
        score -= 10
    
    return max(0, min(100, score))


def _score_sentiment(signals: dict) -> float:
    """
    Sentiment score placeholder.
    Returns 50 (neutral) unless news sentiment data is provided.
    """
    return signals.get("sentiment_score", 50.0)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _apply_risk_profile(score: float, profile: str) -> float:
    """
    Adjust score threshold sensitivity based on risk profile.
    Conservative shifts score toward middle (less extreme).
    Aggressive amplifies extremes slightly.
    """
    center = 50.0
    diff = score - center
    
    if profile == "Conservative":
        return center + diff * 0.75   # dampen swings
    elif profile == "Aggressive":
        return center + diff * 1.20   # amplify swings
    else:
        return score   # Moderate = no change


def _score_to_category(score: float) -> str:
    if score >= 80:
        return "Strong Buy"
    elif score >= 65:
        return "Buy"
    elif score >= 45:
        return "Hold"
    elif score >= 25:
        return "Sell"
    else:
        return "Strong Sell"


def _compute_bullish_probability(signals: dict) -> float:
    """
    Compute a bullish probability % from a simple signal count.
    """
    bullish_signals = 0
    total_signals = 0
    
    checks = [
        (signals.get("rsi", 50) < 40,                         True),
        (signals.get("rsi", 50) > 60,                         False),
        (signals.get("macd_hist", 0) > 0,                     True),
        (signals.get("macd_hist", 0) < 0,                     False),
        (signals.get("price_above_sma50"),                     True),
        (not signals.get("price_above_sma50", True),           False),
        (signals.get("price_above_sma200"),                    True),
        (not signals.get("price_above_sma200", True),          False),
        (signals.get("bb_pct", 0.5) < 0.3,                    True),
        (signals.get("bb_pct", 0.5) > 0.7,                    False),
        (signals.get("volume_ratio", 1) > 1.3,                True),
        (signals.get("macd_bullish_cross"),                    True),
        (signals.get("macd_bearish_cross"),                    False),
        (signals.get("golden_cross"),                          True),
        (signals.get("death_cross"),                           False),
    ]
    
    for condition, is_bullish in checks:
        if condition:
            total_signals += 1
            if is_bullish:
                bullish_signals += 1
    
    if total_signals == 0:
        return 50.0
    
    return round((bullish_signals / total_signals) * 100, 1)


def get_score_color(score: float) -> str:
    """Return a hex color based on score."""
    if score >= 80:
        return "#00FF88"
    elif score >= 65:
        return "#00CCAA"
    elif score >= 45:
        return "#FFDD00"
    elif score >= 25:
        return "#FF8844"
    else:
        return "#FF3344"


def get_recommendation_color(category: str) -> str:
    """Return color for recommendation badge."""
    colors = {
        "Strong Buy":  "#00FF88",
        "Buy":         "#00CCAA",
        "Hold":        "#FFDD00",
        "Sell":        "#FF8844",
        "Strong Sell": "#FF3344",
    }
    return colors.get(category, "#FFFFFF")
