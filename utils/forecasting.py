"""
forecasting.py
Price forecasting using XGBoost regression with feature engineering.
Falls back gracefully if xgboost is unavailable.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def forecast_price(df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """
    Generate price forecast using XGBoost (falls back to linear regression).
    
    Args:
        df: OHLCV DataFrame with computed indicators
        days_ahead: Number of future trading days to forecast
    
    Returns:
        DataFrame with forecast dates and predicted prices
    """
    try:
        return _xgboost_forecast(df, days_ahead)
    except ImportError:
        return _sklearn_forecast(df, days_ahead)
    except Exception as e:
        return _linear_forecast(df, days_ahead)


def _xgboost_forecast(df: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    """XGBoost-based price forecast."""
    from xgboost import XGBRegressor
    
    features_df = _build_features(df)
    if features_df.empty or len(features_df) < 30:
        return pd.DataFrame()
    
    X = features_df.drop(columns=["target"])
    y = features_df["target"]
    
    # Train/test split (no shuffle — time series)
    split = int(len(X) * 0.85)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    
    # Generate future features iteratively
    return _generate_forecast_df(df, model, features_df, X, days_ahead, method="xgb")


def _sklearn_forecast(df: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    """Gradient Boosting from sklearn as fallback."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    features_df = _build_features(df)
    if features_df.empty or len(features_df) < 30:
        return pd.DataFrame()
    
    X = features_df.drop(columns=["target"])
    y = features_df["target"]
    
    split = int(len(X) * 0.85)
    model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    return _generate_forecast_df(df, model, features_df, X, days_ahead, method="gbr")


def _linear_forecast(df: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
    """Simple linear trend extrapolation as last resort."""
    from sklearn.linear_model import LinearRegression
    
    close_prices = df["Close"].dropna().values
    X = np.arange(len(close_prices)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X[-60:], close_prices[-60:])  # use last 60 days
    
    future_X = np.arange(len(close_prices), len(close_prices) + days_ahead).reshape(-1, 1)
    preds = model.predict(future_X)
    
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=days_ahead + 1)[1:]
    
    return pd.DataFrame({
        "Date": future_dates[:days_ahead],
        "Forecast": preds[:days_ahead],
        "Upper": preds[:days_ahead] * 1.03,
        "Lower": preds[:days_ahead] * 0.97,
        "Method": "Linear Trend",
    }).set_index("Date")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for ML forecasting."""
    data = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        data[f"rolling_mean_{window}"] = data["Close"].rolling(window).mean()
        data[f"rolling_std_{window}"]  = data["Close"].rolling(window).std()
    
    # Returns
    data["return_1d"]  = data["Close"].pct_change(1)
    data["return_5d"]  = data["Close"].pct_change(5)
    data["return_20d"] = data["Close"].pct_change(20)
    
    # Volume features
    data["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
    
    # Technical indicators (if present)
    for col in ["RSI", "MACD", "MACD_Hist", "ATR", "BB_Pct"]:
        if col in data.columns:
            data[col.lower()] = data[col]
    
    # Target: next day close
    data["target"] = data["Close"].shift(-1)
    
    # Drop NaN rows
    data = data.dropna()
    
    # Keep only numeric feature columns
    feature_cols = [c for c in data.columns if c.startswith(
        ("close_lag", "rolling", "return", "volume", "rsi", "macd", "atr", "bb", "target")
    )]
    
    return data[feature_cols]


def _generate_forecast_df(df, model, features_df, X, days_ahead, method="xgb") -> pd.DataFrame:
    """Generate future predictions from trained model."""
    last_row = X.iloc[-1:].copy()
    last_close = df["Close"].iloc[-1]
    
    forecasts = []
    current_close = last_close
    
    for i in range(days_ahead):
        pred = float(model.predict(last_row)[0])
        # Add small noise to simulate uncertainty
        noise = np.random.normal(0, current_close * 0.001)
        pred_noisy = pred + noise
        forecasts.append(pred_noisy)
        
        # Shift lag features for next step (approximate)
        if "close_lag_1" in last_row.columns:
            new_row = last_row.copy()
            # Shift lags
            for lag in [20, 10, 5, 3, 2]:
                col_prev = f"close_lag_{lag-1}" if lag > 1 else None
                col_curr = f"close_lag_{lag}"
                if col_prev and col_prev in new_row.columns:
                    new_row[col_curr] = new_row[col_prev].values[0]
            new_row["close_lag_1"] = pred_noisy
            last_row = new_row
        
        current_close = pred_noisy
    
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=days_ahead + 1)[1:]
    
    forecasts_arr = np.array(forecasts[:days_ahead])
    std = forecasts_arr.std() if len(forecasts_arr) > 1 else forecasts_arr[0] * 0.02
    
    return pd.DataFrame({
        "Date":     future_dates[:days_ahead],
        "Forecast": np.round(forecasts_arr, 2),
        "Upper":    np.round(forecasts_arr + 2 * std, 2),
        "Lower":    np.round(forecasts_arr - 2 * std, 2),
        "Method":   method.upper(),
    }).set_index("Date")
