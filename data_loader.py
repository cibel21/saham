"""
data_loader.py
Handles all data fetching from yfinance with caching and error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import requests


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Load historical stock data from yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g. 'BBCA.JK')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame with OHLCV data, or empty DataFrame on failure
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        # Clean column names
        df.columns = [col.capitalize() for col in df.columns]
        
        # Drop any rows with all NaN OHLCV
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        
        # Ensure we have enough data
        if len(df) < 20:
            return pd.DataFrame()
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(ticker: str) -> dict:
    """
    Get fundamental information about the stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant fields with fallbacks
        result = {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", info.get("forwardPE", 0)),
            "pb_ratio": info.get("priceToBook", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", 1.0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "currency": info.get("currency", "IDR"),
            "exchange": info.get("exchange", "N/A"),
            "description": info.get("longBusinessSummary", "No description available."),
            "eps": info.get("trailingEps", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "profit_margins": info.get("profitMargins", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "current_ratio": info.get("currentRatio", 0),
            "roe": info.get("returnOnEquity", 0),
        }
        
        return result
    
    except Exception:
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "pe_ratio": 0,
            "pb_ratio": 0,
            "dividend_yield": 0,
            "beta": 1.0,
            "52w_high": 0,
            "52w_low": 0,
            "avg_volume": 0,
            "currency": "IDR",
            "exchange": "N/A",
            "description": "Info not available.",
            "eps": 0,
            "revenue_growth": 0,
            "profit_margins": 0,
            "debt_to_equity": 0,
            "current_ratio": 0,
            "roe": 0,
        }


@st.cache_data(ttl=300)
def get_multiple_stocks(tickers: list, period: str = "1y") -> dict:
    """
    Load data for multiple stocks for comparison.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
    
    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    result = {}
    for ticker in tickers:
        df = load_stock_data(ticker, period)
        if not df.empty:
            result[ticker] = df
    return result


def get_market_news(ticker: str) -> list:
    """
    Get recent news headlines for a stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        List of news dictionaries
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            return news[:10]  # Return top 10 news items
        return []
    except Exception:
        return []


def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol is valid.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d")
        return not df.empty
    except Exception:
        return False


def format_number(value: float, currency: str = "IDR") -> str:
    """
    Format large numbers for display.
    
    Args:
        value: Numeric value
        currency: Currency symbol
    
    Returns:
        Formatted string
    """
    if value == 0:
        return "N/A"
    
    abs_val = abs(value)
    
    if abs_val >= 1e12:
        return f"{currency} {value/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{currency} {value/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{currency} {value/1e6:.2f}M"
    else:
        return f"{currency} {value:,.0f}"
