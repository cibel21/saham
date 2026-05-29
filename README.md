# 📈 Stock AI Advisor

**Bloomberg Terminal meets Hedge Fund AI — in a Streamlit app.**

A professional AI-powered stock analysis and recommendation engine for Indonesian (IDX) and global equities. Combines technical analysis, AI scoring, forecasting, and risk management into one clean interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Live Charts** | Candlestick + MA overlays, RSI, MACD, Volume |
| 🤖 **AI Scoring** | 0–100 score across Trend, Momentum, Volume, Volatility |
| 💼 **Trade Setup** | Entry price, Stop Loss, Take Profits, Risk/Reward |
| 📈 **Forecasting** | XGBoost / GradientBoosting price forecast (7–60 days) |
| 🏦 **Fundamentals** | P/E, P/B, ROE, Market Cap, 52W Range, and more |
| 🔀 **Multi-Compare** | Normalized return comparison across multiple tickers |
| 💰 **Position Sizing** | ATR-based lot sizing based on your capital |
| 🧠 **AI Summary** | Analyst-style written summary of current setup |

---

## 🚀 Quick Start

### Local

```bash
git clone https://github.com/yourname/stock-ai-advisor
cd stock-ai-advisor
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud

1. Fork/push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 📁 Project Structure

```
stock-ai-advisor/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
│
├── utils/
│   ├── data_loader.py      # yfinance data fetching & caching
│   ├── indicators.py       # Technical indicators (ta library)
│   ├── scoring.py          # AI scoring system (0–100)
│   ├── recommendation.py   # Trade setup & analyst summary
│   ├── forecasting.py      # XGBoost / sklearn price forecast
│   └── charts.py           # All Plotly chart definitions
│
├── .streamlit/
│   └── config.toml         # Dark theme config
│
└── README.md
```

---

## 📊 AI Scoring System

Scores are computed as a **weighted multi-factor model**:

| Factor | Weight | Signals |
|---|---|---|
| Trend | 30% | SMA position, Golden/Death Cross |
| Momentum | 25% | RSI, MACD crossover, Stochastic |
| Volume | 20% | Volume ratio, OBV |
| Volatility | 15% | Bollinger Band position, ATR |
| Sentiment | 10% | News sentiment placeholder |

**Score → Recommendation:**
- 80–100 → **Strong Buy** 🟢
- 65–79  → **Buy** 🟢
- 45–64  → **Hold** 🟡
- 25–44  → **Sell** 🔴
- 0–24   → **Strong Sell** 🔴

---

## 🛡️ Risk Profiles

| Profile | Stop Loss Multiplier | TP Multiplier |
|---|---|---|
| Conservative | 1.5× ATR | Lower targets |
| Moderate | 2.0× ATR | Balanced |
| Aggressive | 2.5× ATR | Wider targets |

---

## 📦 Dependencies

```
streamlit, pandas, numpy, yfinance, plotly, ta,
scikit-learn, matplotlib, requests, xgboost
```

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**.
It is **NOT** financial advice. Always do your own research (DYOR) before making any investment decisions.
Past performance does not guarantee future results.

---

## 🔮 Roadmap

- [ ] Telegram alerts for strong signals
- [ ] Backtesting engine
- [ ] News sentiment API integration
- [ ] Portfolio tracker
- [ ] PDF report export
- [ ] Fear & Greed Index

---

Made with ❤️ using Streamlit + yfinance + Plotly
