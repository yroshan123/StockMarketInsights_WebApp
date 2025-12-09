# Stock & Sentiment – Multi-Horizon Price Prediction (Streamlit)

A Streamlit web application that combines **stock price data** and **Twitter sentiment** to train a small **LSTM model** and forecast **future returns and prices** over multiple horizons (1, 3, 7, 14 days).

> ⚠️ **Disclaimer**  
> This project is for **educational and research purposes only**.  
> It is **not** financial advice and should not be used for live trading or investment decisions.

---

## Features

- 📈 **Multi-horizon forecasts**  
  Predict future returns and prices for **1, 3, 7, or 14 days** ahead.

- 🧠 **Hybrid data: prices + tweets**  
  - Historical OHLCV stock data  
  - Tweets mapped to the same dates and tickers  
  - Sentiment scores computed per tweet/day

- 💬 **Pluggable sentiment engines**
  - **Quick sentiment (VADER)** – rule-based, fast
  - **Deep sentiment (BERT)** – transformer-based, slower but more expressive

- 🔁 **Time-series aware training**
  - Time-ordered train/test split (no shuffling)
  - Scalers fit only on the training set (no data leakage)
  - Configurable:
    - Sequence length (lookback window)
    - Epochs
    - Batch size
    - Learning rate
    - Test split ratio

- 📊 **Interactive visualizations**
  - **Price & Sentiment Over Time**  
    Dual-axis Plotly chart showing:
    - Closing price
    - Sentiment score
    - Last close highlighted
  - **Sentiment vs. Future Returns**  
    Correlation bar chart answering:
    > “How does sentiment relate to future 1/3/7/14-day returns?”

- ⚙️ **Cached sentiment**
  - Sentiment scores stored to a **Parquet cache file**
  - Re-uses previous results when possible to avoid recomputing

---

## Project Structure

```text
StockMarketInsights_WebApp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Data/                  # Example / user-provided CSV data (prices & tweets)
├── cache/                 # Sentiment cache (Parquet) – optional, created at runtime
└── ml/
    ├── config.py          # Paths & configuration (defaults for CSVs, cache, output)
    ├── sentiment.py       # SentimentEngine + sentiment computation utilities
    └── pipeline.py        # Feature building + LSTM training & prediction
