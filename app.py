# app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.config import (
    DEFAULT_STOCKS_CSV,
    DEFAULT_TWEETS_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SENTIMENT_CACHE,
)
from ml.sentiment import SentimentEngine, compute_or_load_sentiment
from ml.pipeline import fit_and_predict_one_ticker, build_features_for_ticker

st.set_page_config(page_title="Stock & Sentiment – Multi-Horizon Prediction", layout="wide")

# =======================
# INTRO
# =======================
st.title("📈 Stock & Sentiment: Next Price Prediction")

st.markdown(
    """
**What this app does (in simple terms):**
- Reads your **stock prices** and **tweets** for the company you choose.
- Turns the text into a **sentiment score** (how positive/negative the crowd feels).
- Trains a small **LSTM** on the past and predicts the **future return** for the horizon you pick (1/3/7/14 days).
- Converts that return into a **projected price** and shows clear **charts**.

*Note:* This is a learning tool. It shows patterns, not guarantees.
"""
)

st.caption("No data leakage: time-ordered split; scalers fit on train only.")

# =======================
# SIDEBAR – CONFIG & HELP
# =======================
st.sidebar.header("Configuration")

stocks_path = st.sidebar.text_input("Stocks CSV path", value=DEFAULT_STOCKS_CSV)
tweets_path = st.sidebar.text_input("Tweets CSV path", value=DEFAULT_TWEETS_CSV)
output_dir = st.sidebar.text_input("Output folder", value=DEFAULT_OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

sentiment_cache = st.sidebar.text_input("Sentiment cache (parquet)", value=DEFAULT_SENTIMENT_CACHE)

# Human-friendly sentiment model names
sentiment_options = {
    "Quick sentiment (rule-based, fast)": "vader",
    "Deep sentiment (AI model, slower)": "bert",
}
sentiment_choice = st.sidebar.selectbox(
    "Sentiment analysis method",
    list(sentiment_options.keys()),
    index=0,
)
sentiment_mode = sentiment_options[sentiment_choice]

horizon = st.sidebar.selectbox("Forecast horizon (days)", [1, 3, 7, 14], index=2)  # default 7
seq_len = st.sidebar.slider("Sequence length (lookback days)", 10, 60, 20, step=2)
epochs = st.sidebar.slider("Epochs", 10, 200, 50, step=10)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
lr = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
test_split = st.sidebar.slider("Test split (time-ordered)", 0.1, 0.4, 0.2, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("How to use this app")
st.sidebar.markdown(
    """
1. Set paths (or use defaults).  
2. Choose a **sentiment analysis method**  
   - *Quick sentiment* → faster, simpler.  
   - *Deep sentiment* → slower, more advanced.  
3. Pick a stock.  
4. Pick a forecast **horizon** (1/3/7/14 days).  
5. Click **Run training & predict**.  
6. Read results; scroll for graphs.
"""
)

# =======================
# “How to read this page”
# =======================
with st.expander("ℹ️ How to read this page"):
    st.markdown(
        """
**Top numbers (KPIs):**
- **RMSE (return):** error on unseen test data (lower is better).  
- **Predicted return:** expected gain/loss for the next *H* days.  
- **Predicted price:** last close × (1 + predicted return).

**Price & Sentiment chart:**
- Blue = price; Green = sentiment; Red dashed = neutral (0).  
- Final dot = most recent close (the starting point for the prediction).

**Correlation chart:**
- Shows how different **sentiment measures** move with **future returns** (1, 3, 7, 14 days).  
- Bars above zero → move in the same direction; bars below zero → tend to move in opposite directions.
"""
    )

# =======================
# LOAD DATA
# =======================
@st.cache_data(show_spinner=True)
def load_sources(_stocks_path, _tweets_path):
    stocks = pd.read_csv(_stocks_path)
    tweets = pd.read_csv(_tweets_path)
    return stocks, tweets

try:
    stocks, tweets = load_sources(stocks_path, tweets_path)
except Exception as e:
    st.error(f"Failed to load CSVs: {e}")
    st.stop()

needed_stocks = {"Stock Name", "Date", "Open", "High", "Low", "Close", "Volume"}
needed_tweets = {"Stock Name", "Date", "Tweet"}
if not needed_stocks.issubset(stocks.columns):
    st.error(f"Stocks CSV missing columns: {needed_stocks - set(stocks.columns)}")
    st.stop()
if not needed_tweets.issubset(tweets.columns):
    st.error(f"Tweets CSV missing columns: {needed_tweets - set(tweets.columns)}")
    st.stop()

tickers = sorted(stocks["Stock Name"].dropna().unique().tolist())
ticker = st.selectbox("Choose a stock", options=tickers)

# =======================
# SENTIMENT (CACHED)
# =======================
@st.cache_resource(show_spinner=True)
def get_engine(mode):
    return SentimentEngine(mode=mode)

engine = get_engine(sentiment_mode)

@st.cache_data(show_spinner=True)
def get_scored_tweets(tweets_df, cache_path, mode_key):
    # mode_key ensures cache invalidates if you switch sentiment models
    eng = SentimentEngine(mode=mode_key)
    return compute_or_load_sentiment(tweets_df, cache_path, eng)

with st.spinner("Analyzing tweet sentiment…"):
    tweets_scored = get_scored_tweets(tweets, sentiment_cache, sentiment_mode)

# =======================
# SUBSET
# =======================
s_df = stocks[stocks["Stock Name"] == ticker].copy()
t_df = tweets_scored[tweets_scored["Stock Name"] == ticker].copy()
if s_df.empty:
    st.warning("No price data for this ticker.")
    st.stop()
if t_df.empty:
    st.warning("No tweets for this ticker; using zeros and forward-fill.")
    t_df = pd.DataFrame(
        {"Stock Name": [ticker], "Date": [s_df["Date"].iloc[0]], "Tweet": [""], "Sentiment_Score": [0.0]}
    )

# =======================
# TRAIN & PREDICT BUTTON
# =======================
col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("🚀 Run training & predict"):
        st.session_state["run_model"] = True

run_it = st.session_state.get("run_model", False)

if run_it:
    with st.spinner("Training LSTM model & generating prediction…"):
        try:
            res = fit_and_predict_one_ticker(
                s_df,
                t_df,
                seq_len=seq_len,
                epochs=epochs,
                batch=batch_size,
                lr=lr,
                test_ratio=test_split,
                horizon=horizon,  # <— dynamic horizon
            )
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        st.success("Training complete.")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(f"RMSE ({horizon}-day return)", f"{res['rmse_return']:.4f}")
        kpi2.metric(f"Predicted {horizon}-day return", f"{res['pred_next_return']*100:.2f}%")
        kpi3.metric(f"Predicted price in {horizon} days", f"${res['pred_price_in_h']:.2f}")

# =======================
# VISUAL – PRICE & SENTIMENT
# =======================
st.markdown("## Price & Sentiment (time-aligned)")

# Build aligned dataset (unscaled) for plotting, with same horizon for target selection
with st.spinner("Preparing data for charts…"):
    df_full, feat_cols, target_col, _, _, _, _, _, _ = build_features_for_ticker(
        s_df, t_df, seq_len=seq_len, test_ratio=test_split, horizon=horizon
    )

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_full["Date"], y=df_full["Close"], name="Close", mode="lines"))
fig.add_trace(
    go.Scatter(
        x=df_full["Date"],
        y=df_full["Sentiment_Score"],
        name="Sentiment",
        mode="lines",
        yaxis="y2",
        opacity=0.7,
    )
)
fig.update_layout(
    title=f"{ticker}: Price & Sentiment",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Sentiment (−1..+1)", overlaying="y", side="right"),
    legend=dict(orientation="h", x=0, y=1.15),
    margin=dict(l=10, r=10, t=50, b=10),
    hovermode="x unified",
)
# mark last close
last_close = float(df_full["Close"].iloc[-1])
last_date = df_full["Date"].iloc[-1]
fig.add_trace(
    go.Scatter(
        x=[last_date],
        y=[last_close],
        mode="markers+text",
        name="Last Close",
        text=["Last close"],
        textposition="top center",
        marker=dict(size=10, line=dict(width=1, color="black")),
    )
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    """
**What to look for:**
- When **sentiment** stays above 0 for a while, prices often trend up.  
- When it stays below 0, prices often trend down.  
- The **final dot** is the latest close — the starting point for the prediction.
"""
)

# =======================
# ALTERNATIVE VISUAL – CORRELATION BAR CHART (BEGINNER-FRIENDLY)
# =======================
st.markdown("## How sentiment relates to future returns")

corr_cols = [
    "Sentiment_Score",
    "sentiment_volatility",
    "lagged_sentiment",
    "future_returns_lag_1",
    "future_returns_lag_3",
    "future_returns_lag_7",
    "future_returns_lag_14",
]

plot_df = df_full.copy()
plot_df["ma_20"] = plot_df["Close"].rolling(20, min_periods=1).mean()
plot_df["z_score"] = (plot_df["Close"] - plot_df["ma_20"]) / plot_df["Close"].rolling(7, min_periods=1).std()
plot_df["sentiment_volatility"] = plot_df["Sentiment_Score"].rolling(3, min_periods=1).std()
plot_df["lagged_sentiment"] = plot_df["Sentiment_Score"].shift(1)
plot_df = plot_df.dropna()

if len(plot_df) > 5:
    corr = plot_df[corr_cols].corr()

    feature_side = ["Sentiment_Score", "sentiment_volatility", "lagged_sentiment"]
    target_side = [
        "future_returns_lag_1",
        "future_returns_lag_3",
        "future_returns_lag_7",
        "future_returns_lag_14",
    ]

    rows = []
    for f in feature_side:
        for t in target_side:
            if f in corr.index and t in corr.columns:
                val = corr.loc[f, t]
                if pd.notna(val):
                    rows.append({"Feature": f, "Target_Horizon": t, "Correlation": float(val)})

    if rows:
        corr_long = pd.DataFrame(rows)

        # Human-readable labels
        feature_labels = {
            "Sentiment_Score": "Average daily sentiment",
            "sentiment_volatility": "How much sentiment jumps around",
            "lagged_sentiment": "Yesterday's sentiment",
        }
        horizon_labels = {
            "future_returns_lag_1": "1-day return",
            "future_returns_lag_3": "3-day return",
            "future_returns_lag_7": "7-day return",
            "future_returns_lag_14": "14-day return",
        }

        corr_long["Feature_label"] = corr_long["Feature"].map(feature_labels)
        corr_long["Target_label"] = corr_long["Target_Horizon"].map(horizon_labels)

        bar_fig = px.bar(
            corr_long,
            x="Feature_label",
            y="Correlation",
            color="Target_label",
            barmode="group",
            title="How different sentiment measures move with future returns",
        )
        bar_fig.update_layout(
            xaxis_title="Sentiment measure",
            yaxis=dict(
                title="Relationship strength (−1 = opposite, +1 = move together)",
                range=[-1, 1],
            ),
            legend_title_text="Future return",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        # Show values on top of bars
        bar_fig.update_traces(text=corr_long["Correlation"].round(2), textposition="outside")

        st.plotly_chart(bar_fig, use_container_width=True)

        # Auto-insights
        pairs_sorted = sorted(rows, key=lambda x: abs(x["Correlation"]), reverse=True)
        top_pos = next((p for p in pairs_sorted if p["Correlation"] > 0), None)
        top_neg = next((p for p in pairs_sorted if p["Correlation"] < 0), None)

        bullets = []
        if top_pos:
            bullets.append(
                f"**Strongest same-direction pattern:** "
                f"`{feature_labels[top_pos['Feature']]}` vs `{horizon_labels[top_pos['Target_Horizon']]}` "
                f"= **{top_pos['Correlation']:.2f}**"
            )
        if top_neg:
            bullets.append(
                f"**Strongest opposite-direction pattern:** "
                f"`{feature_labels[top_neg['Feature']]}` vs `{horizon_labels[top_neg['Target_Horizon']]}` "
                f"= **{top_neg['Correlation']:.2f}**"
            )
        if not bullets:
            bullets.append("No strong patterns found yet.")

        st.info(
            """
**How to read this chart (no chart experience needed):**

- Each group of bars on the **x-axis** is a type of *sentiment measure*:
  - **Average daily sentiment** → overall mood (more positive vs more negative).
  - **How much sentiment jumps around** → whether the mood is calm or very noisy.
  - **Yesterday's sentiment** → what people felt most recently.
- The **colours** show different future returns:
  - 1-day, 3-day, 7-day, and 14-day returns.
- The **height of each bar** tells you the relationship:
  - **Above 0** → when that sentiment goes up, that future return tends to go up too.
  - **Below 0** → when that sentiment goes up, that future return tends to go down.
  - Values closer to **+1 or −1** mean a stronger pattern; values near **0** mean little or no pattern.

This is about **historical patterns**, not guaranteed future moves.
"""
        )

       
    else:
        st.info("Not enough valid correlations to visualize yet.")
else:
    st.info("Not enough data to compute correlations.")

st.caption(
    f"Predictions target **{horizon}-day forward returns**; projected price = last close × (1 + predicted {horizon}-day return)."
)