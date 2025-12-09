import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DEFAULT_STOCKS_CSV = os.path.join("data", "Stock_yfinance_data.csv")
DEFAULT_TWEETS_CSV = os.path.join("data", "stock_tweets.csv")
DEFAULT_OUTPUT_DIR = os.path.join("output")
DEFAULT_SENTIMENT_CACHE = os.path.join("cache", "tweet_sentiment_cache.parquet")

os.makedirs(os.path.dirname(DEFAULT_SENTIMENT_CACHE), exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)