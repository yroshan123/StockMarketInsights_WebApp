import os, re, numpy as np, pandas as pd

# Optional backends
try:
    from transformers import BertTokenizer, BertForSequenceClassification
    from scipy.special import softmax as scipy_softmax
    _TRANSFORMERS = True
except Exception:
    _TRANSFORMERS = False

# NLTK VADER
try:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER = True
except Exception:
    _VADER = False


def _clean(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#\w+", " ", t)
    t = t.replace("&amp", " and ").replace("&gt", " greater than ")
    t = t.encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\$[a-zA-Z]+", " stock ", t)
    t = re.sub(r"s&p", " sp500 ", t)
    t = re.sub(r"p/e", " pe ratio ", t)
    t = re.sub(r"q&a", " question and answer ", t)
    return re.sub(r"\s+", " ", t).strip()


class SentimentEngine:
    """score(text) -> float in [-1, 1]."""
    def __init__(self, mode: str = "vader"):
        self.mode = mode
        self.vader = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._torch = None

        if self.mode == "bert":
            if not _TRANSFORMERS:
                self.mode = "vader"
            else:
                # import torch lazily
                import torch
                self._torch = torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._init_bert()

        if self.mode == "vader":
            if not _VADER:
                raise RuntimeError("VADER backend not available (install nltk and download vader_lexicon).")
            self.vader = SentimentIntensityAnalyzer()

    def _init_bert(self):
        name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.model = BertForSequenceClassification.from_pretrained(name).to(self.device)
        self.model.eval()

    def score(self, text: str) -> float:
        text = _clean(text)
        if self.mode == "vader":
            return float(self.vader.polarity_scores(text)["compound"])
        # BERT path
        with self._torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            logits = out.logits[0].detach().cpu().numpy()
            probs = scipy_softmax(logits)
            # 1..5 stars -> [-1, 1]
            stars = float(np.dot(np.arange(1, 6, dtype=np.float32), probs))
            return 2.0 * ((stars - 1.0) / 4.0)


def compute_or_load_sentiment(tweets_df: pd.DataFrame, cache_path: str, engine: SentimentEngine) -> pd.DataFrame:
    needed = {"Stock Name", "Date", "Tweet"}
    if not needed.issubset(tweets_df.columns):
        missing = needed - set(tweets_df.columns)
        raise ValueError(f"Tweets CSV missing columns: {missing}")

    df = tweets_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Tweet"])

    cache = None
    if os.path.exists(cache_path):
        try:
            cache = pd.read_parquet(cache_path)
        except Exception:
            cache = None

    if cache is not None and {"Tweet", "Sentiment_Score"}.issubset(cache.columns):
        df = df.merge(cache[["Tweet", "Sentiment_Score"]], on="Tweet", how="left")
        missing_mask = df["Sentiment_Score"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "Sentiment_Score"] = df.loc[missing_mask, "Tweet"].apply(engine.score)
            updated = pd.concat([cache, df.loc[missing_mask, ["Tweet", "Sentiment_Score"]]], ignore_index=True)
            updated = updated.drop_duplicates("Tweet", keep="last")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            updated.to_parquet(cache_path, index=False)
    else:
        df["Sentiment_Score"] = df["Tweet"].apply(engine.score)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df[["Tweet", "Sentiment_Score"]].drop_duplicates("Tweet").to_parquet(cache_path, index=False)
        except Exception:
            pass

    return df