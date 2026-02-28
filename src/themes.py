"""
Theme and keyword extraction module.

Responsibilities:
1. Extract top keywords overall and per sentiment (TF-IDF).
2. Compute feature vs sentiment correlations.
3. Build keyword summary and feature summary DataFrames ready for export.
"""

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_MAX_FEATURES, TOP_KEYWORDS_N

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword extraction via TF-IDF
# ---------------------------------------------------------------------------

def extract_keywords(
    texts: list[str],
    n: int = TOP_KEYWORDS_N,
    max_features: int = TFIDF_MAX_FEATURES,
    extra_stopwords: Optional[set[str]] = None,
) -> list[tuple[str, float]]:
    """
    Extract top-n keywords from a list of cleaned texts using TF-IDF.

    Returns:
        List of (keyword, mean_tfidf_score) tuples, sorted descending.
    """
    if not texts:
        return []

    stop_words = "english"
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        stop_words=stop_words,
        min_df=2,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Empty vocabulary after stopword removal
        return []

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    ranked = sorted(zip(feature_names, mean_scores), key=lambda x: x[1], reverse=True)

    if extra_stopwords:
        ranked = [(w, s) for w, s in ranked if w not in extra_stopwords]

    return ranked[:n]


def build_keyword_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame of top keywords broken down by sentiment.

    Columns: keyword, sentiment, tfidf_score, rank
    """
    records = []
    sentiments = ["positive", "negative", "neutral", "overall"]

    for sentiment in sentiments:
        if sentiment == "overall":
            texts = df["clean_tfidf"].dropna().tolist()
        else:
            mask = df["predicted_sentiment"] == sentiment
            texts = df.loc[mask, "clean_tfidf"].dropna().tolist()

        keywords = extract_keywords(texts, n=TOP_KEYWORDS_N)
        for rank, (word, score) in enumerate(keywords, start=1):
            records.append({
                "keyword": word,
                "sentiment": sentiment,
                "tfidf_score": round(float(score), 6),
                "rank": rank,
            })

    result = pd.DataFrame(records)
    logger.info("Keyword summary: %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature vs sentiment correlation
# ---------------------------------------------------------------------------

def _explode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the semicolon-separated features_mentioned column."""
    df = df.copy()
    df["feature"] = df["features_mentioned"].fillna("").str.split(";")
    exploded = df.explode("feature")
    exploded["feature"] = exploded["feature"].str.strip()
    return exploded[exploded["feature"] != ""]


def build_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, per feature:
        - total mentions
        - positive / negative / neutral mention counts
        - negative_rate (negative mentions / total mentions)
        - mean_rating
        - mean_sentiment_score

    Returns a DataFrame sorted by negative_rate descending.
    """
    exploded = _explode_features(df)

    agg = (
        exploded.groupby("feature")
        .agg(
            total_mentions=("review_id", "count"),
            positive_mentions=("predicted_sentiment", lambda s: (s == "positive").sum()),
            negative_mentions=("predicted_sentiment", lambda s: (s == "negative").sum()),
            neutral_mentions=("predicted_sentiment", lambda s: (s == "neutral").sum()),
            mean_rating=("rating", "mean"),
            mean_sentiment_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    agg["negative_rate"] = (agg["negative_mentions"] / agg["total_mentions"]).round(4)
    agg["positive_rate"] = (agg["positive_mentions"] / agg["total_mentions"]).round(4)
    agg["mean_rating"] = agg["mean_rating"].round(2)
    agg["mean_sentiment_score"] = agg["mean_sentiment_score"].round(4)

    # Only include features with enough mentions to be meaningful
    agg = agg[agg["total_mentions"] >= 10].copy()
    agg = agg.sort_values("negative_rate", ascending=False).reset_index(drop=True)

    logger.info("Feature summary: %d features extracted", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Complaint-driving features (for dashboard recommendations)
# ---------------------------------------------------------------------------

def top_complaint_drivers(feature_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the top-n features most strongly associated with negative reviews,
    weighted by both negative_rate and total_mentions volume.
    """
    df = feature_df.copy()
    # Combined score: high negative rate + high volume
    df["complaint_score"] = df["negative_rate"] * np.log1p(df["total_mentions"])
    return (
        df.nlargest(n, "complaint_score")
        .reset_index(drop=True)[
            ["feature", "total_mentions", "negative_mentions", "negative_rate", "complaint_score"]
        ]
    )
