"""
Sentiment classification module.

Supports two backends:
  - VADER (default): rule-based, no GPU required, fast.
  - DistilBERT: transformer-based, enabled via --model distilbert flag.

Both backends produce a unified output schema:
    sentiment_score  : float  (compound for VADER, logit diff for DistilBERT)
    predicted_sentiment : str  ("positive" | "negative" | "neutral")
"""

import logging
from typing import Literal

import pandas as pd

from src.config import (
    VADER_POS_THRESHOLD,
    VADER_NEG_THRESHOLD,
    SENTIMENT_BACKEND,
)

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# VADER backend
# ---------------------------------------------------------------------------

def _load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def _vader_label(compound: float) -> SentimentLabel:
    if compound >= VADER_POS_THRESHOLD:
        return "positive"
    if compound <= VADER_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score_with_vader(texts: list[str]) -> tuple[list[float], list[SentimentLabel]]:
    """
    Score a list of pre-cleaned texts with VADER.

    Returns:
        scores  : compound scores in [-1, 1]
        labels  : sentiment labels
    """
    analyzer = _load_vader()
    scores: list[float] = []
    labels: list[SentimentLabel] = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"]
        scores.append(round(compound, 4))
        labels.append(_vader_label(compound))
    return scores, labels


# ---------------------------------------------------------------------------
# DistilBERT backend (optional, behind flag)
# ---------------------------------------------------------------------------

def score_with_distilbert(texts: list[str]) -> tuple[list[float], list[SentimentLabel]]:
    """
    Score texts with distilbert-base-uncased-finetuned-sst-2-english.

    Requires: pip install transformers torch
    Falls back to VADER if transformers is not installed.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.warning(
            "transformers not installed — falling back to VADER. "
            "Run: pip install transformers torch"
        )
        return score_with_vader(texts)

    logger.info("Loading DistilBERT sentiment pipeline …")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    scores: list[float] = []
    labels: list[SentimentLabel] = []

    # Batch to avoid OOM on large datasets
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = pipe(batch)
        for res in results:
            raw_label = res["label"].lower()  # "positive" or "negative"
            confidence = float(res["score"])
            # Map to our 3-class schema: keep neutral band via confidence
            if raw_label == "positive":
                score = confidence
                label: SentimentLabel = "positive" if confidence > 0.6 else "neutral"
            else:
                score = -confidence
                label = "negative" if confidence > 0.6 else "neutral"
            scores.append(round(score, 4))
            labels.append(label)

    return scores, labels


# ---------------------------------------------------------------------------
# Unified scorer
# ---------------------------------------------------------------------------

def score_dataframe(
    df: pd.DataFrame,
    backend: str = SENTIMENT_BACKEND,
) -> pd.DataFrame:
    """
    Add sentiment_score and predicted_sentiment columns to the reviews DataFrame.

    Args:
        df      : DataFrame with 'clean_vader' (and optionally 'clean_tfidf') columns
        backend : "vader" or "distilbert"

    Returns:
        DataFrame with two new columns appended.
    """
    if "clean_vader" not in df.columns:
        raise ValueError("DataFrame must have 'clean_vader' column. Run preprocessing first.")

    texts = df["clean_vader"].tolist()
    logger.info("Scoring %d reviews with backend='%s' …", len(df), backend)

    if backend == "distilbert":
        scores, labels = score_with_distilbert(texts)
    else:
        if backend != "vader":
            logger.warning("Unknown backend '%s'; falling back to VADER.", backend)
        scores, labels = score_with_vader(texts)

    df = df.copy()
    df["sentiment_score"] = scores
    df["predicted_sentiment"] = labels
    logger.info("Scoring complete. Distribution: %s", df["predicted_sentiment"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(df: pd.DataFrame) -> dict:
    """
    Compare predicted_sentiment against ground_truth_sentiment.

    Returns a dict with classification_report (as dict) and confusion_matrix.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    if "ground_truth_sentiment" not in df.columns or "predicted_sentiment" not in df.columns:
        raise ValueError("DataFrame must have 'ground_truth_sentiment' and 'predicted_sentiment'.")

    y_true = df["ground_truth_sentiment"]
    y_pred = df["predicted_sentiment"]
    labels = ["positive", "negative", "neutral"]

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
        "accuracy": report["accuracy"],
    }
