"""
Text preprocessing pipeline.

Cleans and normalises raw review text before NLP analysis.
Design principle: each step is a pure function so the pipeline
is easy to unit-test and extend.
"""

import re
import string
import logging
from typing import Optional

import pandas as pd
import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data on first import (idempotent)
def _ensure_nltk_data() -> None:
    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

from nltk.corpus import stopwords  # noqa: E402 (must be after download)

ENGLISH_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))

# Contractions we want to expand so VADER scores them correctly
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
    flags=re.IGNORECASE,
)
_CONTRACTIONS_LOWER = {k.lower(): v for k, v in _CONTRACTIONS.items()}


# ---------------------------------------------------------------------------
# Individual transformation functions
# ---------------------------------------------------------------------------

def expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(
        lambda m: _CONTRACTIONS_LOWER[m.group(0).lower()], text
    )


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_special_chars(text: str, keep_punct: bool = False) -> str:
    """Remove non-alphanumeric characters. Optionally retain sentence punctuation."""
    if keep_punct:
        # Keep letters, digits, spaces, and basic punctuation
        return re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", " ", text)
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in ENGLISH_STOPWORDS)


# ---------------------------------------------------------------------------
# Composite pipelines
# ---------------------------------------------------------------------------

def clean_for_vader(text: str) -> str:
    """
    Lightweight cleaning for VADER sentiment analysis.

    VADER is rule-based and uses capitalisation, punctuation, and
    intensifiers — so we preserve those signals.
    """
    text = remove_urls(text)
    text = remove_html(text)
    text = expand_contractions(text)
    text = remove_special_chars(text, keep_punct=True)
    text = normalise_whitespace(text)
    return text


def clean_for_tfidf(text: str) -> str:
    """
    Aggressive cleaning for keyword/TF-IDF extraction.

    Removes stopwords and all punctuation; lowercases.
    """
    text = remove_urls(text)
    text = remove_html(text)
    text = expand_contractions(text)
    text = to_lowercase(text)
    text = remove_special_chars(text, keep_punct=False)
    text = normalise_whitespace(text)
    text = remove_stopwords(text)
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both cleaning pipelines to a reviews DataFrame in-place (returns copy).

    Expects a 'review_text' column.
    Returns the same DataFrame with two additional columns:
        - clean_vader : text ready for VADER
        - clean_tfidf : text ready for TF-IDF
    """
    if "review_text" not in df.columns:
        raise ValueError("DataFrame must contain a 'review_text' column.")

    logger.info("Preprocessing %d reviews …", len(df))
    df = df.copy()
    df["clean_vader"] = df["review_text"].fillna("").apply(clean_for_vader)
    df["clean_tfidf"] = df["review_text"].fillna("").apply(clean_for_tfidf)
    logger.info("Preprocessing complete.")
    return df


def parse_features(features_str: Optional[str]) -> list[str]:
    """Parse semicolon-separated feature string into a list."""
    if not features_str or pd.isna(features_str):
        return []
    return [f.strip() for f in str(features_str).split(";") if f.strip()]
