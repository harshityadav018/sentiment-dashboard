"""
Central configuration for the sentiment analysis pipeline.

All paths are resolved relative to the project root so the project
is portable regardless of where it is cloned.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Input / output file paths
# ---------------------------------------------------------------------------
RAW_REVIEWS_CSV = DATA_DIR / "reviews.csv"

SCORED_REVIEWS_PARQUET = OUTPUTS_DIR / "scored_reviews.parquet"
FEATURE_SUMMARY_CSV = OUTPUTS_DIR / "feature_summary.csv"
KEYWORD_SUMMARY_CSV = OUTPUTS_DIR / "keyword_summary.csv"
TIME_TRENDS_CSV = OUTPUTS_DIR / "time_trends.csv"
METRICS_JSON = OUTPUTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# Sentiment backend: "vader" or "distilbert"
# Pass --model distilbert to pipeline.py to override
SENTIMENT_BACKEND: str = "vader"

# VADER compound score thresholds
VADER_POS_THRESHOLD: float = 0.05
VADER_NEG_THRESHOLD: float = -0.05

# TF-IDF keyword extraction
TFIDF_MAX_FEATURES: int = 500
TOP_KEYWORDS_N: int = 20

# Time-series aggregation window
ROLLING_WINDOW_DAYS: int = 30

# ---------------------------------------------------------------------------
# Dashboard settings
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "Customer Sentiment Dashboard — Wireless Earbuds"
PAGE_ICON: str = "🎧"
CHART_HEIGHT: int = 400

SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

BRAND_ORDER = [
    "AirPods Pro", "BeatsFlex", "JabRa", "SonyX", "SoundCore", "UrbanBeats"
]
