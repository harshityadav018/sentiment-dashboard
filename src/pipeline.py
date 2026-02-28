"""
Pipeline orchestrator.

Runs the full end-to-end pipeline:
    1. Load raw reviews CSV
    2. Preprocess text
    3. Score sentiment
    4. Extract keywords and feature correlations
    5. Compute analytics (trends, brand summary, KPIs)
    6. Evaluate against ground truth
    7. Save all outputs

Usage:
    python -m src.pipeline                    # uses VADER (default)
    python -m src.pipeline --model distilbert # uses DistilBERT
    python -m src.pipeline --debug            # verbose logging
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.preprocessing import preprocess_dataframe
from src.sentiment import score_dataframe, evaluate_predictions
from src.themes import build_keyword_summary, build_feature_summary
from src.analytics import (
    compute_kpis,
    compute_brand_summary,
    compute_time_trends,
    compute_monthly_trends,
    compute_country_summary,
    generate_recommendations,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Reviews CSV not found at {path}. "
            "Run `python data/generate_reviews.py` first."
        )
    df = pd.read_csv(path, dtype={"review_id": str})
    logger.info("Loaded %d reviews from %s", len(df), path)

    # Parse boolean columns (CSV stores them as strings)
    for col in ("verified_purchase", "return_flag"):
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    return df


def run_pipeline(backend: str = config.SENTIMENT_BACKEND) -> dict:
    """
    Execute the full pipeline and return a summary dict.

    Args:
        backend : "vader" or "distilbert"

    Returns:
        A dict with paths to all output files.
    """
    t0 = time.time()
    logger.info("=== Sentiment Analysis Pipeline START ===")
    logger.info("Backend: %s", backend)

    # 1. Load
    df = load_data(config.RAW_REVIEWS_CSV)

    # 2. Preprocess
    df = preprocess_dataframe(df)

    # 3. Sentiment scoring
    df = score_dataframe(df, backend=backend)

    # 4. Theme extraction
    keyword_df = build_keyword_summary(df)
    feature_df = build_feature_summary(df)

    # 5. Analytics
    kpis = compute_kpis(df)
    brand_df = compute_brand_summary(df)
    time_trends_df = compute_time_trends(df)
    monthly_df = compute_monthly_trends(df)
    country_df = compute_country_summary(df)

    # 6. Recommendations
    recommendations = generate_recommendations(df, feature_df, brand_df, kpis)

    # 7. Evaluate
    metrics = evaluate_predictions(df)
    metrics["kpis"] = kpis
    metrics["brand_summary"] = brand_df.to_dict(orient="records")
    metrics["country_summary"] = country_df.to_dict(orient="records")
    metrics["recommendations"] = recommendations

    elapsed = round(time.time() - t0, 2)
    metrics["pipeline_elapsed_seconds"] = elapsed
    metrics["sentiment_backend"] = backend
    metrics["total_reviews"] = len(df)

    # 8. Save outputs
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Scored reviews → Parquet
    df.to_parquet(config.SCORED_REVIEWS_PARQUET, index=False)
    logger.info("Saved scored reviews → %s", config.SCORED_REVIEWS_PARQUET)

    # Feature summary → CSV
    feature_df.to_csv(config.FEATURE_SUMMARY_CSV, index=False)
    logger.info("Saved feature summary → %s", config.FEATURE_SUMMARY_CSV)

    # Keyword summary → CSV
    keyword_df.to_csv(config.KEYWORD_SUMMARY_CSV, index=False)
    logger.info("Saved keyword summary → %s", config.KEYWORD_SUMMARY_CSV)

    # Time trends (daily + monthly) → CSV
    # Merge both into one file with a 'resolution' column
    daily_df = time_trends_df.copy()
    daily_df["resolution"] = "daily"
    monthly_df["resolution"] = "monthly"
    monthly_df = monthly_df.rename(
        columns={"month": "review_date", "count": "daily_count", "mean_score": "daily_rate"}
    )
    combined_trends = pd.concat(
        [daily_df[["review_date", "predicted_sentiment", "daily_count", "rolling_avg", "resolution"]],
         monthly_df[["review_date", "predicted_sentiment", "daily_count", "resolution"]]],
        ignore_index=True,
    )
    combined_trends.to_csv(config.TIME_TRENDS_CSV, index=False)
    logger.info("Saved time trends → %s", config.TIME_TRENDS_CSV)

    # Metrics → JSON
    with open(config.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics → %s", config.METRICS_JSON)

    logger.info("=== Pipeline COMPLETE in %.2fs ===", elapsed)
    logger.info("Accuracy: %.3f | Positive: %d%% | Negative: %d%%",
                metrics["accuracy"], kpis["positive_pct"], kpis["negative_pct"])

    return {
        "scored_reviews": str(config.SCORED_REVIEWS_PARQUET),
        "feature_summary": str(config.FEATURE_SUMMARY_CSV),
        "keyword_summary": str(config.KEYWORD_SUMMARY_CSV),
        "time_trends": str(config.TIME_TRENDS_CSV),
        "metrics": str(config.METRICS_JSON),
        "accuracy": metrics["accuracy"],
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline")
    parser.add_argument(
        "--model",
        choices=["vader", "distilbert"],
        default=config.SENTIMENT_BACKEND,
        help="Sentiment backend (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.debug)
    result = run_pipeline(backend=args.model)
    print("\nOutputs:")
    for k, v in result.items():
        print(f"  {k}: {v}")
