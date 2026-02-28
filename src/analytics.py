"""
Analytics layer: time-series trends, brand aggregations, and KPI computation.

All functions are pure (take DataFrames, return DataFrames/dicts)
and output-format agnostic — saving is handled by the pipeline.
"""

import logging
from typing import Any

import pandas as pd
import numpy as np

from src.config import ROLLING_WINDOW_DAYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute top-level KPIs for the Overview dashboard tab.
    """
    total = len(df)
    sentiment_counts = df["predicted_sentiment"].value_counts()
    return_rate = df["return_flag"].mean()
    verified_pct = df["verified_purchase"].mean()

    kpis = {
        "total_reviews": total,
        "positive_count": int(sentiment_counts.get("positive", 0)),
        "negative_count": int(sentiment_counts.get("negative", 0)),
        "neutral_count": int(sentiment_counts.get("neutral", 0)),
        "positive_pct": round(sentiment_counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(sentiment_counts.get("negative", 0) / total * 100, 1),
        "neutral_pct": round(sentiment_counts.get("neutral", 0) / total * 100, 1),
        "mean_rating": round(df["rating"].mean(), 2),
        "return_rate_pct": round(return_rate * 100, 1),
        "verified_pct": round(verified_pct * 100, 1),
        "mean_sentiment_score": round(df["sentiment_score"].mean(), 4),
    }
    logger.info("KPIs computed: %s", kpis)
    return kpis


# ---------------------------------------------------------------------------
# Brand-level aggregations
# ---------------------------------------------------------------------------

def compute_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-brand: review count, avg rating, sentiment breakdown, return rate.
    """
    agg = (
        df.groupby("brand")
        .agg(
            review_count=("review_id", "count"),
            mean_rating=("rating", "mean"),
            mean_sentiment_score=("sentiment_score", "mean"),
            positive_count=("predicted_sentiment", lambda s: (s == "positive").sum()),
            negative_count=("predicted_sentiment", lambda s: (s == "negative").sum()),
            neutral_count=("predicted_sentiment", lambda s: (s == "neutral").sum()),
            return_rate=("return_flag", "mean"),
            mean_price=("price_paid", "mean"),
        )
        .reset_index()
    )
    agg["positive_pct"] = (agg["positive_count"] / agg["review_count"] * 100).round(1)
    agg["negative_pct"] = (agg["negative_count"] / agg["review_count"] * 100).round(1)
    agg["return_rate_pct"] = (agg["return_rate"] * 100).round(1)
    agg["mean_rating"] = agg["mean_rating"].round(2)
    agg["mean_sentiment_score"] = agg["mean_sentiment_score"].round(4)
    agg["mean_price"] = agg["mean_price"].round(2)
    return agg.sort_values("mean_rating", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-series trends
# ---------------------------------------------------------------------------

def compute_time_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sentiment counts and compute rolling 30-day averages.

    Returns a DataFrame with columns:
        review_date, sentiment, daily_count,
        rolling_avg (30-day rolling mean of daily_count)
    """
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])

    # Daily counts per sentiment
    daily = (
        df.groupby(["review_date", "predicted_sentiment"])
        .size()
        .reset_index(name="daily_count")
    )

    # Build a full date range so gaps don't break rolling windows
    all_dates = pd.date_range(df["review_date"].min(), df["review_date"].max(), freq="D")
    sentiments = ["positive", "negative", "neutral"]
    idx = pd.MultiIndex.from_product([all_dates, sentiments], names=["review_date", "predicted_sentiment"])
    full = (
        daily.set_index(["review_date", "predicted_sentiment"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    # Rolling average per sentiment
    full = full.sort_values(["predicted_sentiment", "review_date"])
    full["rolling_avg"] = (
        full.groupby("predicted_sentiment")["daily_count"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_DAYS, min_periods=1).mean().round(2))
    )

    # Also add daily positive_rate for the trend line
    daily_total = (
        full.groupby("review_date")["daily_count"]
        .sum()
        .rename("total_daily")
        .reset_index()
    )
    full = full.merge(daily_total, on="review_date", how="left")
    full["daily_rate"] = (full["daily_count"] / full["total_daily"].replace(0, np.nan)).round(4)

    logger.info("Time trends: %d rows, date range %s to %s",
                len(full), full["review_date"].min().date(), full["review_date"].max().date())
    return full


# ---------------------------------------------------------------------------
# Monthly summary for charting
# ---------------------------------------------------------------------------

def compute_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Coarser monthly aggregation for trend charts."""
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["month"] = df["review_date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["month", "predicted_sentiment"])
        .agg(count=("review_id", "count"), mean_score=("sentiment_score", "mean"))
        .reset_index()
    )
    monthly["mean_score"] = monthly["mean_score"].round(4)
    return monthly.sort_values(["month", "predicted_sentiment"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Country-level breakdown
# ---------------------------------------------------------------------------

def compute_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment distribution by country."""
    agg = (
        df.groupby("country")
        .agg(
            review_count=("review_id", "count"),
            mean_rating=("rating", "mean"),
            positive_count=("predicted_sentiment", lambda s: (s == "positive").sum()),
            negative_count=("predicted_sentiment", lambda s: (s == "negative").sum()),
        )
        .reset_index()
    )
    agg["positive_pct"] = (agg["positive_count"] / agg["review_count"] * 100).round(1)
    agg["negative_pct"] = (agg["negative_count"] / agg["review_count"] * 100).round(1)
    agg["mean_rating"] = agg["mean_rating"].round(2)
    return agg.sort_values("review_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rule-based recommendations engine
# ---------------------------------------------------------------------------

def generate_recommendations(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    brand_df: pd.DataFrame,
    kpis: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Produce deterministic, rule-based business insights.

    Returns a list of dicts: {category, insight, severity}
    severity: "high" | "medium" | "low"
    """
    recs: list[dict[str, str]] = []

    # --- Feature complaints ---
    for _, row in feature_df.iterrows():
        if row["negative_rate"] >= 0.50 and row["total_mentions"] >= 100:
            recs.append({
                "category": "Product Quality",
                "insight": (
                    f"'{row['feature'].title()}' has a {row['negative_rate']*100:.0f}% negative mention rate "
                    f"across {int(row['total_mentions'])} reviews. This is a high-priority quality concern."
                ),
                "severity": "high",
            })
        elif row["negative_rate"] >= 0.35 and row["total_mentions"] >= 50:
            recs.append({
                "category": "Product Quality",
                "insight": (
                    f"'{row['feature'].title()}' complaints appear in {row['negative_rate']*100:.0f}% of mentions. "
                    f"Consider a design review or firmware improvement."
                ),
                "severity": "medium",
            })

    # --- Brand return rate ---
    high_return = brand_df[brand_df["return_rate_pct"] > 12].sort_values("return_rate_pct", ascending=False)
    for _, row in high_return.iterrows():
        recs.append({
            "category": "Brand Performance",
            "insight": (
                f"{row['brand']} has a {row['return_rate_pct']:.1f}% return rate — "
                f"significantly above the 8% industry benchmark. Investigate root causes."
            ),
            "severity": "high" if row["return_rate_pct"] > 18 else "medium",
        })

    # --- Best performers (marketing opportunity) ---
    top_features = feature_df[feature_df["positive_rate"] >= 0.90].sort_values("positive_rate", ascending=False)
    for _, row in top_features.head(3).iterrows():
        recs.append({
            "category": "Marketing Opportunity",
            "insight": (
                f"'{row['feature'].title()}' has a {row['positive_rate']*100:.0f}% positive mention rate. "
                f"Emphasise this in product marketing and listing descriptions."
            ),
            "severity": "low",
        })

    # --- Overall sentiment ---
    if kpis["positive_pct"] > 60:
        recs.append({
            "category": "Overall Health",
            "insight": (
                f"Strong overall sentiment: {kpis['positive_pct']}% positive reviews. "
                f"Net Promoter opportunity — consider requesting reviews from happy customers."
            ),
            "severity": "low",
        })

    if kpis["negative_pct"] > 30:
        recs.append({
            "category": "Overall Health",
            "insight": (
                f"Elevated negative sentiment at {kpis['negative_pct']}%. "
                f"Prioritise product quality improvements before next marketing push."
            ),
            "severity": "high",
        })

    # --- Rating vs sentiment gap ---
    # If many 1-2 star reviews don't get flagged for return, there's a UX friction issue
    low_rated_no_return = df[(df["rating"] <= 2) & (~df["return_flag"])].shape[0]
    low_rated_total = (df["rating"] <= 2).sum()
    if low_rated_total > 0:
        friction_rate = low_rated_no_return / low_rated_total
        if friction_rate > 0.60:
            recs.append({
                "category": "Customer Experience",
                "insight": (
                    f"{friction_rate*100:.0f}% of 1–2 star reviewers did not return the product. "
                    f"Simplify the returns process and offer proactive support to dissatisfied customers."
                ),
                "severity": "medium",
            })

    # Deduplicate
    seen = set()
    unique_recs = []
    for r in recs:
        key = r["insight"][:60]
        if key not in seen:
            seen.add(key)
            unique_recs.append(r)

    logger.info("Generated %d recommendations", len(unique_recs))
    return unique_recs
