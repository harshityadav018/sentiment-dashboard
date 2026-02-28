"""
Tests for synthetic data generation.

Validates schema, row counts, value ranges, and feature correlations.
"""

import sys
import csv
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_reviews import generate_reviews, save_reviews


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reviews_small():
    """Generate a small reproducible dataset for fast tests."""
    return generate_reviews(n=500)


@pytest.fixture(scope="module")
def reviews_medium():
    return generate_reviews(n=2000)


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

def test_row_count_small(reviews_small):
    assert len(reviews_small) == 500


def test_row_count_medium(reviews_medium):
    assert len(reviews_medium) == 2000


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "review_id", "brand", "rating", "review_text", "review_date",
    "verified_purchase", "country", "price_paid", "return_flag",
    "features_mentioned", "ground_truth_sentiment",
}


def test_schema(reviews_small):
    assert set(reviews_small[0].keys()) == EXPECTED_COLUMNS


def test_no_missing_text(reviews_small):
    for r in reviews_small:
        assert r["review_text"].strip(), f"Empty review_text for {r['review_id']}"


def test_review_ids_unique(reviews_small):
    ids = [r["review_id"] for r in reviews_small]
    assert len(ids) == len(set(ids)), "Duplicate review_ids found"


# ---------------------------------------------------------------------------
# Value ranges and types
# ---------------------------------------------------------------------------

def test_ratings_in_range(reviews_small):
    for r in reviews_small:
        assert 1 <= int(r["rating"]) <= 5, f"Rating out of range: {r['rating']}"


def test_sentiments_valid(reviews_small):
    valid = {"positive", "negative", "neutral"}
    for r in reviews_small:
        assert r["ground_truth_sentiment"] in valid


def test_countries_valid(reviews_small):
    valid = {"US", "UK", "CA", "IN", "AU"}
    for r in reviews_small:
        assert r["country"] in valid


def test_prices_positive(reviews_small):
    for r in reviews_small:
        assert float(r["price_paid"]) > 0


def test_verified_purchase_boolean(reviews_small):
    for r in reviews_small:
        assert r["verified_purchase"] in (True, False, "True", "False")


# ---------------------------------------------------------------------------
# Sentiment distribution
# ---------------------------------------------------------------------------

def test_sentiment_distribution(reviews_medium):
    labels = [r["ground_truth_sentiment"] for r in reviews_medium]
    pos = labels.count("positive") / len(labels)
    neg = labels.count("negative") / len(labels)
    # Rough bounds: positive ~55%, negative ~25%
    assert 0.45 <= pos <= 0.65, f"Unexpected positive rate: {pos:.2f}"
    assert 0.15 <= neg <= 0.40, f"Unexpected negative rate: {neg:.2f}"


# ---------------------------------------------------------------------------
# Sentiment-rating correlation
# ---------------------------------------------------------------------------

def test_positive_reviews_higher_rating(reviews_medium):
    pos_ratings = [int(r["rating"]) for r in reviews_medium if r["ground_truth_sentiment"] == "positive"]
    neg_ratings = [int(r["rating"]) for r in reviews_medium if r["ground_truth_sentiment"] == "negative"]
    avg_pos = sum(pos_ratings) / len(pos_ratings)
    avg_neg = sum(neg_ratings) / len(neg_ratings)
    assert avg_pos > avg_neg + 0.5, (
        f"Positive avg rating ({avg_pos:.2f}) should be substantially higher than "
        f"negative ({avg_neg:.2f})"
    )


# ---------------------------------------------------------------------------
# Feature mentions
# ---------------------------------------------------------------------------

def test_features_not_empty(reviews_small):
    for r in reviews_small:
        features = r["features_mentioned"]
        assert features.strip(), f"Empty features_mentioned for {r['review_id']}"


def test_features_no_duplicates(reviews_small):
    for r in reviews_small:
        parts = [f.strip() for f in r["features_mentioned"].split(";")]
        assert len(parts) == len(set(parts)), (
            f"Duplicate features in {r['review_id']}: {r['features_mentioned']}"
        )


# ---------------------------------------------------------------------------
# CSV save/load round-trip
# ---------------------------------------------------------------------------

def test_csv_roundtrip(reviews_small):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_reviews.csv"
        save_reviews(reviews_small, path)

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            loaded = list(reader)

        assert len(loaded) == len(reviews_small)
        assert set(loaded[0].keys()) == EXPECTED_COLUMNS
