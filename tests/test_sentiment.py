"""
Tests for the sentiment classification module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentiment import (
    score_with_vader,
    _vader_label,
    score_dataframe,
    evaluate_predictions,
)
from src.preprocessing import preprocess_dataframe


# ---------------------------------------------------------------------------
# VADER label mapping
# ---------------------------------------------------------------------------

class TestVaderLabel:
    def test_positive_threshold(self):
        assert _vader_label(0.06) == "positive"

    def test_negative_threshold(self):
        assert _vader_label(-0.06) == "negative"

    def test_neutral_band(self):
        assert _vader_label(0.0) == "neutral"
        assert _vader_label(0.04) == "neutral"
        assert _vader_label(-0.04) == "neutral"

    def test_boundary_positive(self):
        assert _vader_label(0.05) == "positive"

    def test_boundary_negative(self):
        assert _vader_label(-0.05) == "negative"


# ---------------------------------------------------------------------------
# VADER scoring
# ---------------------------------------------------------------------------

class TestScoreWithVader:
    TEXTS = [
        "This product is absolutely amazing and I love it!",
        "Terrible quality. Broke after one day. Complete waste of money.",
        "It arrived on time and seems fine.",
    ]

    def test_returns_correct_length(self):
        scores, labels = score_with_vader(self.TEXTS)
        assert len(scores) == 3
        assert len(labels) == 3

    def test_positive_text_scores_positive(self):
        scores, labels = score_with_vader(["Incredible sound quality! Best earbuds I've ever used!"])
        assert labels[0] == "positive"
        assert scores[0] > 0

    def test_negative_text_scores_negative(self):
        scores, labels = score_with_vader(["Absolutely terrible. Broke immediately. Total garbage."])
        assert labels[0] == "negative"
        assert scores[0] < 0

    def test_score_range(self):
        scores, _ = score_with_vader(self.TEXTS)
        for s in scores:
            assert -1.0 <= s <= 1.0

    def test_empty_input(self):
        scores, labels = score_with_vader([])
        assert scores == []
        assert labels == []

    def test_labels_are_valid(self):
        _, labels = score_with_vader(self.TEXTS)
        valid = {"positive", "negative", "neutral"}
        for label in labels:
            assert label in valid


# ---------------------------------------------------------------------------
# DataFrame scoring
# ---------------------------------------------------------------------------

class TestScoreDataframe:
    @pytest.fixture
    def sample_df(self):
        df = pd.DataFrame({
            "review_id": ["R1", "R2", "R3"],
            "review_text": [
                "These earbuds are fantastic! Great sound!",
                "Worst earbuds ever. Complete garbage.",
                "They work, nothing special.",
            ],
            "ground_truth_sentiment": ["positive", "negative", "neutral"],
        })
        return preprocess_dataframe(df)

    def test_adds_score_and_label(self, sample_df):
        out = score_dataframe(sample_df)
        assert "sentiment_score" in out.columns
        assert "predicted_sentiment" in out.columns

    def test_correct_positive(self, sample_df):
        out = score_dataframe(sample_df)
        assert out.loc[0, "predicted_sentiment"] == "positive"

    def test_correct_negative(self, sample_df):
        out = score_dataframe(sample_df)
        assert out.loc[1, "predicted_sentiment"] == "negative"

    def test_does_not_mutate_input(self, sample_df):
        original_cols = set(sample_df.columns)
        _ = score_dataframe(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_raises_without_clean_vader(self):
        df = pd.DataFrame({"review_text": ["good product"]})
        with pytest.raises(ValueError, match="clean_vader"):
            score_dataframe(df)

    def test_unknown_backend_falls_back_to_vader(self, sample_df):
        # Should not raise, just warn
        out = score_dataframe(sample_df, backend="nonexistent")
        assert "predicted_sentiment" in out.columns


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:
    @pytest.fixture
    def eval_df(self):
        return pd.DataFrame({
            "ground_truth_sentiment": ["positive", "negative", "neutral", "positive", "negative"],
            "predicted_sentiment":    ["positive", "negative", "positive", "positive", "positive"],
        })

    def test_returns_dict_with_required_keys(self, eval_df):
        result = evaluate_predictions(eval_df)
        assert "accuracy" in result
        assert "classification_report" in result
        assert "confusion_matrix" in result

    def test_accuracy_in_range(self, eval_df):
        result = evaluate_predictions(eval_df)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, eval_df):
        result = evaluate_predictions(eval_df)
        cm = result["confusion_matrix"]["matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_raises_missing_columns(self):
        df = pd.DataFrame({"predicted_sentiment": ["positive"]})
        with pytest.raises(ValueError):
            evaluate_predictions(df)

    def test_perfect_accuracy(self):
        df = pd.DataFrame({
            "ground_truth_sentiment": ["positive", "negative", "neutral"],
            "predicted_sentiment":    ["positive", "negative", "neutral"],
        })
        result = evaluate_predictions(df)
        assert result["accuracy"] == pytest.approx(1.0)
