"""
Tests for the preprocessing pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    expand_contractions,
    remove_urls,
    remove_html,
    remove_special_chars,
    normalise_whitespace,
    to_lowercase,
    remove_stopwords,
    clean_for_vader,
    clean_for_tfidf,
    preprocess_dataframe,
    parse_features,
)


# ---------------------------------------------------------------------------
# Unit tests for individual transforms
# ---------------------------------------------------------------------------

class TestExpandContractions:
    def test_wont(self):
        assert "will not" in expand_contractions("I won't buy this")

    def test_cant(self):
        assert "cannot" in expand_contractions("I can't believe it")

    def test_no_change_on_clean(self):
        text = "This is great"
        assert expand_contractions(text) == text


class TestRemoveUrls:
    def test_removes_http(self):
        result = remove_urls("Visit https://example.com for details")
        assert "https" not in result

    def test_removes_www(self):
        result = remove_urls("Go to www.example.com now")
        assert "www" not in result

    def test_no_false_positive(self):
        text = "Battery life is amazing"
        assert remove_urls(text) == text


class TestRemoveHtml:
    def test_strips_tags(self):
        result = remove_html("<b>Great</b> product <br/>")
        assert "<" not in result and ">" not in result
        assert "Great" in result

    def test_no_change_on_plain(self):
        text = "Solid build quality"
        assert remove_html(text) == text


class TestNormaliseWhitespace:
    def test_collapses_spaces(self):
        result = normalise_whitespace("too   many    spaces")
        assert "  " not in result

    def test_strips_ends(self):
        assert normalise_whitespace("  hello  ") == "hello"


class TestToLowercase:
    def test_lowercases(self):
        assert to_lowercase("AMAZING Battery Life") == "amazing battery life"


class TestRemoveStopwords:
    def test_removes_the(self):
        result = remove_stopwords("the battery is great")
        assert "the" not in result.split()

    def test_keeps_content_words(self):
        result = remove_stopwords("battery life excellent")
        assert "battery" in result
        assert "excellent" in result


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestCleanForVader:
    def test_preserves_exclamation(self):
        text = "This is AMAZING!!!"
        result = clean_for_vader(text)
        assert "!" in result

    def test_removes_url(self):
        text = "Check https://example.com — great product"
        result = clean_for_vader(text)
        assert "https" not in result

    def test_expands_contractions(self):
        result = clean_for_vader("Won't buy again")
        assert "will not" in result.lower() or "not" in result.lower()


class TestCleanForTfidf:
    def test_lowercase_output(self):
        result = clean_for_tfidf("GREAT Battery LIFE")
        assert result == result.lower()

    def test_no_punctuation(self):
        result = clean_for_tfidf("Amazing! Quality—premium.")
        assert "!" not in result
        assert "." not in result

    def test_no_stopwords(self):
        result = clean_for_tfidf("the battery is very good")
        tokens = result.split()
        assert "the" not in tokens
        assert "is" not in tokens


class TestPreprocessDataframe:
    def test_adds_clean_columns(self):
        df = pd.DataFrame({"review_text": ["Great battery!", "Terrible connectivity."]})
        out = preprocess_dataframe(df)
        assert "clean_vader" in out.columns
        assert "clean_tfidf" in out.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"review_text": ["Good product"]})
        original_cols = set(df.columns)
        _ = preprocess_dataframe(df)
        assert set(df.columns) == original_cols

    def test_handles_missing_text(self):
        df = pd.DataFrame({"review_text": [None, "", "Good"]})
        out = preprocess_dataframe(df)
        assert out["clean_vader"].notna().all()

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"text": ["some text"]})
        with pytest.raises(ValueError, match="review_text"):
            preprocess_dataframe(df)


class TestParseFeatures:
    def test_parses_semicolon(self):
        result = parse_features("battery life;sound quality;comfort")
        assert result == ["battery life", "sound quality", "comfort"]

    def test_handles_none(self):
        assert parse_features(None) == []

    def test_handles_empty(self):
        assert parse_features("") == []

    def test_strips_whitespace(self):
        result = parse_features(" battery ; comfort ")
        assert result == ["battery", "comfort"]
