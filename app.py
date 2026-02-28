"""
Customer Sentiment Analysis Dashboard — Wireless Earbuds
Streamlit + Plotly interactive dashboard.

Run: streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = config.SENTIMENT_COLORS
CHART_H = config.CHART_HEIGHT


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading pipeline outputs …")
def load_scored_reviews() -> pd.DataFrame:
    df = pd.read_parquet(config.SCORED_REVIEWS_PARQUET)
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


@st.cache_data(show_spinner=False)
def load_feature_summary() -> pd.DataFrame:
    return pd.read_csv(config.FEATURE_SUMMARY_CSV)


@st.cache_data(show_spinner=False)
def load_keyword_summary() -> pd.DataFrame:
    return pd.read_csv(config.KEYWORD_SUMMARY_CSV)


@st.cache_data(show_spinner=False)
def load_time_trends() -> pd.DataFrame:
    df = pd.read_csv(config.TIME_TRENDS_CSV)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(config.METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _check_outputs() -> bool:
    required = [
        config.SCORED_REVIEWS_PARQUET,
        config.FEATURE_SUMMARY_CSV,
        config.KEYWORD_SUMMARY_CSV,
        config.TIME_TRENDS_CSV,
        config.METRICS_JSON,
    ]
    return all(Path(p).exists() for p in required)


# ---------------------------------------------------------------------------
# Helper: KPI card row
# ---------------------------------------------------------------------------

def _kpi_row(kpis: dict) -> None:
    cols = st.columns(6)
    cards = [
        ("Total Reviews", f"{kpis['total_reviews']:,}", ""),
        ("Positive", f"{kpis['positive_pct']}%", f"{kpis['positive_count']:,} reviews"),
        ("Negative", f"{kpis['negative_pct']}%", f"{kpis['negative_count']:,} reviews"),
        ("Avg Rating", f"{kpis['mean_rating']} ⭐", "out of 5"),
        ("Return Rate", f"{kpis['return_rate_pct']}%", ""),
        ("Verified", f"{kpis['verified_pct']}%", "verified purchases"),
    ]
    for col, (label, value, subtext) in zip(cols, cards):
        with col:
            st.metric(label=label, value=value, help=subtext if subtext else None)


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def tab_overview(df: pd.DataFrame, kpis: dict, metrics: dict) -> None:
    st.header("Overview")
    _kpi_row(kpis)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Sentiment distribution donut
        sent_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [kpis["positive_count"], kpis["negative_count"], kpis["neutral_count"]],
        })
        fig = px.pie(
            sent_data, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": COLORS["positive"],
                "Negative": COLORS["negative"],
                "Neutral": COLORS["neutral"],
            },
            hole=0.45,
            title="Sentiment Distribution",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=CHART_H, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rating histogram
        fig = px.histogram(
            df, x="rating", nbins=5,
            color_discrete_sequence=["#3498db"],
            title="Rating Distribution (1–5 Stars)",
            labels={"rating": "Star Rating", "count": "Reviews"},
        )
        fig.update_layout(height=CHART_H, bargap=0.1)
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Accuracy Evaluation vs Ground Truth")

    report = metrics.get("classification_report", {})
    if report:
        rows = []
        for label in ["positive", "negative", "neutral"]:
            r = report.get(label, {})
            rows.append({
                "Sentiment": label.title(),
                "Precision": round(r.get("precision", 0), 3),
                "Recall": round(r.get("recall", 0), 3),
                "F1-Score": round(r.get("f1-score", 0), 3),
                "Support": int(r.get("support", 0)),
            })
        eval_df = pd.DataFrame(rows)
        st.dataframe(eval_df, hide_index=True, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Confusion matrix heatmap
        cm_data = metrics.get("confusion_matrix", {})
        if cm_data:
            labels = cm_data["labels"]
            matrix = cm_data["matrix"]
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"Pred {l.title()}" for l in labels],
                    y=[f"True {l.title()}" for l in labels],
                    colorscale="Blues",
                    text=matrix,
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig.update_layout(title="Confusion Matrix", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Brand overview bar
        brand_data = pd.DataFrame(metrics.get("brand_summary", []))
        if not brand_data.empty:
            fig = px.bar(
                brand_data.sort_values("mean_rating"),
                x="mean_rating", y="brand",
                orientation="h",
                color="positive_pct",
                color_continuous_scale="RdYlGn",
                title="Brand Average Rating",
                labels={"mean_rating": "Avg Rating", "brand": "Brand"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Trends
# ---------------------------------------------------------------------------

def tab_trends(trends_df: pd.DataFrame) -> None:
    st.header("Sentiment Trends Over Time")

    # Daily trend
    daily = trends_df[trends_df["resolution"] == "daily"].copy()
    daily["review_date"] = pd.to_datetime(daily["review_date"])

    granularity = st.radio(
        "Aggregation", ["Daily (Rolling 30-day avg)", "Raw Daily Count"],
        horizontal=True
    )

    metric_col = "rolling_avg" if "Rolling" in granularity else "daily_count"

    fig = px.line(
        daily,
        x="review_date", y=metric_col,
        color="predicted_sentiment",
        color_discrete_map={
            "positive": COLORS["positive"],
            "negative": COLORS["negative"],
            "neutral": COLORS["neutral"],
        },
        title="Sentiment Over Time",
        labels={
            "review_date": "Date",
            metric_col: "Review Count",
            "predicted_sentiment": "Sentiment",
        },
    )
    fig.update_layout(height=CHART_H + 100, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Monthly stacked bar
    monthly = trends_df[trends_df["resolution"] == "monthly"].copy()
    monthly = monthly.rename(columns={"daily_count": "count"})
    if not monthly.empty:
        fig2 = px.bar(
            monthly,
            x="review_date", y="count",
            color="predicted_sentiment",
            color_discrete_map={
                "positive": COLORS["positive"],
                "negative": COLORS["negative"],
                "neutral": COLORS["neutral"],
            },
            title="Monthly Review Volume by Sentiment",
            labels={"review_date": "Month", "count": "Reviews", "predicted_sentiment": "Sentiment"},
            barmode="stack",
        )
        fig2.update_layout(height=CHART_H)
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Feature Analysis
# ---------------------------------------------------------------------------

def tab_features(feature_df: pd.DataFrame) -> None:
    st.header("Feature Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Negative rate by feature
        fig = px.bar(
            feature_df.sort_values("negative_rate", ascending=True).tail(15),
            x="negative_rate", y="feature",
            orientation="h",
            color="negative_rate",
            color_continuous_scale="Reds",
            title="Top Features by Negative Mention Rate",
            labels={"negative_rate": "Negative Rate", "feature": "Feature"},
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=CHART_H + 50, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Mean sentiment score by feature
        fig2 = px.bar(
            feature_df.sort_values("mean_sentiment_score"),
            x="mean_sentiment_score", y="feature",
            orientation="h",
            color="mean_sentiment_score",
            color_continuous_scale="RdYlGn",
            title="Mean VADER Sentiment Score by Feature",
            labels={"mean_sentiment_score": "Avg Score", "feature": "Feature"},
        )
        fig2.update_layout(height=CHART_H + 50, coloraxis_showscale=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Feature Detail Table")

    display_cols = [
        "feature", "total_mentions", "positive_mentions", "negative_mentions",
        "negative_rate", "mean_rating", "mean_sentiment_score"
    ]
    st.dataframe(
        feature_df[display_cols].style.background_gradient(
            subset=["negative_rate"], cmap="Reds"
        ).background_gradient(
            subset=["mean_sentiment_score"], cmap="RdYlGn"
        ).format({
            "negative_rate": "{:.1%}",
            "mean_sentiment_score": "{:.3f}",
            "mean_rating": "{:.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # Scatter: total mentions vs negative rate (bubble = mean rating)
    st.subheader("Volume vs Complaint Rate")
    fig3 = px.scatter(
        feature_df,
        x="total_mentions", y="negative_rate",
        size="total_mentions", color="negative_rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="feature",
        hover_data={"mean_rating": ":.2f", "negative_rate": ":.1%"},
        title="Feature Mention Volume vs Negative Rate (size = volume)",
        labels={"total_mentions": "Total Mentions", "negative_rate": "Negative Rate"},
    )
    fig3.update_yaxes(tickformat=".0%")
    fig3.update_layout(height=CHART_H)
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Keyword Insights
# ---------------------------------------------------------------------------

def tab_keywords(keyword_df: pd.DataFrame) -> None:
    st.header("Keyword Insights")

    col1, col2 = st.columns(2)

    with col1:
        pos_kw = keyword_df[keyword_df["sentiment"] == "positive"].head(15)
        fig = px.bar(
            pos_kw.sort_values("tfidf_score"),
            x="tfidf_score", y="keyword",
            orientation="h",
            color_discrete_sequence=[COLORS["positive"]],
            title="Top Keywords in Positive Reviews",
            labels={"tfidf_score": "TF-IDF Score", "keyword": "Keyword"},
        )
        fig.update_layout(height=CHART_H + 50)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        neg_kw = keyword_df[keyword_df["sentiment"] == "negative"].head(15)
        fig2 = px.bar(
            neg_kw.sort_values("tfidf_score"),
            x="tfidf_score", y="keyword",
            orientation="h",
            color_discrete_sequence=[COLORS["negative"]],
            title="Top Keywords in Negative Reviews",
            labels={"tfidf_score": "TF-IDF Score", "keyword": "Keyword"},
        )
        fig2.update_layout(height=CHART_H + 50)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        # Neutral keywords
        neu_kw = keyword_df[keyword_df["sentiment"] == "neutral"].head(10)
        fig3 = px.bar(
            neu_kw.sort_values("tfidf_score"),
            x="tfidf_score", y="keyword",
            orientation="h",
            color_discrete_sequence=[COLORS["neutral"]],
            title="Top Keywords in Neutral Reviews",
        )
        fig3.update_layout(height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Overall top keywords
        ov_kw = keyword_df[keyword_df["sentiment"] == "overall"].head(10)
        fig4 = px.bar(
            ov_kw.sort_values("tfidf_score"),
            x="tfidf_score", y="keyword",
            orientation="h",
            color_discrete_sequence=["#3498db"],
            title="Overall Top Keywords",
        )
        fig4.update_layout(height=320)
        st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Review Explorer
# ---------------------------------------------------------------------------

def tab_explorer(df: pd.DataFrame) -> None:
    st.header("Review Explorer")

    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            brands = ["All"] + sorted(df["brand"].unique().tolist())
            brand_sel = st.selectbox("Brand", brands)
        with col2:
            ratings = st.multiselect("Rating (stars)", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        with col3:
            sentiments = st.multiselect(
                "Sentiment", ["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"]
            )
        with col4:
            date_min = df["review_date"].min().date()
            date_max = df["review_date"].max().date()
            date_range = st.date_input(
                "Date range",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max,
            )

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    if brand_sel != "All":
        mask &= df["brand"] == brand_sel
    if ratings:
        mask &= df["rating"].isin(ratings)
    if sentiments:
        mask &= df["predicted_sentiment"].isin(sentiments)
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= (df["review_date"] >= start) & (df["review_date"] <= end)

    filtered = df[mask].copy()
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} reviews")

    # Summary mini-KPIs for filtered subset
    if len(filtered) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positive", f"{(filtered['predicted_sentiment']=='positive').mean()*100:.1f}%")
        c2.metric("Negative", f"{(filtered['predicted_sentiment']=='negative').mean()*100:.1f}%")
        c3.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")
        c4.metric("Return Rate", f"{filtered['return_flag'].mean()*100:.1f}%")

    # Table
    display_cols = [
        "review_id", "brand", "rating", "predicted_sentiment",
        "sentiment_score", "review_date", "country", "price_paid",
        "verified_purchase", "return_flag", "review_text",
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "review_id": "ID",
            "predicted_sentiment": "Sentiment",
            "sentiment_score": "Score",
            "review_date": "Date",
            "price_paid": "Price ($)",
            "verified_purchase": "Verified",
            "return_flag": "Returned",
            "review_text": "Review",
        }).reset_index(drop=True),
        height=500,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Tab 6 — Recommendations
# ---------------------------------------------------------------------------

def tab_recommendations(metrics: dict) -> None:
    st.header("Business Recommendations")
    st.caption("Rule-based insights generated from pipeline analytics. Deterministic, not LLM-generated.")

    recommendations = metrics.get("recommendations", [])
    if not recommendations:
        st.info("No recommendations generated. Run the pipeline first.")
        return

    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    severity_order = {"high": 0, "medium": 1, "low": 2}

    sorted_recs = sorted(recommendations, key=lambda r: severity_order.get(r.get("severity", "low"), 3))

    categories = list(dict.fromkeys(r["category"] for r in sorted_recs))
    for cat in categories:
        st.subheader(f"📌 {cat}")
        cat_recs = [r for r in sorted_recs if r["category"] == cat]
        for rec in cat_recs:
            icon = severity_icons.get(rec.get("severity", "low"), "ℹ️")
            severity = rec.get("severity", "low").upper()
            with st.container():
                st.markdown(
                    f"{icon} **[{severity}]** {rec['insight']}"
                )

    st.divider()
    st.subheader("Pipeline Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Backend", metrics.get("sentiment_backend", "vader").upper())
    col2.metric("Total Reviews", f"{metrics.get('total_reviews', 0):,}")
    col3.metric("Pipeline Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    col4.metric("Run Time", f"{metrics.get('pipeline_elapsed_seconds', 0):.1f}s")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Headphones.svg/120px-Headphones.svg.png",
            width=80,
        )
        st.title("Sentiment Dashboard")
        st.caption(f"**{len(df):,}** reviews analysed")
        st.divider()

        st.markdown("**Quick Links**")
        st.markdown(
            "- [GitHub](https://github.com)\n"
            "- [Run Pipeline](# 'python -m src.pipeline')\n"
        )

        st.divider()
        st.caption(
            f"Data: {df['review_date'].min().date()} → {df['review_date'].max().date()}"
        )
        st.caption("Backend: VADER Sentiment")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_outputs():
        st.error(
            "Pipeline outputs not found. Please run:\n\n"
            "```\npython data/generate_reviews.py\npython -m src.pipeline\n```"
        )
        st.stop()

    df = load_scored_reviews()
    feature_df = load_feature_summary()
    keyword_df = load_keyword_summary()
    trends_df = load_time_trends()
    metrics = load_metrics()

    render_sidebar(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Trends",
        "🔧 Feature Analysis",
        "🔑 Keyword Insights",
        "🔍 Review Explorer",
        "💡 Recommendations",
    ])

    with tab1:
        tab_overview(df, metrics.get("kpis", {}), metrics)
    with tab2:
        tab_trends(trends_df)
    with tab3:
        tab_features(feature_df)
    with tab4:
        tab_keywords(keyword_df)
    with tab5:
        tab_explorer(df)
    with tab6:
        tab_recommendations(metrics)


if __name__ == "__main__":
    main()
