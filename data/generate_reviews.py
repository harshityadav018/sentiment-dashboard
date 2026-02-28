"""
Synthetic customer review generator for wireless earbuds.

Produces 10,000 realistic reviews with:
- Correlated ratings and sentiments (with realistic noise)
- Feature mentions that drive sentiment (battery, sound, comfort, etc.)
- Human-like prose variation across brands and sentiments
- Geographic and price distribution
"""

import random
import csv
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "reviews.csv"

BRANDS = ["SoundCore", "JabRa", "SonyX", "UrbanBeats", "AirPods Pro", "BeatsFlex"]
COUNTRIES = ["US", "UK", "CA", "IN", "AU"]

# Feature pools with sentiment bias
POSITIVE_FEATURES = [
    "sound quality", "bass", "noise cancellation", "battery life",
    "comfort", "fit", "call quality", "microphone", "fast charging",
    "connectivity", "build quality", "touch controls", "case design",
    "transparency mode", "spatial audio",
]
NEGATIVE_FEATURES = [
    "battery life", "connectivity", "microphone", "fit", "comfort",
    "touch controls", "call quality", "pairing", "durability",
    "charging case", "ear tips", "app support",
]
NEUTRAL_FEATURES = [
    "sound quality", "bass", "treble", "design", "weight",
    "water resistance", "codec support", "latency",
]

# ---------------------------------------------------------------------------
# Text templates: pros/cons/mixed, varied writing styles
# ---------------------------------------------------------------------------
POSITIVE_OPENERS = [
    "Absolutely love these earbuds.",
    "Best purchase I've made this year.",
    "These have completely replaced my old pair.",
    "Blown away by the quality for the price.",
    "Finally found earbuds that actually fit my ears.",
    "Five stars, no question.",
    "Honestly exceeded my expectations.",
    "These are a game-changer for my daily commute.",
    "Couldn't be happier with this purchase.",
    "Solid upgrade from my previous pair.",
]

NEGATIVE_OPENERS = [
    "Very disappointed with this product.",
    "Returned after three days.",
    "Save your money and look elsewhere.",
    "Expected so much more based on the reviews.",
    "These stopped working after two weeks.",
    "Not worth the price at all.",
    "Wouldn't recommend to my worst enemy.",
    "Had high hopes but left frustrated.",
    "These are going straight back.",
    "Genuinely the worst earbuds I've owned.",
]

MIXED_OPENERS = [
    "Good but not great.",
    "Has some nice features but also some serious drawbacks.",
    "Decent for the price, but with caveats.",
    "Half the things work great, the other half let me down.",
    "Mixed feelings after two months of use.",
    "Some things I love, some things drive me crazy.",
    "Pretty good overall but there's room for improvement.",
    "Not perfect but a reasonable buy.",
]

POSITIVE_BODIES = [
    "The {f1} is exceptional — way better than anything I've tried at this price point. "
    "I use them for {use_case} and they never let me down. "
    "The {f2} is also surprisingly good. Highly recommend.",

    "What really stands out is the {f1}. I was skeptical but after a month of heavy use "
    "I'm a convert. The {f2} works flawlessly too. {brand} really nailed it here.",

    "I was comparing this against two other brands and {brand} won on {f1} and {f2}. "
    "Setup was painless, connection is stable, and they look great. "
    "My only tiny gripe is that the case could be sleeker but that's splitting hairs.",

    "The {f1} alone is worth the price. I work in a loud office and these have "
    "transformed my focus. {f2} is solid too. Battery easily lasts my whole workday.",

    "Bought these for {use_case} and they're perfect. {f1} is top tier, "
    "{f2} impressed me too. Would buy again without hesitation.",
]

NEGATIVE_BODIES = [
    "The {f1} is terrible. Drops out constantly even within a meter of my phone. "
    "The {f2} is equally bad — scratchy and distant on calls. "
    "I cannot believe {brand} shipped this at this price point.",

    "After just {weeks} weeks the {f1} started failing. The right earbud "
    "died first, then the left. Support was no help. Avoid.",

    "The {f1} drains so fast I need to charge mid-day. That's unacceptable "
    "in {year}. The {f2} doesn't work half the time either. Very frustrating.",

    "{brand} has really let quality slip. The {f1} is laughably poor, "
    "and the {f2} is even worse. My $20 earbuds from three years ago "
    "outperformed these in every way.",

    "Returned these after one week. The {f1} gave me constant headaches "
    "from the poor fit and the {f2} kept disconnecting. Buyer beware.",
]

MIXED_BODIES = [
    "The {f1} is genuinely impressive — I was not expecting that at this price. "
    "However, the {f2} is a letdown. It {complaint}. "
    "If they fix that in a firmware update I'd bump this to five stars.",

    "Great {f1}, average {f2}. The positives outweigh the negatives for me "
    "but your mileage may vary depending on what you prioritize.",

    "Love the {f1} and the {f2} is fine for casual use. "
    "My main complaint is {complaint}. Not a dealbreaker but worth knowing.",

    "Two months in: the {f1} still holds up well. The {f2} has been inconsistent "
    "— sometimes perfect, sometimes frustrating. Overall a decent buy.",

    "The {f1} surprised me positively. The {f2} is hit or miss. "
    "{complaint}. If you can live with that, it's a fair deal.",
]

USE_CASES = [
    "working from home", "the gym", "my commute", "running",
    "long flights", "office work", "gaming", "cycling",
]

COMPLAINTS = [
    "keeps disconnecting when my phone screen locks",
    "the touch controls are way too sensitive",
    "takes forever to pair on startup",
    "the app is buggy and crashes",
    "ear tips don't stay in during exercise",
    "call quality is mediocre at best",
    "the bass is too boosted for my taste",
    "the companion app is basically useless",
]


# ---------------------------------------------------------------------------
# Price ranges per brand
# ---------------------------------------------------------------------------
BRAND_PRICES = {
    "SoundCore": (35, 80),
    "JabRa": (90, 200),
    "SonyX": (70, 180),
    "UrbanBeats": (25, 60),
    "AirPods Pro": (180, 280),
    "BeatsFlex": (100, 220),
}

# Brand return rate bias (higher = more likely to be returned)
BRAND_RETURN_BIAS = {
    "SoundCore": 0.08,
    "JabRa": 0.05,
    "SonyX": 0.06,
    "UrbanBeats": 0.18,
    "AirPods Pro": 0.04,
    "BeatsFlex": 0.10,
}

# Country distribution weights
COUNTRY_WEIGHTS = [0.45, 0.20, 0.15, 0.12, 0.08]


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def _pick_rating(sentiment: str) -> int:
    """Rating correlated with sentiment, with realistic noise."""
    if sentiment == "positive":
        weights = [2, 3, 8, 25, 62]   # 1–5 star weights
    elif sentiment == "negative":
        weights = [45, 30, 15, 7, 3]
    else:  # neutral/mixed
        weights = [5, 10, 40, 30, 15]
    return random.choices([1, 2, 3, 4, 5], weights=weights)[0]


def _pick_features(sentiment: str, n_min: int = 1, n_max: int = 4) -> list[str]:
    """Pick feature mentions biased toward sentiment-matching pools."""
    n = random.randint(n_min, n_max)
    if sentiment == "positive":
        pool = POSITIVE_FEATURES + NEUTRAL_FEATURES
    elif sentiment == "negative":
        pool = NEGATIVE_FEATURES + NEUTRAL_FEATURES
    else:
        pool = POSITIVE_FEATURES + NEGATIVE_FEATURES + NEUTRAL_FEATURES
    unique_pool = list(dict.fromkeys(pool))  # deduplicate while preserving order
    return random.sample(unique_pool, min(n, len(unique_pool)))


def _build_text(sentiment: str, brand: str, features: list[str]) -> str:
    """Compose a human-like review paragraph."""
    f1 = features[0] if features else "sound quality"
    f2 = features[1] if len(features) > 1 else "battery life"
    use_case = random.choice(USE_CASES)
    complaint = random.choice(COMPLAINTS)
    weeks = random.randint(1, 4)
    year = 2025

    if sentiment == "positive":
        opener = random.choice(POSITIVE_OPENERS)
        body_tpl = random.choice(POSITIVE_BODIES)
        body = body_tpl.format(
            f1=f1, f2=f2, brand=brand, use_case=use_case, weeks=weeks, year=year
        )
    elif sentiment == "negative":
        opener = random.choice(NEGATIVE_OPENERS)
        body_tpl = random.choice(NEGATIVE_BODIES)
        body = body_tpl.format(
            f1=f1, f2=f2, brand=brand, use_case=use_case, weeks=weeks, year=year
        )
    else:
        opener = random.choice(MIXED_OPENERS)
        body_tpl = random.choice(MIXED_BODIES)
        body = body_tpl.format(
            f1=f1, f2=f2, brand=brand, use_case=use_case,
            complaint=complaint, weeks=weeks, year=year
        )

    # Occasionally append a short tail sentence for variety
    tails = [
        f" Shipping was fast too.",
        f" Packaging was nice.",
        f" Worth every penny.",
        f" Update your firmware first — it makes a difference.",
        f" The companion app needs work though.",
        f" Would make a great gift.",
        "",  # no tail sometimes
        "",
    ]
    return f"{opener} {body}{random.choice(tails)}"


def _random_date(start: datetime, end: datetime) -> str:
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_reviews(n: int = 10_000) -> list[dict]:
    end_date = datetime(2025, 12, 31)
    start_date = end_date - timedelta(days=365 * 2)

    # Sentiment distribution: ~55% positive, 25% negative, 20% neutral
    sentiments = random.choices(
        ["positive", "negative", "neutral"],
        weights=[55, 25, 20],
        k=n,
    )

    records = []
    for i, sentiment in enumerate(sentiments):
        brand = random.choice(BRANDS)
        features = _pick_features(sentiment)
        rating = _pick_rating(sentiment)
        price_lo, price_hi = BRAND_PRICES[brand]
        price_paid = round(random.uniform(price_lo, price_hi), 2)
        return_prob = BRAND_RETURN_BIAS[brand]
        # Higher return probability for negative reviews
        if sentiment == "negative":
            return_prob = min(return_prob * 2.5, 0.55)
        return_flag = random.random() < return_prob

        records.append({
            "review_id": f"R{i+1:05d}",
            "brand": brand,
            "rating": rating,
            "review_text": _build_text(sentiment, brand, features),
            "review_date": _random_date(start_date, end_date),
            "verified_purchase": random.random() < 0.78,
            "country": random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0],
            "price_paid": price_paid,
            "return_flag": return_flag,
            "features_mentioned": ";".join(features),
            "ground_truth_sentiment": sentiment,
        })

    return records


def save_reviews(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[generate_reviews] Saved {len(records):,} reviews → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    print(f"[generate_reviews] Generating {n:,} synthetic reviews …")
    records = generate_reviews(n)
    save_reviews(records, OUTPUT_PATH)

    # Quick sanity check
    sentiments = [r["ground_truth_sentiment"] for r in records]
    for label in ("positive", "negative", "neutral"):
        count = sentiments.count(label)
        print(f"  {label:>10}: {count:,} ({count/len(records)*100:.1f}%)")
