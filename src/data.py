"""Data preparation utilities for the Series B prediction project.

This module keeps the data-specific logic in one place:
1. read the raw Crunchbase-derived Kaggle CSV,
2. coerce date, money, numeric, and categorical columns,
3. construct the approximate Series B-within-36-months target, and
4. define the strict and expanded feature specifications used by the models.

The target is intentionally documented here because it is the most important
modeling assumption in the project. The raw dataset does not provide the exact
Series B event date, so the label should be interpreted as a useful proxy rather
than a perfect event-history outcome.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# A startup must have first funding on or before this cutoff so the project can
# observe a full 36-month outcome window in the available historical data.
OBSERVATION_CUTOFF = pd.Timestamp("2015-01-01")

# Average month length used when converting day differences into month counts.
DAYS_PER_MONTH = 30.4375  # 365.25 / 12

# Strict features are intended to be available near the first funding event.
STRICT_NUMERIC = [
    "founded_year",
    "months_to_first_funding",
    "category_count",
    "seed",
    "angel",
    "grant",
    "convertible_note",
    "debt_financing",
    "equity_crowdfunding",
]
STRICT_CATEGORICAL = ["market", "country_code", "state_code"]

# Expanded features are predictive but less pure as an early-stage forecast.
EXPANDED_EXTRA_NUMERIC = ["funding_total_usd", "funding_rounds"]


@dataclass
class PreparedData:
    """Container returned by ``load_and_prepare``.

    Attributes
    ----------
    df:
        Cleaned analysis DataFrame after eligibility filtering.
    feature_specs:
        Dictionary defining the strict and expanded feature groups.
    metadata:
        Human-readable dataset summary used in reports and the README.
    """

    df: pd.DataFrame
    feature_specs: dict
    metadata: dict


def _parse_money(val):
    """Convert money-like strings such as ``"1,000,000"`` to floats."""
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().replace(",", ""))
    except ValueError:
        return np.nan


def _coerce_dates(df, cols):
    """Parse date columns in-place, turning invalid values into ``NaT``."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _coerce_numeric(df, cols):
    """Convert numeric-like columns while preserving missing values."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fill_categorical(df, cols):
    """Standardize categorical columns and add ``Unknown`` where needed."""
    for c in cols:
        if c not in df.columns:
            df[c] = "Unknown"
            continue
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
        df[c] = df[c].replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
    return df


def load_and_prepare(csv_path, encoding="latin1"):
    """Load the raw CSV and return the cleaned modeling sample.

    Parameters
    ----------
    csv_path:
        Path to the Crunchbase-derived Kaggle CSV.
    encoding:
        File encoding. The public Kaggle dump commonly requires ``latin1``.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    raw_rows = len(df)

    # Normalize column names to protect the pipeline from accidental whitespace.
    df.columns = [c.strip() for c in df.columns]

    df = _coerce_dates(df, ["founded_at", "first_funding_at", "last_funding_at"])

    if "funding_total_usd" in df.columns:
        df["funding_total_usd"] = df["funding_total_usd"].map(_parse_money)

    numeric_like = [
        "funding_rounds", "founded_year", "seed", "angel", "grant",
        "convertible_note", "debt_financing", "equity_crowdfunding", "round_B",
    ]
    df = _coerce_numeric(df, numeric_like)

    # ``category_list`` is pipe-delimited and often wrapped with leading/trailing
    # pipes, so the raw pipe count is one higher than the number of categories.
    if "category_list" in df.columns:
        pipe_counts = df["category_list"].fillna("").astype(str).str.count(r"\|")
        df["category_count"] = np.where(pipe_counts > 0, pipe_counts - 1, 0)
    else:
        df["category_count"] = 0

    df = _fill_categorical(df, ["market", "country_code", "state_code"])

    df["founded_year"] = pd.to_numeric(df.get("founded_year"), errors="coerce")
    df["months_to_first_funding"] = (
        (df["first_funding_at"] - df["founded_at"]).dt.days / DAYS_PER_MONTH
    )

    obs_months = (df["last_funding_at"] - df["first_funding_at"]).dt.days / DAYS_PER_MONTH
    df["obs_months_after_first_funding"] = obs_months

    # Approximate target: the dataset indicates a Series B round and the observed
    # funding window places that milestone within 36 months of first funding.
    df["target_series_b_36m"] = (
        (df["round_B"].fillna(0) > 0) & (obs_months <= 36)
    ).astype(int)

    have_dates = df["first_funding_at"].notna() & df["last_funding_at"].notna()
    within_cutoff = df["first_funding_at"] <= OBSERVATION_CUTOFF
    non_negative = obs_months >= 0
    eligible = df[have_dates & within_cutoff & non_negative].copy()

    # Keep positives. For negatives, require at least 36 months of observation;
    # otherwise the startup could still become positive later.
    is_positive = eligible["target_series_b_36m"] == 1
    enough_window = eligible["obs_months_after_first_funding"] >= 36
    eligible = eligible[is_positive | enough_window].copy()

    positive_rate = float(eligible["target_series_b_36m"].mean()) if len(eligible) else 0.0

    metadata = {
        "raw_rows": int(raw_rows),
        "eligible_rows": int(len(eligible)),
        "positive_rate": positive_rate,
        "observation_cutoff": str(OBSERVATION_CUTOFF.date()),
        "target_definition": (
            "Approximate label: round_B > 0 and last_funding_at - first_funding_at "
            "<= 36 months; kept only if first_funding_at <= 2015-01-01 and (positive "
            "or at least 36 months of observation)."
        ),
        "important_limitation": (
            "The CSV does not include the exact date of the Series B event, "
            "so the milestone label is approximate."
        ),
    }

    feature_specs = {
        "strict": {
            "numeric": [c for c in STRICT_NUMERIC if c in eligible.columns],
            "categorical": [c for c in STRICT_CATEGORICAL if c in eligible.columns],
        },
        "expanded": {
            "numeric": [c for c in (STRICT_NUMERIC + EXPANDED_EXTRA_NUMERIC) if c in eligible.columns],
            "categorical": [c for c in STRICT_CATEGORICAL if c in eligible.columns],
        },
    }

    return PreparedData(df=eligible, feature_specs=feature_specs, metadata=metadata)
