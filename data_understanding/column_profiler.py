import pandas as pd
import numpy as np
from scipy.stats import skew


def _is_non_boolean_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _safe_abs_skew(series):
    """
    Returns absolute skew for sufficiently variable numeric data.
    Prevents scipy precision-loss warnings on near-constant vectors.
    """
    s = series.dropna()

    if len(s) < 5:
        return None

    # scipy.stats.skew can emit runtime warnings on (near) constant values
    if s.nunique(dropna=True) <= 1:
        return 0.0

    if np.isclose(float(s.std(ddof=0)), 0.0):
        return 0.0

    return abs(float(skew(s)))

def is_constant(series):
    """
    True if column has zero variance (only one unique value).
    """
    return series.nunique(dropna=True) <= 1


# missing pattern
def missing_pattern(series, df):
    """
    Detect whether missingness is correlated with other variables.
    Safe against zero-variance columns.
    """

    # no missing values
    if series.isna().sum() == 0:
        return "none"

    indicator = series.isna().astype(int)

    # indicator constant → cannot correlate
    if indicator.nunique(dropna=True) <= 1:
        return "random"

    numeric = df.select_dtypes(include="number")

    # remove constant numeric columns
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    for col in numeric.columns:

        if col == series.name:
            continue

        other = numeric[col].fillna(0)

        # other constant after fill → skip
        if other.nunique(dropna=True) <= 1:
            continue

        try:
            corr = indicator.corr(other)
        except Exception:
            continue

        if corr is not None and not pd.isna(corr) and abs(corr) > 0.3:
            return "MAR"

    return "random"


# distribution
def distribution_shape(series):
    if not _is_non_boolean_numeric(series):
        return "N/A"

    abs_skew = _safe_abs_skew(series)
    if abs_skew is None:
        return "unknown"

    return "symmetric" if abs_skew < 0.5 else "skewed"


def outliers_present(series):
    if not _is_non_boolean_numeric(series):
        return False
    s = series.dropna()
    if len(s) < 5:
        return False
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return bool(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum() > 0)


# cardinality
def cardinality_level(unique_ratio):
    if unique_ratio > 0.9:
        return "very_high"
    if unique_ratio > 0.3:
        return "high"
    if unique_ratio > 0.05:
        return "medium"
    return "low"


# encoding need
def encoding_required(semantic_type):
    return semantic_type in {
        "categorical_nominal",
        "categorical_ordinal",
        "text_freeform"
    }


# text complexity
def text_complexity(series):
    if series.dtype != "object":
        return None
    avg_len = series.dropna().astype(str).str.len().mean()
    if avg_len > 100:
        return "long"
    if avg_len > 30:
        return "medium"
    return "short"


# category imbalance
def category_imbalance(series):
    if series.nunique() > 30:
        return None
    p = series.value_counts(normalize=True)
    if len(p) == 0:
        return None
    return float(p.iloc[0])


def correlation_strength(col, df):
    """
    Returns max absolute correlation with other numeric columns.
    Safely ignores constant columns to avoid divide-by-zero warnings.
    """

    numeric = df.select_dtypes(include="number")

    # remove constant columns (zero variance)
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    # if column not valid after filtering
    if col not in numeric.columns:
        return None

    # need at least 2 numeric columns to compute correlation
    if numeric.shape[1] < 2:
        return None

    corr_matrix = numeric.corr()

    if col not in corr_matrix:
        return None

    max_corr = corr_matrix[col].drop(col).abs().max()

    if max_corr is None or pd.isna(max_corr):
        return None

    return float(max_corr)


# transform hint
def transform_hint(series):
    if not _is_non_boolean_numeric(series):
        return None

    abs_skew = _safe_abs_skew(series)
    if abs_skew is None:
        return None

    if abs_skew > 1:
        return "log_candidate"

    return None


# modeling hint
def modeling_hint(semantic_type):
    if semantic_type.startswith("categorical"):
        return "encoding"
    if semantic_type == "numeric_continuous":
        return "scaling"
    if semantic_type == "text_freeform":
        return "nlp"
    return None
