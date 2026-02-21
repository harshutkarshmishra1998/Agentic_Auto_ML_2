import numpy as np


def apply(df):
    """
    Remove structurally useless columns BEFORE any transformation.
    """

    dropped = []

    # -----------------------------
    # 1. all-null columns
    # -----------------------------
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        df = df.drop(columns=null_cols)
        dropped.extend(null_cols)

    # -----------------------------
    # 2. constant columns
    # -----------------------------
    constant_cols = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        dropped.extend(constant_cols)

    # -----------------------------
    # 3. near-zero variance numeric
    # -----------------------------
    numeric = df.select_dtypes(include=np.number)
    low_var = numeric.var()
    low_var_cols = low_var[low_var < 1e-12].index.tolist()

    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        dropped.extend(low_var_cols)

    return df, {
        "strategy": "data_sanity",
        "dropped_columns": list(set(dropped))
    }