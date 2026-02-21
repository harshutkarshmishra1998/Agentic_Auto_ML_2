# from sklearn.preprocessing import StandardScaler

# def apply(df, columns):

#     if not columns:
#         return df, None

#     scaler = StandardScaler()
#     df[columns] = scaler.fit_transform(df[columns])

#     return df, {
#         "strategy": "scaling",
#         "columns": columns
#     }

import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply(df, columns):

    if not columns:
        return df, None

    numeric_cols = []
    coerced_cols = []
    skipped_cols = []

    # ---------------------------------
    # validate numeric compatibility
    # ---------------------------------
    for col in columns:

        if col not in df.columns:
            continue

        # already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue

        # try safe conversion
        converted = pd.to_numeric(df[col], errors="coerce")

        # if conversion preserves most values → accept
        non_na_ratio = converted.notna().mean()

        if non_na_ratio > 0.95:
            df[col] = converted
            numeric_cols.append(col)
            coerced_cols.append(col)
        else:
            skipped_cols.append(col)

    # ---------------------------------
    # nothing numeric → skip
    # ---------------------------------
    if not numeric_cols:
        return df, {
            "strategy": "scaling",
            "skipped_columns": skipped_cols,
            "reason": "no_numeric_columns"
        }

    # ---------------------------------
    # apply scaling
    # ---------------------------------
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, {
        "strategy": "scaling",
        "scaled_columns": numeric_cols,
        "coerced_to_numeric": coerced_cols,
        "skipped_columns": skipped_cols
    }