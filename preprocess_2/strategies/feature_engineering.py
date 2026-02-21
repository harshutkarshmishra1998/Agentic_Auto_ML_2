import numpy as np
import pandas as pd


def apply(df, rules=None):
    """
    Generic deterministic feature engineering.
    If rules None -> automatic heuristics.
    """

    log = {"strategy": "feature_engineering", "created_features": []}

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    # --------------------------------
    # 1. LOG STABILIZATION
    # --------------------------------
    for col in numeric:
        if (df[col] > 0).all() and df[col].skew() > 1:
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            log["created_features"].append(new_col)

    # --------------------------------
    # 2. POLYNOMIAL DEGREE 2
    # --------------------------------
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):

            c1 = numeric[i]
            c2 = numeric[j]

            new_col = f"{c1}_x_{c2}"
            df[new_col] = df[c1] * df[c2]
            log["created_features"].append(new_col)

    # --------------------------------
    # 3. RATIO FEATURES
    # --------------------------------
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):

            c1 = numeric[i]
            c2 = numeric[j]

            if (df[c2] != 0).all():
                new_col = f"{c1}_div_{c2}"
                df[new_col] = df[c1] / (df[c2] + 1e-9)
                log["created_features"].append(new_col)

    # --------------------------------
    # 4. FREQUENCY ENCODING (categorical)
    # --------------------------------
    categorical = df.select_dtypes(include="object").columns.tolist()

    for col in categorical:
        freq = df[col].value_counts(normalize=True)
        new_col = f"{col}_freq"
        df[new_col] = df[col].map(freq)
        log["created_features"].append(new_col)

    return df, log