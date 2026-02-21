import numpy as np


def column_quality_score(df, col):
    """
    Higher score = better column (keep it)
    """

    missing_ratio = df[col].isna().mean()
    variance = df[col].var()

    return (1 - missing_ratio) * (variance + 1e-9)


def apply(df, threshold=0.95):

    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = set()

    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:

                if row in drop_cols or col in drop_cols:
                    continue

                score_row = column_quality_score(df, row)
                score_col = column_quality_score(df, col)

                drop = row if score_row < score_col else col
                drop_cols.add(drop)

    df = df.drop(columns=list(drop_cols), errors="ignore")

    return df, {
        "strategy": "correlation_resolution",
        "threshold": threshold,
        "dropped_columns": list(drop_cols)
    }