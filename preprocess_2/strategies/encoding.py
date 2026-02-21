import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def apply(df, columns):

    if not columns:
        return df, None

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    arr = enc.fit_transform(df[columns])

    enc_df = pd.DataFrame(
        arr,
        columns=enc.get_feature_names_out(columns),
        index=df.index
    )

    df = df.drop(columns=columns)
    df = pd.concat([df, enc_df], axis=1)

    return df, {
        "strategy": "encoding",
        "columns": columns,
        "method": "one_hot"
    }