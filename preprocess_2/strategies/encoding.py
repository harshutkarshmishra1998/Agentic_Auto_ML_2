# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

# def apply(df, columns):

#     if not columns:
#         return df, None

#     enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

#     arr = enc.fit_transform(df[columns])

#     enc_df = pd.DataFrame(
#         arr,
#         columns=enc.get_feature_names_out(columns),
#         index=df.index
#     )

#     df = df.drop(columns=columns)
#     df = pd.concat([df, enc_df], axis=1)

#     return df, {
#         "strategy": "encoding",
#         "columns": columns,
#         "method": "one_hot"
#     }

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

MAX_ONEHOT_CARDINALITY = 50


def apply(df, columns):

    if not columns:
        return df, None

    onehot_cols = []
    freq_cols = []
    dropped_cols = []

    # classify columns by cardinality
    for col in columns:

        nunique = df[col].nunique(dropna=False)

        if nunique <= MAX_ONEHOT_CARDINALITY:
            onehot_cols.append(col)
        else:
            freq_cols.append(col)

    logs = {
        "strategy": "encoding",
        "onehot_columns": onehot_cols,
        "frequency_columns": freq_cols,
    }

    # frequency encoding
    for col in freq_cols:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)
        df.drop(columns=col, inplace=True)

    # one-hot encoding (sparse safe)
    if onehot_cols:

        enc = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

        sparse_mat = enc.fit_transform(df[onehot_cols])

        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            sparse_mat,
            columns=enc.get_feature_names_out(onehot_cols),
            index=df.index
        )

        # encoded_df = encoded_df.sparse.to_dense().astype("int8")
        encoded_df = (
            encoded_df
            .sparse.to_dense()
            .fillna(0)
            .astype("int8")
        )

        df.drop(columns=onehot_cols, inplace=True)
        df = pd.concat([df, encoded_df], axis=1)

    logs["final_columns_added"] = (
        len(onehot_cols) + len(freq_cols)
    )

    return df, logs