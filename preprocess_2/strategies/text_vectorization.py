from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def apply(df, columns):

    if not columns:
        return df, None

    log = {"strategy": "text_vectorization", "columns": columns}

    for col in columns:

        tf = TfidfVectorizer(max_features=200)
        arr = tf.fit_transform(df[col].astype(str)).toarray() #type:ignore

        tf_df = pd.DataFrame(
            arr,
            columns=[f"{col}_tfidf_{i}" for i in range(arr.shape[1])],
            index=df.index
        )

        df = df.drop(columns=[col])
        df = pd.concat([df, tf_df], axis=1)

    return df, log