import pandas as pd


def load_dataset(path, target_column=None):

    df = pd.read_csv(path)

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None

    return X, y