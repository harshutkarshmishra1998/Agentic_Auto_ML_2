from sklearn.preprocessing import StandardScaler

def apply(df, columns):

    if not columns:
        return df, None

    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df, {
        "strategy": "scaling",
        "columns": columns
    }