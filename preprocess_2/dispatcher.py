from .strategies import scaling, encoding, correlation, text_vectorization, feature_engineering

STRATEGY_MAP = {
    "scaling": scaling.apply,
    "encoding": encoding.apply,
    "correlation_resolution": correlation.apply,
    "text_vectorization": text_vectorization.apply,
    "feature_engineering": feature_engineering.apply,
}

def run_strategies(df, context):

    logs = []

    for strategy in STRATEGY_MAP:

        cols = context.get_columns(strategy)

        if strategy == "correlation_resolution":
            df, log = STRATEGY_MAP[strategy](df)
        else:
            df, log = STRATEGY_MAP[strategy](df, cols)

        if log:
            logs.append(log)

    return df, logs