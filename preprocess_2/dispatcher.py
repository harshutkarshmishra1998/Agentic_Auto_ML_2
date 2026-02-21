from .strategies import (
    data_sanity,
    scaling,
    encoding,
    correlation,
    feature_engineering,
    text_vectorization
)

# strict order
PIPELINE_ORDER = [
    "data_sanity",
    "encoding",
    "scaling",
    "correlation_resolution",
    "feature_engineering",
]

STRATEGY_MAP = {
    "data_sanity": data_sanity.apply,
    "scaling": scaling.apply,
    "encoding": encoding.apply,
    "correlation_resolution": correlation.apply,
    "feature_engineering": feature_engineering.apply,
    "text_vectorization": text_vectorization.apply,
}


def run_strategies(df, context):

    logs = []

    for strategy in PIPELINE_ORDER:

        if strategy == "data_sanity":
            df, log = STRATEGY_MAP[strategy](df)

        elif strategy == "correlation_resolution":
            df, log = STRATEGY_MAP[strategy](df)

        else:
            cols = context.get_columns(strategy)
            df, log = STRATEGY_MAP[strategy](df, cols)

        if log:
            logs.append(log)

    return df, logs