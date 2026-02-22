from .schema import DatasetSignals


def _get_strategy(exp, name):
    return next(
        (s for s in exp["strategy_logs"] if s["strategy"] == name),
        {}
    )


def extract_signals(exp):

    rows = exp["rows"]
    cols = exp["columns"]

    encoding = _get_strategy(exp, "encoding")
    scaling = _get_strategy(exp, "scaling")

    onehot = encoding.get("onehot_columns", [])
    freq = encoding.get("frequency_columns", [])
    label = encoding.get("label_columns", [])  # optional if present

    is_small = rows < 10_000
    is_large = rows > 100_000
    is_high_dim = cols > 100

    feature_sample_ratio = cols / max(rows, 1)
    overfit = feature_sample_ratio > 0.1

    clustering_ready = bool(scaling)

    return DatasetSignals(
        rows=rows,
        cols=cols,
        is_small_data=is_small,
        is_large_data=is_large,
        is_high_dim=is_high_dim,
        overfit_risk=overfit,
        onehot_count=len(onehot),
        freq_encoded_count=len(freq),
        label_encoded_count=len(label),
        clustering_ready=clustering_ready
    )