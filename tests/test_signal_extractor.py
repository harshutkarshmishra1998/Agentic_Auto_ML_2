from model_intializer.signal_extractor import extract_signals


def test_extract_signals_with_none_rows_columns_defaults_to_zero():
    exp = {
        "rows": None,
        "columns": None,
        "strategy_logs": [],
    }

    signals = extract_signals(exp)

    assert signals.rows == 0
    assert signals.cols == 0
    assert signals.is_small_data is True
    assert signals.is_large_data is False


def test_extract_signals_supports_alternate_dimension_keys():
    exp = {
        "row_count": "2500",
        "n_columns": 150,
        "strategy_logs": [{"strategy": "scaling", "scaled_columns": ["a"]}],
    }

    signals = extract_signals(exp)

    assert signals.rows == 2500
    assert signals.cols == 150
    assert signals.is_high_dim is True
    assert signals.clustering_ready is True
