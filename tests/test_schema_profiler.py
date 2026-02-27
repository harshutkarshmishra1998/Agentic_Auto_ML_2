import pandas as pd

from schema_engine.profiler import profile_dataframe


def test_profile_dataframe_handles_mixed_object_column_min_max():
    df = pd.DataFrame({
        "mixed": ["A", 1.5, "B", None],
    })

    profile = profile_dataframe(df)["mixed"]

    assert profile.min_val == "1.5"
    assert profile.max_val == "B"


def test_profile_dataframe_handles_all_missing_column_min_max():
    df = pd.DataFrame({
        "empty": [None, None, None],
    })

    profile = profile_dataframe(df)["empty"]

    assert profile.min_val is None
    assert profile.max_val is None
