import pandas as pd

from data_understanding.column_profiler import (
    distribution_shape,
    outliers_present,
    transform_hint,
)


def test_boolean_series_is_not_profiled_as_numeric_distribution_or_outliers():
    series = pd.Series([True, False, True, False, True, False])

    assert distribution_shape(series) == "N/A"
    assert outliers_present(series) is False
    assert transform_hint(series) is None
