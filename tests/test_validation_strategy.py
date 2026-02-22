import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ml_engine.validation_strategy import _prepare_features, run_validation


def test_prepare_features_encodes_string_datetime_columns():
    X = pd.DataFrame(
        {
            "ts": [
                "2017-09-13 21:47:47.498520+00",
                "2017-09-13 21:47:47.598520+00",
                "2017-09-13 21:47:47.698520+00",
            ],
            "value": [1.2, 3.4, 5.6],
        }
    )

    prepared = _prepare_features(X)

    assert prepared.shape[1] >= 2
    assert all(dtype.kind in {"i", "u", "f", "b"} for dtype in prepared.dtypes)


def test_run_validation_with_datetime_strings_does_not_crash():
    X = pd.DataFrame(
        {
            "ts": [
                "2017-09-13 21:47:47.498520+00",
                "2017-09-13 21:47:47.598520+00",
                "2017-09-13 21:47:47.698520+00",
                "2017-09-13 21:47:47.798520+00",
                "2017-09-13 21:47:47.898520+00",
                "2017-09-13 21:47:47.998520+00",
            ],
            "city": ["a", "b", "a", "b", "a", "b"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    result = run_validation(
        model=RandomForestClassifier(n_estimators=5, random_state=42),
        X=X,
        y=y,
        strategy={"type": "kfold", "n_splits": 3},
        metric_fn=accuracy_score,
        task="classification",
    )

    assert "cv_mean" in result
    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 3
