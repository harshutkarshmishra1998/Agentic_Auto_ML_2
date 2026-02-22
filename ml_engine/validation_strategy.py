import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error


def detect_imbalance(y):
    if y is None:
        return False
    _, counts = np.unique(y, return_counts=True)
    return counts.min() / counts.max() < 0.2




def _can_use_stratified_kfold(y, n_splits):
    """Return True only when y is a valid classification target for stratification."""
    if y is None:
        return False

    y_type = type_of_target(y)
    if y_type not in {"binary", "multiclass"}:
        return False

    _, counts = np.unique(y, return_counts=True)
    # StratifiedKFold requires at least n_splits members in every class.
    return counts.min() >= n_splits

def choose_strategy(X, y, task):

    n = len(X)

    if task == "clustering":
        return {"type": "fit_all"}

    if n < 200:
        return {"type": "loo"}

    if n < 2000:
        if task == "classification" and detect_imbalance(y) and _can_use_stratified_kfold(y, 5):
            return {"type": "stratified_kfold", "n_splits": 5}
        return {"type": "kfold", "n_splits": 5}

    return {"type": "holdout"}


def _prepare_features(X):
    """
    Convert all feature columns to model-safe numeric values.

    - Low-cardinality categorical columns are one-hot encoded.
    - High-cardinality categorical columns are ordinal-encoded.
    - Numeric columns are retained as-is.
    """
    if X is None or X.shape[1] == 0:
        raise ValueError("Training requires at least one feature column.")

    max_categories_for_one_hot = 100

    low_cardinality_cols = []
    high_cardinality_cols = []

    for col in X.columns:
        dtype = X[col].dtype
        is_categorical = (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        if is_categorical:
            if X[col].nunique(dropna=True) > max_categories_for_one_hot:
                high_cardinality_cols.append(col)
            else:
                low_cardinality_cols.append(col)

    numeric_cols = [
        col for col in X.columns
        if col not in low_cardinality_cols and col not in high_cardinality_cols
    ]

    feature_parts = [X[numeric_cols].copy()]

    for col in high_cardinality_cols:
        encoded = pd.factorize(X[col], sort=False)[0].astype("int32")
        feature_parts.append(pd.DataFrame({f"{col}__encoded": encoded}, index=X.index))

    if low_cardinality_cols:
        feature_parts.append(pd.get_dummies(X[low_cardinality_cols], dummy_na=True))

    X_prepared = pd.concat(feature_parts, axis=1)

    if X_prepared.shape[1] == 0:
        raise ValueError("Training requires at least one usable feature column.")

    return X_prepared


def run_validation(model, X, y, strategy, metric_fn, task):
    X = _prepare_features(X)

    # -----------------------------
    # clustering
    # -----------------------------
    if strategy["type"] == "fit_all":
        X_numeric = X.select_dtypes(include=["number"])

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "Clustering requires numeric features. "
                "No numeric columns found after preprocessing."
            )

        model.fit(X_numeric)

        labels = getattr(model, "labels_", None)

        return {
            "status": "trained_full_dataset",
            "n_clusters": len(set(labels)) if labels is not None else None,
            "noise_ratio": float((labels == -1).sum() / len(labels))
            if labels is not None and -1 in labels else 0.0,
            "n_numeric_features_used": int(X_numeric.shape[1])
        }

    # -----------------------------
    # holdout
    # -----------------------------
    if strategy["type"] == "holdout":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(Xtr, ytr)

        train_pred = model.predict(Xtr)
        val_pred = model.predict(Xte)

        if task == "classification":
            cm = confusion_matrix(yte, val_pred)
            return {
                "train_score": float(metric_fn(ytr, train_pred)),
                "validation_score": float(metric_fn(yte, val_pred)),
                "confusion_matrix": cm.tolist()
            }

        if task == "regression":
            return {
                "train_score": float(metric_fn(ytr, train_pred)),
                "validation_score": float(metric_fn(yte, val_pred)),
                "mae": float(mean_absolute_error(yte, val_pred)),
                "rmse": float(np.sqrt(mean_squared_error(yte, val_pred)))
            }

    # -----------------------------
    # cross validation
    # -----------------------------
    if strategy["type"] == "loo":
        splitter = LeaveOneOut()
    elif strategy["type"] == "kfold":
        splitter = KFold(strategy["n_splits"], shuffle=True, random_state=42)
    elif strategy["type"] == "stratified_kfold" and _can_use_stratified_kfold(y, strategy["n_splits"]):
        splitter = StratifiedKFold(strategy["n_splits"], shuffle=True, random_state=42)
    else:
        # Safe fallback prevents crashes when target/task metadata is inconsistent.
        splitter = KFold(strategy.get("n_splits", 5), shuffle=True, random_state=42)

    scores = []

    for tr, te in splitter.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        scores.append(metric_fn(y.iloc[te], preds))

    return {
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "fold_scores": [float(s) for s in scores]
    }
