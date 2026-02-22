# import numpy as np
# from sklearn.model_selection import (
#     train_test_split,
#     KFold,
#     StratifiedKFold,
#     LeaveOneOut
# )


# def detect_imbalance(y):
#     if y is None:
#         return False
#     values, counts = np.unique(y, return_counts=True)
#     ratio = counts.min() / counts.max()
#     return ratio < 0.2


# def choose_strategy(X, y, task):

#     n = len(X)

#     if task == "clustering":
#         return {"type": "fit_all"}

#     if n < 200:
#         return {"type": "loo"}

#     if n < 2000:
#         if task == "classification" and detect_imbalance(y):
#             return {"type": "stratified_kfold", "n_splits": 5}
#         return {"type": "kfold", "n_splits": 5}

#     if n < 20000:
#         if task == "classification":
#             return {"type": "stratified_kfold", "n_splits": 5}
#         return {"type": "kfold", "n_splits": 5}

#     return {"type": "holdout"}


# def run_validation(model, X, y, strategy, metric_fn, task):

#     if strategy["type"] == "fit_all":
#         labels = model.fit_predict(X)
#         return metric_fn(X, labels)

#     if strategy["type"] == "holdout":
#         Xtr, Xte, ytr, yte = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#         model.fit(Xtr, ytr)
#         preds = model.predict(Xte)
#         return metric_fn(yte, preds)

#     if strategy["type"] == "loo":
#         splitter = LeaveOneOut()
#     elif strategy["type"] == "kfold":
#         splitter = KFold(strategy["n_splits"], shuffle=True, random_state=42)
#     elif strategy["type"] == "stratified_kfold":
#         splitter = StratifiedKFold(strategy["n_splits"], shuffle=True, random_state=42)
#     else:
#         raise ValueError("Unknown strategy")

#     scores = []

#     for tr, te in splitter.split(X, y):
#         model.fit(X.iloc[tr], y.iloc[tr])
#         preds = model.predict(X.iloc[te])
#         scores.append(metric_fn(y.iloc[te], preds))

#     return {
#         "mean": float(np.mean(scores)),
#         "std": float(np.std(scores)),
#         "folds": len(scores)
#     }

import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error


def detect_imbalance(y):
    if y is None:
        return False
    _, counts = np.unique(y, return_counts=True)
    return counts.min() / counts.max() < 0.2


def choose_strategy(X, y, task):

    n = len(X)

    if task == "clustering":
        return {"type": "fit_all"}

    if n < 200:
        return {"type": "loo"}

    if n < 2000:
        if task == "classification" and detect_imbalance(y):
            return {"type": "stratified_kfold", "n_splits": 5}
        return {"type": "kfold", "n_splits": 5}

    return {"type": "holdout"}


def run_validation(model, X, y, strategy, metric_fn, task):

    # -----------------------------
    # clustering
    # -----------------------------
    # if strategy["type"] == "fit_all":
    #     model.fit(X)
    #     labels = getattr(model, "labels_", None)

    #     return {
    #         "status": "trained_full_dataset",
    #         "n_clusters": len(set(labels)) if labels is not None else None,
    #         "noise_ratio": float((labels == -1).sum() / len(labels)) if labels is not None and -1 in labels else 0.0
    #     }

    if strategy["type"] == "fit_all":
        # --------------------------------
        # Ensure numeric-only features
        # --------------------------------
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
    else:
        splitter = StratifiedKFold(strategy["n_splits"], shuffle=True, random_state=42)

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