def init_random_forest(sig):

    if sig.is_large_data:
        n_estimators = 400
    elif sig.is_small_data:
        n_estimators = 150
    else:
        n_estimators = 250

    if sig.overfit_risk:
        max_depth = 10
        min_leaf = 5
    else:
        max_depth = None
        min_leaf = 1

    if sig.is_high_dim:
        max_features = "sqrt"
    else:
        max_features = None

    return dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=42
    )


def init_boosting(sig):

    if sig.is_small_data:
        lr = 0.03
        depth = 5
    elif sig.is_large_data:
        lr = 0.1
        depth = 8
    else:
        lr = 0.05
        depth = 6

    if sig.overfit_risk:
        subsample = 0.7
        colsample = 0.7
    else:
        subsample = 1.0
        colsample = 1.0

    return dict(
        n_estimators=500,
        learning_rate=lr,
        max_depth=depth,
        subsample=subsample,
        colsample_bytree=colsample,
        random_state=42
    )


def init_linear(sig):
    C = 0.3 if sig.is_high_dim else 1.0
    return dict(penalty="l2", C=C, max_iter=2000)


def init_kmeans(sig):

    if sig.rows < 10000:
        k = 5
    elif sig.rows < 50000:
        k = 8
    else:
        k = 12

    return dict(n_clusters=k, n_init=20, random_state=42)


def init_dbscan(sig):

    eps = 0.8 if sig.is_high_dim else 0.5
    min_samples = max(5, int(sig.rows * 0.005))

    return dict(eps=eps, min_samples=min_samples)