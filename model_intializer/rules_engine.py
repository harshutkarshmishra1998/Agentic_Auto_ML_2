from .model_defaults import *


MODEL_RULES = {
    "random_forest": init_random_forest,
    "xgboost": init_boosting,
    "lightgbm": init_boosting,
    "logistic_regression": init_linear,
    "linear_regression": lambda s: {},
    "kmeans": init_kmeans,
    "dbscan": init_dbscan
}