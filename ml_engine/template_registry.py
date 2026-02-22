from ml_engine.templates.random_forest import RandomForestTemplate
from ml_engine.templates.xgboost import XGBoostTemplate
from ml_engine.templates.lightgbm import LightGBMTemplate
from ml_engine.templates.logistic_regression import LogisticTemplate
from ml_engine.templates.linear_regression import LinearTemplate
from ml_engine.templates.kmeans import KMeansTemplate
from ml_engine.templates.dbscan import DBSCANTemplate


TEMPLATE_REGISTRY = {
    "random_forest": RandomForestTemplate,
    "xgboost": XGBoostTemplate,
    "lightgbm": LightGBMTemplate,
    "logistic_regression": LogisticTemplate,
    "linear_regression": LinearTemplate,
    "kmeans": KMeansTemplate,
    "dbscan": DBSCANTemplate,
}