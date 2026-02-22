from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from .base import ModelTemplate


class XGBoostTemplate(ModelTemplate):

    def build(self):
        if self.config["task"] == "regression":
            self.model = XGBRegressor(**self.config["init"])
        else:
            self.model = XGBClassifier(**self.config["init"])

    def metric(self, y_true, y_pred):
        if self.config["task"] == "regression":
            return r2_score(y_true, y_pred)
        return accuracy_score(y_true, y_pred)