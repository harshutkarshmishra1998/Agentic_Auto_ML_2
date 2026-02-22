from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from .base import ModelTemplate


class LinearTemplate(ModelTemplate):

    def build(self):
        self.model = LinearRegression(**self.config["init"])

    def metric(self, y_true, y_pred):
        return r2_score(y_true, y_pred)