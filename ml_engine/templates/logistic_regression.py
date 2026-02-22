from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .base import ModelTemplate


class LogisticTemplate(ModelTemplate):

    def build(self):
        self.model = LogisticRegression(**self.config["init"])

    def metric(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)