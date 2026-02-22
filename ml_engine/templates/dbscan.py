from sklearn.cluster import DBSCAN
# from ml_engine.evaluator import clustering_metrics
from .base import ModelTemplate


class DBSCANTemplate(ModelTemplate):

    def build(self):
        self.model = DBSCAN(**self.config["init"])

    # def metric(self, X, labels):
    #     return clustering_metrics(X, labels)