from sklearn.cluster import KMeans
# from ml_engine.evaluator import clustering_metrics
from .base import ModelTemplate


class KMeansTemplate(ModelTemplate):

    def build(self):
        self.model = KMeans(**self.config["init"])

    # def metric(self, X, labels):
    #     return clustering_metrics(X, labels)