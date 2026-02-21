import json
from pathlib import Path


class PreprocessContext:

    def __init__(self, record: dict):

        self.record = record

        self.dataset_path = Path(record["dataset"]["preprocessed_path"])
        self.primary_model = record["model_selection"]["primary_model"]

        if not self.primary_model:
            raise ValueError("Primary model missing")

        self.deferred = record["preprocessing"]["deferred"]
        self.model_requirements = record["llm_analysis"]["model_dependent_preprocessing"]

        self.model_key = self._match_model_key()

    def _normalize(self, x):
        return x.lower().replace("_", "").replace("-", "")

    def _match_model_key(self):
        for k in self.model_requirements:
            if self._normalize(k) == self._normalize(self.primary_model):
                return k
        raise ValueError("Model preprocessing requirements not found")

    def get_required_model_prep(self):
        return self.model_requirements[self.model_key]

    def get_columns(self, strategy):
        return [
            x["column"]
            for x in self.deferred
            if x["strategy"] == strategy
        ]