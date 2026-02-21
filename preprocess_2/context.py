# import json
# from pathlib import Path


# class PreprocessContext:

#     def __init__(self, record: dict):

#         self.record = record

#         self.dataset_path = Path(record["dataset"]["preprocessed_path"])
#         self.primary_model = record["model_selection"]["primary_model"]

#         if not self.primary_model:
#             raise ValueError("Primary model missing")

#         self.deferred = record["preprocessing"]["deferred"]
#         self.model_requirements = record["llm_analysis"]["model_dependent_preprocessing"]

#         self.model_key = self._match_model_key()

#     def _normalize(self, x):
#         return x.lower().replace("_", "").replace("-", "")

#     def _match_model_key(self):
#         for k in self.model_requirements:
#             if self._normalize(k) == self._normalize(self.primary_model):
#                 return k
#         raise ValueError("Model preprocessing requirements not found")

#     def get_required_model_prep(self):
#         return self.model_requirements[self.model_key]

#     def get_columns(self, strategy):
#         return [
#             x["column"]
#             for x in self.deferred
#             if x["strategy"] == strategy
#         ]

import json
from pathlib import Path


# -------------------------------------------------
# MODEL NAME ALIAS MAP (must mirror model_rules)
# -------------------------------------------------

MODEL_NAME_ALIASES = {
    "randomforestclassifier": "random_forest",
    "random forest": "random_forest",
    "randomforest": "random_forest",

    "xgboostclassifier": "xgboost",
    "xgboost": "xgboost",

    "lightgbmclassifier": "lightgbm",
    "lightgbm": "lightgbm",

    "logisticregression": "logistic_regression",

    "kmeans": "kmeans",
    "k-means": "kmeans",

    "dbscan": "dbscan",
    "hierarchicalclustering": "dbscan"
}


def canonical(name: str | None):
    if not name:
        return None

    n = name.lower().replace("_", "").replace("-", "").strip()

    if n in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[n]

    return n


# -------------------------------------------------
# CONTEXT
# -------------------------------------------------

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

    # -------------------------------------------------

    def _match_model_key(self):
        """
        Match canonical primary model to LLM key.
        """

        target = canonical(self.primary_model)

        for llm_key in self.model_requirements.keys():
            if canonical(llm_key) == target:
                return llm_key

        # ---------- DEBUGGING INFO ----------
        available = list(self.model_requirements.keys())

        raise ValueError(
            f"Model preprocessing requirements not found.\n"
            f"Primary model: {self.primary_model}\n"
            f"Canonical: {target}\n"
            f"Available LLM keys: {available}"
        )

    # -------------------------------------------------

    def get_required_model_prep(self):
        return self.model_requirements[self.model_key]

    # -------------------------------------------------

    def get_columns(self, strategy):
        return [
            x["column"]
            for x in self.deferred
            if x["strategy"] == strategy
        ]