import re
from pathlib import Path


# -------------------------------------------------
# MODEL NAME ALIAS MAP (must mirror model_rules)
# -------------------------------------------------

MODEL_NAME_ALIASES = {
    "randomforestclassifier": "random_forest",
    "randomforestregressor": "random_forest",
    "random forest": "random_forest",
    "random forest classifier": "random_forest",
    "random forest regressor": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",

    "xgboostclassifier": "xgboost",
    "xgboostregressor": "xgboost",
    "xgboost classifier": "xgboost",
    "xgboost regressor": "xgboost",
    "xgboost": "xgboost",
    "xgb": "xgboost",

    "lightgbmclassifier": "lightgbm",
    "lightgbmregressor": "lightgbm",
    "lightgbm classifier": "lightgbm",
    "lightgbm regressor": "lightgbm",
    "lightgbm": "lightgbm",

    "logisticregression": "logistic_regression",
    "linearregression": "linear_regression",

    "kmeans": "kmeans",
    "k-means": "kmeans",

    "dbscan": "dbscan",
    "hierarchicalclustering": "dbscan",
}

GENERIC_MODEL_TOKENS = {
    "model",
    "estimator",
    "algorithm",
    "classifier",
    "regressor",
    "classification",
    "regression",
}


def _sanitize(name: str | None) -> str | None:
    if not name:
        return None
    n = name.lower().strip()
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def canonical(name: str | None):
    sanitized = _sanitize(name)
    if not sanitized:
        return None

    if sanitized in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[sanitized]

    squashed = sanitized.replace(" ", "")
    if squashed in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[squashed]

    squashed = re.sub(r"(classifier|regressor)$", "", squashed)
    if squashed in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[squashed]

    canonical_keys = {
        "randomforest": "random_forest",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "logisticregression": "logistic_regression",
        "linearregression": "linear_regression",
        "kmeans": "kmeans",
        "dbscan": "dbscan",
    }
    if squashed in canonical_keys:
        return canonical_keys[squashed]

    return squashed


def _model_tokens(name: str | None) -> set[str]:
    sanitized = _sanitize(name)
    if not sanitized:
        return set()

    tokens = set(sanitized.split(" "))
    compact = "".join(tokens)

    # Add decomposed tokens from known compact names.
    if "randomforest" in compact:
        tokens.update({"random", "forest"})
    if "xgboost" in compact or "xgb" in compact:
        tokens.add("xgboost")
    if "lightgbm" in compact:
        tokens.add("lightgbm")

    return {t for t in tokens if t and t not in GENERIC_MODEL_TOKENS}


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

        # 1) Exact canonical match.
        for llm_key in self.model_requirements.keys():
            if canonical(llm_key) == target:
                return llm_key

        # 2) Token-overlap fallback so unseen variants still resolve.
        target_tokens = _model_tokens(self.primary_model)
        best_key = None
        best_score = 0.0

        for llm_key in self.model_requirements.keys():
            llm_tokens = _model_tokens(llm_key)
            if not llm_tokens or not target_tokens:
                continue

            overlap = len(target_tokens & llm_tokens)
            union = len(target_tokens | llm_tokens)
            score = overlap / union if union else 0.0

            if score > best_score:
                best_score = score
                best_key = llm_key

        # conservative threshold: require meaningful overlap.
        if best_key and best_score >= 0.5:
            return best_key

        # ---------- DEBUGGING INFO ----------
        available = list(self.model_requirements.keys())

        raise ValueError(
            f"Model preprocessing requirements not found.\n"
            f"Primary model: {self.primary_model}\n"
            f"Canonical: {target}\n"
            f"Target tokens: {sorted(target_tokens)}\n"
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
