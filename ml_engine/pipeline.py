import json
import time
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ml_engine.template_registry import TEMPLATE_REGISTRY
from ml_engine.data_loader import load_dataset
from ml_engine.logger import log_experiment
from ml_engine.artifacts import save_model_artifact
from ml_engine.user_input_loader import load_last_n_targets


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def _load_initializations():
    path = _get_project_root() / "data" / "model_initialization.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def _normalize_task(task):
    if not task:
        return None

    t = str(task).strip().lower()
    aliases = {
        "classification": "classification",
        "binary_classification": "classification",
        "multiclass_classification": "classification",
        "regression": "regression",
        "unsupervised": "clustering",
        "clustering": "clustering",
    }
    return aliases.get(t)


def _infer_task(target_column, y):
    """
    Infer task from target availability and target distribution.

    Why this exists:
    - Upstream metadata can occasionally drift.
    - Integer-valued regression targets (e.g., count/hour labels) are often misread as multiclass.
    """
    if target_column is None or y is None:
        return "clustering"

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    y_non_null = y.dropna()
    if y_non_null.empty:
        return "classification"

    if is_numeric_dtype(y_non_null):
        n = len(y_non_null)
        unique_count = int(y_non_null.nunique())
        unique_ratio = unique_count / max(n, 1)

        # Heuristic for numeric targets:
        # - very low cardinality numeric labels are usually classes (0/1 etc.)
        # - many distinct numeric values (or sparse repeated counts at scale) are regression.
        if unique_count <= 10:
            return "classification"
        if unique_count > 50 or unique_ratio < 0.05:
            return "regression"

    return "classification"


def _resolve_task(task_from_upstream, inferred_task, target_column, y):
    """
    Resolve task with lightweight safety overrides.

    Upstream metadata is preferred, but we guard against known incompatible
    combinations (for example regression selected while target is textual).
    """
    # No explicit target means unsupervised training.
    if target_column is None:
        return "clustering"

    # Missing upstream task: trust local inference.
    if not task_from_upstream:
        return inferred_task

    # Regression models require numeric target values.
    # If target is non-numeric, force classification to avoid runtime crashes.
    if task_from_upstream == "regression":
        y_non_null = y.dropna() if isinstance(y, pd.Series) else pd.Series(y).dropna()
        if not y_non_null.empty and not is_numeric_dtype(y_non_null):
            return "classification"

    return task_from_upstream


def run_training(last_n=1):

    init_runs = _load_initializations()[-last_n:]
    targets = load_last_n_targets(last_n)

    results = []

    for init_record, target_column in zip(init_runs, targets):

        model_name = init_record["model"]
        init_params = init_record["init_params"]
        dataset_path = init_record["dataset_path"]
        experiment_id = init_record["experiment_id"]

        Template = TEMPLATE_REGISTRY[model_name]

        X, y = load_dataset(dataset_path, target_column)

        task_from_upstream = _normalize_task(init_record.get("task"))
        inferred_task = _infer_task(target_column, y)

        task = _resolve_task(
            task_from_upstream=task_from_upstream,
            inferred_task=inferred_task,
            target_column=target_column,
            y=y,
        )

        template = Template({
            "init": init_params,
            "task": task
        })

        # -----------------------------
        # measure training time
        # -----------------------------
        start = time.perf_counter()
        output = template.run(X, y)
        training_time = time.perf_counter() - start

        model = output["model"]

        # -----------------------------
        # save artifact
        # -----------------------------
        artifact_path = save_model_artifact(
            model=model,
            dataset_path=dataset_path,
            experiment_id=experiment_id
        )

        # -----------------------------
        # log experiment
        # -----------------------------
        log_experiment(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_path=dataset_path,
            target_column=target_column,
            task=task,
            init_params=init_params,
            validation_strategy=output["validation_strategy"],
            training_result=output["result"],
            model_artifact_path=artifact_path,
            training_time_sec=training_time,
            runtime_model=model
        )

        results.append({
            "experiment_id": experiment_id,
            "artifact_path": artifact_path
        })

    return results
