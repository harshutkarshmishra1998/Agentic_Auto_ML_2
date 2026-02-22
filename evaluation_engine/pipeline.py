from .experiment_loader import load_last_n_experiments
from .metric_computation import derive_metrics
from .retraining_llm import analyze_retraining_need
from .evaluation_logger import append_evaluation
from .utils import (
    dataset_name,
    artifact_name,
    new_evaluation_id,
    utc_now,
)

from .utils import get_first_existing

def _extract_training_metrics(exp: dict):
    """
    Resolve metrics field from experiment record.
    Supports multiple possible schema variants.
    """

    candidate_keys = [
        "training_result",
        "metrics",
        "results",
        "model_results",
        "evaluation",
        "performance",
    ]

    for key in candidate_keys:
        if key in exp and exp[key] is not None:
            return exp[key]

    raise ValueError(
        f"Cannot find metrics block in experiment record. "
        f"Available keys: {list(exp.keys())}"
    )


def run_evaluation_pipeline(last_n: int):

    experiments = load_last_n_experiments(last_n)
    evaluations = []

    for exp in experiments:

        raw_metrics = get_first_existing(
            exp,
            ["training_result", "metrics", "result", "training_metrics"],
            {}
        )

        model = get_first_existing(
            exp,
            ["model_name", "model", "estimator"]
        )

        dataset_path = get_first_existing(
            exp,
            ["dataset_path", "data_path", "dataset"]
        )

        artifact_path = get_first_existing(
            exp,
            ["model_artifact_path", "artifact_path", "model_path"]
        )

        training_time = get_first_existing(
            exp,
            ["training_time_sec", "training_time", "fit_time"],
            None
        )

        derived = derive_metrics(raw_metrics)

        # train_score = raw_metrics.get("train_score")
        # val_score = raw_metrics.get("validation_score")

        train_score = None
        val_score = None

        if isinstance(raw_metrics, dict):

            if raw_metrics.get("classification"):
                block = raw_metrics["classification"]

            elif raw_metrics.get("regression"):
                block = raw_metrics["regression"]

            else:
                block = None

            if block:
                train_score = block.get("train_score")
                val_score = block.get("validation_score")

        gap = None
        if train_score is not None and val_score is not None:
            gap = train_score - val_score

        llm_decision = analyze_retraining_need({
            "model": model,
            "task": exp.get("task"),
            "metrics": derived,
            "training_time": training_time,
            "validation_gap": gap,
            "validation_strategy": exp.get("validation_strategy", {}),
        })

        record = {
            "evaluation_id": new_evaluation_id(),
            "timestamp": utc_now(),

            "experiment_id": exp.get("experiment_id"),

            "model": model,
            "task": exp.get("task"),

            "dataset": {
                "name": dataset_name(dataset_path) if dataset_path else None,
                "path": dataset_path
            },

            "artifact": {
                "filename": artifact_name(artifact_path) if artifact_path else None,
                "path": artifact_path
            },

            "training_time_sec": training_time,

            "metrics": {
                "raw": raw_metrics,
                "derived": derived
            },

            "retraining_analysis": llm_decision
        }
        append_evaluation(record)
        evaluations.append(record)

    return evaluations