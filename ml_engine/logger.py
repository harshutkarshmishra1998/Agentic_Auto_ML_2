import json
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# helpers
# --------------------------------------------------

def _get_project_root():
    return Path(__file__).resolve().parents[1]


def _dataset_hash(path: str) -> str:
    """
    File content hash for drift detection and reproducibility.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _dataset_profile(dataset_path, target_column, task):
    """
    Extract structural dataset info for evaluator + drift detection.
    """
    df = pd.read_csv(dataset_path)

    profile = {
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "dataset_hash": _dataset_hash(dataset_path)
    }

    if task == "classification" and target_column and target_column in df.columns:
        dist = df[target_column].value_counts(normalize=True).to_dict()
        profile["class_distribution"] = {
            str(k): float(v) for k, v in dist.items()
        }
    else:
        profile["class_distribution"] = None

    return profile


def _artifact_info(model_artifact_path):
    """
    Convert artifact path into structured metadata.
    """
    if not model_artifact_path:
        return None

    p = Path(model_artifact_path)

    return {
        "path": str(p),
        "directory": str(p.parent),
        "filename": p.name,
        "format": p.suffix.replace(".", "")
    }




def _runtime_model_metadata(model_name: str, task: str, runtime_model=None) -> dict:
    """Return explicit runtime model metadata without changing canonical model key."""
    estimator_class = runtime_model.__class__.__name__ if runtime_model is not None else None

    estimator_type = None
    if estimator_class:
        lowered = estimator_class.lower()
        if "regressor" in lowered:
            estimator_type = "regressor"
        elif "classifier" in lowered:
            estimator_type = "classifier"

    if estimator_type is None:
        if task == "regression":
            estimator_type = "regressor"
        elif task == "classification":
            estimator_type = "classifier"

    model_variant = f"{model_name}_{estimator_type}" if estimator_type else model_name

    return {
        "model_family": model_name,
        "model_variant": model_variant,
        "estimator_type": estimator_type,
        "estimator_class": estimator_class,
    }


def _write_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record))
        f.write("\n")


# --------------------------------------------------
# public logger
# --------------------------------------------------

def log_experiment(
    *,
    experiment_id,
    model_name,
    dataset_path,
    target_column,
    task,
    init_params,
    validation_strategy,
    training_result,
    model_artifact_path=None,
    training_time_sec=None,
    runtime_model=None
):
    """
    Writes evaluator-ready experiment record.

    This is a full experiment registry entry suitable for:
    - evaluation module
    - retraining decisions
    - drift detection
    - model comparison
    - lineage tracking
    """

    root = _get_project_root()
    log_path = root / "data" / "ml_experiments.jsonl"

    # -----------------------------
    # dataset structural info
    # -----------------------------
    profile = _dataset_profile(
        dataset_path,
        target_column,
        task
    )

    # -----------------------------
    # metrics container (unified schema)
    # -----------------------------
    metrics = {
        "classification": None,
        "regression": None,
        "clustering": None
    }

    if task == "classification":
        metrics["classification"] = training_result
    elif task == "regression":
        metrics["regression"] = training_result
    elif task == "clustering":
        metrics["clustering"] = training_result

    # -----------------------------
    # artifact metadata
    # -----------------------------
    artifact = _artifact_info(model_artifact_path)

    # -----------------------------
    # full experiment record
    # -----------------------------
    runtime_meta = _runtime_model_metadata(model_name, task, runtime_model)
    # Backward-compatible alias kept to avoid NameError in partially merged deployments.
    resolved_model = model_name

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),

        # -------------------------
        # model identity
        # -------------------------
        "model": resolved_model,
        "task": task,
        "runtime_model": runtime_meta,

        # -------------------------
        # dataset reference
        # -------------------------
        "dataset_path": dataset_path,
        "target_column": target_column,
        "dataset_profile": profile,

        # -------------------------
        # model initialization (FULL CONTEXT)
        # -------------------------
        "model_initialization": {
            "model_name": resolved_model,
            "init_params": init_params,
            "initializer_version": "v1"
        },

        # -------------------------
        # validation setup
        # -------------------------
        "validation_strategy": validation_strategy,

        # -------------------------
        # metrics unified by task
        # -------------------------
        "metrics": metrics,

        # -------------------------
        # runtime
        # -------------------------
        "training_time_sec": training_time_sec,

        # -------------------------
        # artifact lineage
        # -------------------------
        "artifact": artifact
    }

    _write_jsonl(log_path, record)

    return log_path