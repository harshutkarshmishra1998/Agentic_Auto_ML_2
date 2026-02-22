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
    training_time_sec=None
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
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),

        # -------------------------
        # model identity
        # -------------------------
        "model": model_name,
        "task": task,

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
            "model_name": model_name,
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