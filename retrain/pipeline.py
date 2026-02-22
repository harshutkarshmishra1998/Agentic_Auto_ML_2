import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation_engine.pipeline import run_evaluation_pipeline
from ml_engine.pipeline import run_training
from model_intializer.pipeline import run_initializer


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _append_preprocess_2_seed_entry(experiment_id: str, dataset_path: str, primary_model: str) -> None:
    path = _get_project_root() / "data" / "preprocess_2.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = []

    payload.append(
        {
            "experiment_id": experiment_id,
            "final_dataset": dataset_path,
            "primary_model": primary_model,
            "strategy_logs": [],
            "rows": None,
            "columns": None,
            "trigger": "retrain_loop",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_best_model(experiment_id: str) -> Optional[str]:
    path = _get_project_root() / "data" / "model_selection.jsonl"
    records = _read_jsonl(path)

    for record in reversed(records):
        if record.get("experiment_id") == experiment_id:
            return record.get("model_selection", {}).get("primary_model")

    if records:
        return records[-1].get("model_selection", {}).get("primary_model")

    return None


def _adjust_init_params(
    model_name: str,
    init_params: Dict[str, Any],
    retraining_analysis: Dict[str, Any],
    round_idx: int,
) -> Dict[str, Any]:
    """
    Lightweight retraining-time hyperparameter adjustment.

    Strategy:
    - If overfitting suspected (positive validation gap), regularize.
    - Otherwise, gently increase search capacity across rounds.
    """
    params = dict(init_params)
    gap = retraining_analysis.get("validation_gap")

    if model_name == "random_forest":
        n_estimators = int(params.get("n_estimators", 200))
        params["n_estimators"] = min(1200, n_estimators + 100)

        if gap is not None and gap > 0.1:
            depth = params.get("max_depth")
            params["max_depth"] = 10 if depth is None else max(5, min(int(depth), 10))
            params["min_samples_leaf"] = max(2, int(params.get("min_samples_leaf", 1)) + 1)

    elif model_name in {"xgboost", "lightgbm"}:
        lr = float(params.get("learning_rate", 0.05))
        params["learning_rate"] = max(0.01, lr * 0.85)
        params["n_estimators"] = min(2000, int(params.get("n_estimators", 500)) + 150)

        if gap is not None and gap > 0.1:
            params["subsample"] = max(0.6, float(params.get("subsample", 1.0)) - 0.1)
            if "colsample_bytree" in params:
                params["colsample_bytree"] = max(0.6, float(params.get("colsample_bytree", 1.0)) - 0.1)

    elif model_name == "logistic_regression":
        c_val = float(params.get("C", 1.0))
        params["C"] = max(0.05, c_val * 0.75 if gap is not None and gap > 0.1 else c_val * 0.9)
        params["max_iter"] = max(int(params.get("max_iter", 2000)), 3000 + (round_idx * 500))

    elif model_name == "kmeans":
        clusters = int(params.get("n_clusters", 8))
        params["n_clusters"] = min(max(2, clusters + 1), clusters + 3)
        params["n_init"] = max(20, int(params.get("n_init", 20)) + 5)

    elif model_name == "dbscan":
        eps = float(params.get("eps", 0.5))
        params["eps"] = max(0.1, eps * 0.95)
        params["min_samples"] = max(5, int(params.get("min_samples", 5)) + 1)

    return params


def _append_adjusted_initialization(
    init_record: Dict[str, Any],
    retraining_analysis: Dict[str, Any],
    round_idx: int,
) -> Dict[str, Any]:
    adjusted = dict(init_record)
    adjusted["init_params"] = _adjust_init_params(
        model_name=adjusted.get("model"),
        init_params=adjusted.get("init_params", {}),
        retraining_analysis=retraining_analysis,
        round_idx=round_idx,
    )
    adjusted["retrain_metadata"] = {
        "trigger": "retrain_loop",
        "round": round_idx,
        "source_evaluation_id": retraining_analysis.get("evaluation_id"),
    }

    path = _get_project_root() / "data" / "model_initialization.jsonl"
    _append_jsonl(path, adjusted)
    return adjusted


def run_retraining_loop(max_rounds: int = 5) -> Dict[str, Any]:
    evaluations_path = _get_project_root() / "data" / "evaluation.jsonl"
    evaluations = _read_jsonl(evaluations_path)

    if not evaluations:
        return {"retrain_iterations": [], "retrain_status": "skipped_no_evaluation"}

    latest = evaluations[-1]
    analysis = latest.get("retraining_analysis", {})
    should_retrain = bool(analysis.get("should_retrain", False))

    iterations: List[Dict[str, Any]] = []
    round_idx = 0

    while should_retrain and round_idx < max_rounds:
        round_idx += 1

        experiment_id = latest.get("experiment_id")
        dataset_path = latest.get("dataset", {}).get("path")
        best_model = _resolve_best_model(experiment_id)

        if not experiment_id or not dataset_path or not best_model:
            break

        _append_preprocess_2_seed_entry(
            experiment_id=experiment_id,
            dataset_path=dataset_path,
            primary_model=best_model,
        )

        init_result = run_initializer(last_n=1)
        adjusted_init = None
        if init_result:
            analysis_context = dict(analysis)
            analysis_context["evaluation_id"] = latest.get("evaluation_id")
            adjusted_init = _append_adjusted_initialization(
                init_record=init_result[-1],
                retraining_analysis=analysis_context,
                round_idx=round_idx,
            )

        training_result = run_training(last_n=1)
        evaluation_result = run_evaluation_pipeline(last_n=1)

        latest = evaluation_result[-1] if evaluation_result else latest
        analysis = latest.get("retraining_analysis", {})
        should_retrain = bool(analysis.get("should_retrain", False))

        iterations.append(
            {
                "round": round_idx,
                "best_model": best_model,
                "initialization": init_result,
                "adjusted_initialization": adjusted_init,
                "training": training_result,
                "evaluation": evaluation_result,
            }
        )

    status = "completed"
    if should_retrain and round_idx >= max_rounds:
        status = "stopped_max_rounds"

    return {
        "retrain_iterations": iterations,
        "retrain_status": status,
        "final_should_retrain": should_retrain,
    }
