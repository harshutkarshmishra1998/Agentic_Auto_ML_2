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
