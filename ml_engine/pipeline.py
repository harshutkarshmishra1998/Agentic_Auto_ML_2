import json
import time
from pathlib import Path

from ml_engine.template_registry import TEMPLATE_REGISTRY
from ml_engine.data_loader import load_dataset
from ml_engine.logger import log_experiment
from ml_engine.artifacts import save_model_artifact
from ml_engine.user_input_loader import load_last_n_targets


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def _load_initializations():
    path = _get_project_root() / "data" / "model_initializations.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def _infer_task(target):
    return "clustering" if target is None else "classification"


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
        task = _infer_task(target_column)

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
            training_time_sec=training_time
        )

        results.append({
            "experiment_id": experiment_id,
            "artifact_path": artifact_path
        })

    return results