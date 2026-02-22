import json
from pathlib import Path


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def load_last_n_experiments(last_n: int):
    path = _get_project_root() / "data" / "ml_experiments.jsonl"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    return records[-last_n:]