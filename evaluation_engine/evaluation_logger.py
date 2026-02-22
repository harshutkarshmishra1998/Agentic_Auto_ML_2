import json
from pathlib import Path


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def append_evaluation(record: dict):
    path = _get_project_root() / "data" / "evaluation.jsonl"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")