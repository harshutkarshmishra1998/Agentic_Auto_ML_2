from pathlib import Path
from datetime import datetime
import uuid

def get_first_existing(d: dict, keys: list, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def dataset_name(path):
    return Path(path).stem


def artifact_name(path):
    return Path(path).name


def new_evaluation_id():
    return "eval_" + uuid.uuid4().hex[:12]


def utc_now():
    return datetime.utcnow().isoformat()