import json
import shutil
from pathlib import Path
from datetime import datetime


# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARSER_DATA_DIR = PROJECT_ROOT / "parser" / "data"
CSV_FILE = PARSER_DATA_DIR/"csv"
JOBLIB_FILE=PARSER_DATA_DIR/"joblib"
JSONL_FILE = DATA_DIR / "ml_experiments.jsonl"


# -----------------------------
# HELPERS
# -----------------------------
def parse_timestamp(ts: str):
    """
    Safely parse timestamp.
    Supports ISO format and common datetime formats.
    """
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            raise ValueError(f"Unrecognized timestamp format: {ts}")


def load_latest_experiments(jsonl_path: Path):
    """
    Load JSONL and return dict of latest experiment per experiment_id.
    """
    experiments = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            exp_id = obj.get("experiment_id")
            ts = obj.get("timestamp")

            if not exp_id or not ts:
                continue

            current_ts = parse_timestamp(ts)

            if exp_id not in experiments:
                experiments[exp_id] = (current_ts, obj)
            else:
                stored_ts, _ = experiments[exp_id]
                if current_ts > stored_ts:
                    experiments[exp_id] = (current_ts, obj)

    return {k: v[1] for k, v in experiments.items()}


def copy_file(src_path: Path, dest_dir: Path):
    if not src_path.exists():
        print(f"⚠ Missing: {src_path}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_dir / src_path.name)
    print(f"✔ Copied: {src_path.name}")


def copy_joblib_from_dir(directory: Path, dest_dir: Path):
    if not directory.exists():
        print(f"⚠ Artifact directory missing: {directory}")
        return

    for file in directory.rglob("*.joblib"):
        shutil.copy2(file, dest_dir / file.name)
        print(f"✔ Copied joblib: {file.name}")


# -----------------------------
# MAIN
# -----------------------------
def dataset_extractor():
    if not JSONL_FILE.exists():
        raise FileNotFoundError(f"JSONL not found: {JSONL_FILE}")
    
    CSV_FILE.mkdir(parents=True, exist_ok=True)
    JOBLIB_FILE.mkdir(parents=True, exist_ok=True)

    latest_experiments = load_latest_experiments(JSONL_FILE)

    for exp_id, exp in latest_experiments.items():
        print(f"\nProcessing experiment: {exp_id}")

        dataset_path = exp.get("dataset_path")
        artifact_path = exp.get("artifact", {}).get("path")

        if dataset_path:
            copy_file(Path(dataset_path), CSV_FILE)

        if artifact_path:
            artifact_path = Path(artifact_path)

            # If artifact_path is a file → copy directly
            if artifact_path.is_file():
                copy_file(artifact_path, JOBLIB_FILE)

            # If directory → copy joblib files
            elif artifact_path.is_dir():
                copy_joblib_from_dir(artifact_path, JOBLIB_FILE)

    print("\nDone.")


if __name__ == "__main__":
    dataset_extractor()