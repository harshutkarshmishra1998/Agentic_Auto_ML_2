from pathlib import Path
from datetime import datetime
import joblib


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def _safe_name(path: str):
    """
    Extract dataset filename without extension.
    """
    return Path(path).stem


def save_model_artifact(model, dataset_path: str, experiment_id: str):
    """
    Saves trained model to:

    data/artifacts/<dataset_name>/<experiment_id>_<timestamp>.joblib
    """

    root = _get_project_root()

    dataset_name = _safe_name(dataset_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    folder = root / "data" / "artifacts" / dataset_name
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_name}_{timestamp}.joblib"
    full_path = folder / filename

    joblib.dump(model, full_path)

    return str(full_path)