import json
from pathlib import Path

from .signal_extractor import extract_signals
from .rules_engine import MODEL_RULES


def _get_project_root():
    return Path(__file__).resolve().parents[1]

def save_results(results):

    root = _get_project_root()
    out = root / "data" / "model_initializations.json"

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    return out

def _load_preprocess_log():
    root = _get_project_root()
    path = root / "data" / "preprocess_2.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing preprocess log: {path}")

    with open(path) as f:
        return json.load(f)


def run_initializer(last_n: int = 1):
    """
    Main public function.

    Parameters
    ----------
    last_n : int
        Number of latest experiments to initialize models for.
    """

    logs = _load_preprocess_log()

    if last_n <= 0:
        raise ValueError("last_n must be > 0")

    selected = logs[-last_n:]

    results = []

    for exp in selected:

        model = exp["primary_model"]

        if model not in MODEL_RULES:
            raise ValueError(f"No initializer defined for model: {model}")

        signals = extract_signals(exp)
        params = MODEL_RULES[model](signals)

        results.append({
            "experiment_id": exp["experiment_id"],
            "model": model,
            "init_params": params,
            "signals": signals.__dict__,
            "dataset_path": exp["final_dataset"]
        })

        save_path = save_results(results)
        print(f"Saved to {save_path}")

    return results