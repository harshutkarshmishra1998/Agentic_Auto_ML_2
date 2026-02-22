import json
from pathlib import Path

from .signal_extractor import extract_signals
from .rules_engine import MODEL_RULES


def _get_project_root():
    return Path(__file__).resolve().parents[1]


def _load_json_records(path: Path):
    """
    Load records from a JSON file that may contain:
    - a single JSON array
    - newline-delimited JSON objects
    - multiple top-level JSON payloads concatenated over time
    """

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    decoder = json.JSONDecoder()
    idx = 0
    chunks = []

    while idx < len(raw):
        while idx < len(raw) and raw[idx].isspace():
            idx += 1

        if idx >= len(raw):
            break

        obj, end = decoder.raw_decode(raw, idx)
        chunks.append(obj)
        idx = end

    if len(chunks) == 1 and isinstance(chunks[0], list):
        return chunks[0]

    records = []
    for chunk in chunks:
        if isinstance(chunk, list):
            records.extend(chunk)
        else:
            records.append(chunk)

    return records

# def save_results(results):

#     root = _get_project_root()
#     out = root / "data" / "model_initializations.json"

#     with open(out, "w") as f:
#         json.dump(results, f, indent=2)

#     return out

def save_results(results):

    root = _get_project_root()
    data_dir = root / "data"
    out = data_dir / "model_initialization.jsonl"

    data_dir.mkdir(parents=True, exist_ok=True)

    with open(out, "a") as f:   # ← append mode
        for r in results:
            f.write(json.dumps(r))
            f.write("\n")

    return out

def _load_preprocess_log():
    root = _get_project_root()
    path = root / "data" / "preprocess_2.jsonl"

    if not path.exists():
        # backward compatibility for older runs
        path = root / "data" / "preprocess_2.json"

    if not path.exists():
        raise FileNotFoundError("Missing preprocess log: expected data/preprocess_2.jsonl (or legacy data/preprocess_2.json)")

    return _load_json_records(path)


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

        record = {
            "experiment_id": exp["experiment_id"],
            "model": model,
            "init_params": params,
            "signals": signals.__dict__,
            "dataset_path": exp["final_dataset"]
        }

        save_path = save_results([record])
        print(f"Saved to {save_path}")

        results.append(record)

    return results
