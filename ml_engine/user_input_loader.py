import json
from pathlib import Path


# --------------------------------------------------
# helpers
# --------------------------------------------------

def _get_project_root():
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------
# public loader
# --------------------------------------------------

def load_last_n_targets(last_n: int):
    """
    Load target columns aligned with last n user inputs.

    Each row in user_inputs.jsonl is expected to contain:
        {
            "target_columns": ["col_name"]   # or []
        }

    Returns
    -------
    List[str | None]
        Target column names.
        If no target provided → None (clustering case).
    """

    if last_n <= 0:
        raise ValueError("last_n must be > 0")

    path = _get_project_root() / "data" / "user_input.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"user_input.jsonl not found at: {path}")

    # read all rows
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError("user_inputs.jsonl is empty")

    selected = rows[-last_n:]

    targets = []
    for r in selected:
        cols = r.get("target_columns", [])
        targets.append(cols[0] if cols else None)

    return targets