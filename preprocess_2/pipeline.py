import json
import pandas as pd
from pathlib import Path

from .utils import now_iso, ensure_parent
from .context import PreprocessContext
from .dispatcher import run_strategies


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_SELECTION_LOG = Path("data/model_selection.jsonl")
PREPROCESS_2_LOG = Path("data/preprocess_2.jsonl")


# -------------------------------------------------
# LOAD MODEL SELECTION RECORDS
# -------------------------------------------------

def load_last_n_records(n: int):

    records = []

    with open(MODEL_SELECTION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records[-n:]


# -------------------------------------------------
# APPEND TO GLOBAL LOG
# -------------------------------------------------

def append_preprocess_log(entry: dict):
    """
    Append entry to preprocess_2.jsonl safely.
    """

    ensure_parent(PREPROCESS_2_LOG)

    with open(PREPROCESS_2_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")


# -------------------------------------------------
# PROCESS SINGLE RECORD
# -------------------------------------------------

def process_one(record):

    ctx = PreprocessContext(record)

    df = pd.read_csv(ctx.dataset_path)

    df, strategy_logs = run_strategies(df, ctx)

    # ---------------- save dataset ----------------

    final_path = ctx.dataset_path.with_name(
        ctx.dataset_path.stem + "_final.csv"
    )

    ensure_parent(final_path)
    df.to_csv(final_path, index=False)

    # ---------------- build log entry ----------------

    log_entry = {
        "timestamp": now_iso(),
        "experiment_id": record.get("experiment_id"),
        "source_preprocessed_file": str(ctx.dataset_path),
        "final_dataset": str(final_path),
        "primary_model": ctx.primary_model,
        "problem_type": record.get("problem_definition", {}).get("canonical_type"),
        "strategy_logs": strategy_logs,
        "rows": len(df),
        "columns": len(df.columns)
    }

    append_preprocess_log(log_entry)

    return {
        "final_dataset": str(final_path)
    }


# -------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------

def run_preprocess_2(last_n: int = 1):

    records = load_last_n_records(last_n)

    outputs = []

    for rec in records:
        outputs.append(process_one(rec))

    return outputs