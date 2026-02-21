import json
import pandas as pd
from pathlib import Path

from .utils import now_iso, ensure_parent
from .context import PreprocessContext
from .dispatcher import run_strategies


MODEL_SELECTION_LOG = Path("data/model_selection.jsonl")


def load_last_n_records(n):

    records = []

    with open(MODEL_SELECTION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records[-n:]


def process_one(record):

    ctx = PreprocessContext(record)
    df = pd.read_csv(ctx.dataset_path)

    df, strategy_logs = run_strategies(df, ctx)

    final_path = ctx.dataset_path.with_name(
        ctx.dataset_path.stem + "_final.csv"
    )

    ensure_parent(final_path)
    df.to_csv(final_path, index=False)

    log = {
        "timestamp": now_iso(),
        "primary_model": ctx.primary_model,
        "final_dataset": str(final_path),
        "strategy_logs": strategy_logs,
        "rows": len(df),
        "columns": len(df.columns)
    }

    log_path = final_path.with_suffix(".json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return {
        "dataset": str(final_path),
        "log": str(log_path)
    }


def run_preprocess_2(last_n: int = 1):

    records = load_last_n_records(last_n)

    outputs = []

    for rec in records:
        outputs.append(process_one(rec))

    return outputs