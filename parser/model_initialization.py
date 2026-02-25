# OK
import json
from pathlib import Path
from openpyxl import Workbook

from parser.excel_writer import sheet_key_value, sheet_table, _auto_adjust_column_width_2


INPUT = Path("data/model_initialization.jsonl")
OUTPUT = Path("parser/data/xlsx/model_initialization.xlsx")


# -----------------------------
# load jsonl
# -----------------------------
def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# -----------------------------
# parse record
# -----------------------------
def parse_record(wb, rec, idx):

    prefix = f"record_{idx}_"

    # ---------------- metadata ----------------
    meta = {
        k: v
        for k, v in rec.items()
        if not isinstance(v, (dict, list))
    }
    if meta:
        sheet_key_value(wb, prefix + "metadata", meta)

    # ---------------- hyperparameters ----------------
    params = rec.get("hyperparameters")
    if isinstance(params, dict):
        rows = [[k, v] for k, v in params.items()]
        sheet_table(
            wb,
            prefix + "hyperparameters",
            ["parameter", "value"],
            rows,
        )

    # ---------------- training config ----------------
    train_cfg = rec.get("training_config")
    if isinstance(train_cfg, dict):
        sheet_key_value(wb, prefix + "training_config", train_cfg)

    # ---------------- signals ----------------
    signals = rec.get("signals")
    if isinstance(signals, dict):
        sheet_key_value(wb, prefix + "signals", signals)

    # ---------------- resource config ----------------
    resources = rec.get("resource_config")
    if isinstance(resources, dict):
        sheet_key_value(wb, prefix + "resources", resources)


# -----------------------------
# main
# -----------------------------
def model_initialization():

    records = load_jsonl(INPUT)

    wb = Workbook()
    wb.remove(wb.active)

    for i, rec in enumerate(records, 1):
        parse_record(wb, rec, i)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _auto_adjust_column_width_2(wb)
    wb.save(OUTPUT)

    print("Excel written →", OUTPUT)


if __name__ == "__main__":
    model_initialization()