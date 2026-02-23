import json
from pathlib import Path
from openpyxl import Workbook


# --------------------------------------------------
# SAFE VALUE FORMATTER (critical fix)
# --------------------------------------------------
def excel_safe(value):
    """
    Convert any Python object into Excel-safe representation.
    """

    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    # everything else → JSON string
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


# --------------------------------------------------
# TABLE WRITER
# --------------------------------------------------
def write_table(ws, rows):

    if not rows:
        return

    # collect all possible keys
    headers = sorted({k for r in rows for k in r})

    ws.append(headers)

    for r in rows:
        ws.append([excel_safe(r.get(h)) for h in headers])


# --------------------------------------------------
# KEY VALUE WRITER
# --------------------------------------------------
def write_kv(ws, d):

    ws.append(["key", "value"])

    for k, v in d.items():
        ws.append([k, excel_safe(v)])


# --------------------------------------------------
# MAIN EXPORTER
# --------------------------------------------------
def export_full_audit(data_dir, output_file):

    wb = Workbook()
    wb.remove(wb.active)

    dataset_meta = {}
    column_profiles = []
    preprocess_steps = []
    models = []
    metrics = []
    artifacts = []

    for jf in Path(data_dir).glob("*.jsonl"):

        with open(jf, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        if not lines:
            continue

        rec = lines[-1]
        name = jf.stem

        # ---------------- routing ----------------
        if "column_inspection" in name:
            dataset_meta.update({
                "dataset": rec.get("dataset_file_name"),
                "path": rec.get("dataset_file_path")
            })
            column_profiles.extend(rec.get("column_profiles", []))

        elif "preprocess" in name:
            preprocess_steps.append(rec)

        elif "model_selection" in name:
            models.append(rec)

        elif "evaluation" in name:
            metrics.append(rec)

        else:
            artifacts.append(rec)

    # ---------------- write sheets ----------------

    if dataset_meta:
        ws = wb.create_sheet("dataset_summary")
        write_kv(ws, dataset_meta)

    if column_profiles:
        ws = wb.create_sheet("column_profiles")
        write_table(ws, column_profiles)

    if preprocess_steps:
        ws = wb.create_sheet("preprocessing")
        write_table(ws, preprocess_steps)

    if models:
        ws = wb.create_sheet("model_selection")
        write_table(ws, models)

    if metrics:
        ws = wb.create_sheet("evaluation")
        write_table(ws, metrics)

    if artifacts:
        ws = wb.create_sheet("artifacts")
        write_table(ws, artifacts)

    wb.save(output_file)