# OK
import json
from pathlib import Path
from openpyxl import Workbook
from parser.excel_writer import sheet_key_value


INPUT = Path("data/data_classification.jsonl")
OUTPUT = Path("parser/data/data_classification.xlsx")


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# def parse_record(wb, rec, i):

#     prefix = f"record_{i}_"

#     flat = {}
#     clustering = {}

#     for k, v in rec.items():
#         if k == "clustering":
#             clustering = v
#         else:
#             flat[k] = v

#     if flat:
#         sheet_key_value(wb, prefix + "metadata", flat)

#     if clustering:
#         sheet_key_value(wb, prefix + "clustering", clustering)

def parse_record(wb, rec, i):

    prefix = f"record_{i}_"

    metadata = {}
    feature_mapping = {}
    clustering = {}

    # ---------- separate structures ----------
    for k, v in rec.items():
        if k == "feature_mapping":
            feature_mapping = v
        elif k == "clustering":
            clustering = v
        else:
            metadata[k] = v

    # ---------- metadata sheet ----------
    if metadata:
        sheet_key_value(wb, prefix + "metadata", metadata)

    # ---------- feature mapping table ----------
    if feature_mapping:

        from parser.excel_writer import sheet_table

        headers = ["feature", "role", "confidence"]
        rows = []

        for feature, info in feature_mapping.items():
            rows.append([
                feature,
                info.get("role"),
                info.get("confidence"),
            ])

        sheet_table(
            wb,
            prefix + "feature_mapping",
            headers,
            rows,
        )

    # ---------- clustering ----------
    if clustering:
        sheet_key_value(wb, prefix + "clustering", clustering)

def main():
    records = load(INPUT)

    wb = Workbook()
    wb.remove(wb.active)

    for i, rec in enumerate(records, 1):
        parse_record(wb, rec, i)

    wb.save(OUTPUT)


if __name__ == "__main__":
    main()