# OK
import json
from pathlib import Path
from openpyxl import Workbook
from parser.excel_writer import sheet_key_value, sheet_table


INPUT = Path("data/user_input.jsonl")
OUTPUT = Path("parser/data/user_input.xlsx")


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_record(wb, rec, idx):

    prefix = f"record_{idx}_"

    # ---------- metadata ----------
    meta = {k: v for k, v in rec.items() if not isinstance(v, (list, dict))}
    if meta:
        sheet_key_value(wb, prefix + "metadata", meta)

    # ---------- feature roles ----------
    roles = rec.get("feature_roles")
    if isinstance(roles, dict):
        rows = []
        for feature, info in roles.items():
            rows.append([
                feature,
                info.get("role"),
                info.get("confidence"),
            ])

        sheet_table(
            wb,
            prefix + "feature_roles",
            ["feature", "role", "confidence"],
            rows,
        )


def main():
    records = load_jsonl(INPUT)

    wb = Workbook()
    wb.remove(wb.active)

    for i, rec in enumerate(records, 1):
        parse_record(wb, rec, i)

    wb.save(OUTPUT)


if __name__ == "__main__":
    main()