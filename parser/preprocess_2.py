# OK
import json
from pathlib import Path
from openpyxl import Workbook

from parser.excel_writer import sheet_key_value, sheet_table


INPUT = Path("data/preprocess_2.jsonl")
OUTPUT = Path("parser/data/preprocess_2.xlsx")


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

    # ---------- metadata ----------
    meta = {
        k: v
        for k, v in rec.items()
        if not isinstance(v, (list, dict))
    }
    if meta:
        sheet_key_value(wb, prefix + "metadata", meta)

    # ---------- strategy logs ----------
    # logs = rec.get("strategy_logs")
    # if isinstance(logs, list) and logs:

    #     # collect all possible keys across strategies
    #     headers = sorted({
    #         k for log in logs for k in log.keys()
    #     })

    #     rows = []
    #     for log in logs:
    #         rows.append([log.get(h) for h in headers])

    #     sheet_table(
    #         wb,
    #         prefix + "strategy_logs",
    #         headers,
    #         rows,
    #     )

    logs = rec.get("strategy_logs")
    if isinstance(logs, list) and logs:

        headers = ["strategy", "details"]

        def fmt(v):
            if v is None or v == []:
                return ""
            if isinstance(v, list):
                return ", ".join(str(x) for x in v)
            return str(v)

        rows = []

        for log in logs:
            strategy = log.get("strategy", "")

            details_parts = []
            for k, v in log.items():
                if k == "strategy":
                    continue
                val = fmt(v)
                if val != "":
                    details_parts.append(f"{k}={val}")

            details = ", ".join(details_parts)

            rows.append([strategy, details])

        sheet_table(
            wb,
            prefix + "strategy_logs",
            headers,
            rows,
        )

# -----------------------------
# main
# -----------------------------
def main():

    records = load_jsonl(INPUT)

    wb = Workbook()
    wb.remove(wb.active)

    for i, rec in enumerate(records, 1):
        parse_record(wb, rec, i)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)

    print("Excel written →", OUTPUT)


if __name__ == "__main__":
    main()