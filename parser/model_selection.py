# OK
import json
from pathlib import Path
from openpyxl import Workbook
from parser.excel_writer import sheet_key_value, sheet_table


INPUT = Path("data/model_selection.jsonl")
OUTPUT = Path("parser/data/model_selection.xlsx")


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_record(wb, rec, i):

    prefix = f"record_{i}_"

    meta = rec.get("problem_definition", {})
    if meta:
        sheet_key_value(wb, prefix + "problem_definition", meta)

    ranking = rec.get("model_ranking")
    if ranking:
        rows = [[m, s] for m, s in ranking]
        sheet_table(
            wb,
            prefix + "model_ranking",
            ["model", "score"],
            rows,
        )

    selected = rec.get("primary_model")
    if selected:
        sheet_key_value(wb, prefix + "selected_model", {"model": selected})


def main():
    records = load(INPUT)

    wb = Workbook()
    wb.remove(wb.active)

    for i, rec in enumerate(records, 1):
        parse_record(wb, rec, i)

    wb.save(OUTPUT)


if __name__ == "__main__":
    main()