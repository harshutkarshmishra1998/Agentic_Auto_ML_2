# OK
import json
from pathlib import Path
from openpyxl import Workbook
from parser.excel_writer import _auto_adjust_column_width_2

INPUT = Path("data/evaluation.jsonl")
OUTPUT = Path("parser/data/xlsx/evaluation.xlsx")


# universal flatten (same as ml_experiments)
def flatten_dict(obj, parent_key="", sep="_"):
    items = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_dict(v, new_key, sep))

    elif isinstance(obj, list):
        items[parent_key] = ", ".join(map(str, obj))

    else:
        items[parent_key] = obj

    return items

def remove_empty_columns(ws):
    """
    Delete columns where:
    - header is empty OR
    - entire column has no data
    """

    cols_to_delete = []

    for col in range(1, ws.max_column + 1):

        header = ws.cell(row=1, column=col).value

        # check if column contains any data below header
        has_data = False
        for row in range(2, ws.max_row + 1):
            if ws.cell(row=row, column=col).value not in (None, ""):
                has_data = True
                break

        if (header is None or str(header).strip() == "") or not has_data:
            cols_to_delete.append(col)

    # delete from right to left (important!)
    for col in reversed(cols_to_delete):
        ws.delete_cols(col)


# loader
def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# schema adaptive writer (same as ml_experiments)
def write_record(ws, flat):

    # ---------- first record ----------
    if ws.max_row == 0:
        ws.append(list(flat.keys()))
        ws.append(list(flat.values()))
        return

    headers = [c.value for c in ws[1]]

    # ---------- add new columns (append only) ----------
    new_cols = [k for k in flat.keys() if k not in headers]

    if new_cols:
        start_col = len(headers) + 1

        # add header cells
        for idx, col_name in enumerate(new_cols, start=start_col):
            ws.cell(row=1, column=idx, value=col_name)

        headers.extend(new_cols)

    # ---------- build aligned row ----------
    row = [flat.get(h) for h in headers]

    ws.append(row)


# main
def evaluation():

    records = load(INPUT)

    if OUTPUT.exists():
        OUTPUT.unlink()

    wb = Workbook()
    wb.remove(wb.active)      # CRITICAL
    ws = wb.create_sheet("evaluation")

    for rec in records:
        flat = flatten_dict(rec)
        write_record(ws, flat)
    
    remove_empty_columns(ws)

    if ws.max_row == 0: #type: ignore
        ws.append(["message"])
        ws.append(["no evaluation records"])

    _auto_adjust_column_width_2(wb)
    wb.save(OUTPUT)


if __name__ == "__main__":
    evaluation()