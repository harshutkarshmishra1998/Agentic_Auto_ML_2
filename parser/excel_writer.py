# OK
import json
from openpyxl import Workbook


# def fmt(v):
#     if v is None:
#         return ""
#     if isinstance(v, (dict, list)):
#         return json.dumps(v, indent=2)
#     return v

def fmt(v):
    if v is None:
        return ""

    # list → comma separated string
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)

    # dict → keep JSON (optional, can change later)
    if isinstance(v, dict):
        return json.dumps(v)

    return v


def sheet_table(wb, name, headers, rows):
    ws = wb.create_sheet(name[:31])
    ws.append(headers)
    for r in rows:
        ws.append([fmt(x) for x in r])


def sheet_key_value(wb, name, data: dict):
    ws = wb.create_sheet(name[:31])
    ws.append(["key", "value"])
    for k, v in data.items():
        ws.append([k, fmt(v)])


def sheet_list(wb, name, values, header="value"):
    ws = wb.create_sheet(name[:31])
    ws.append([header])
    for v in values:
        ws.append([fmt(v)])