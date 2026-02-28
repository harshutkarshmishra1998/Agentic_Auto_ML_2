from pathlib import Path
from typing import List, Dict
import fnmatch


# PIPELINE ORDER

FILES_IN_ORDER = [
    "user_input.jsonl",
    "data_classification.jsonl",
    "column_inspection.jsonl",
    "preprocesses_1.jsonl",
    "model_selection.jsonl",
    "preprocess_2.jsonl",
    "model_initialization.jsonl",
    "ml_experiments.jsonl",
    "evaluation.jsonl",

    "user_input.xlsx",
    "data_classification.xlsx",
    "column_inspection.xlsx",
    "preprocesses_1.xlsx",
    "model_selection.xlsx",
    "preprocess_2.xlsx",
    "model_initialization.xlsx",
    "ml_experiments.xlsx",
    "evaluation.xlsx",

    "*preprocessed*final.csv",
    "*.joblib",
]


# SEARCH DIRECTORIES

SEARCH_DIRS = [
    "data",
    "parser/data/xlsx",
    "parser/data/csv",
    "parser/data/joblib",
]


# ⭐ EXPLICIT DISPLAY NAME MAPPING (EDIT THIS)
# pattern → display label

DISPLAY_NAME_MAPPING = {

    # LOG FILES
    "user_input.jsonl": "USER_INPUT_LOGS",
    "data_classification.jsonl": "DATA_CLASSIFICATION_LOGS",
    "column_inspection.jsonl": "COLUMN_INSPECTION_LOGS",
    "preprocesses_1.jsonl": "PREPROCESS_1_LOGS",
    "model_selection.jsonl": "MODEL_SELECTION_LOGS",
    "preprocess_2.jsonl": "PREPROCESS_2_LOGS",
    "model_initialization.jsonl": "MODEL_INITIALIZATION_LOGS",
    "ml_experiments.jsonl": "ML_EXPERIMENTS_LOGS",
    "evaluation.jsonl": "EVALUATION_LOGS",

    # XLSX REPORT FILES
    "user_input.xlsx": "USER_INPUT_REPORT",
    "data_classification.xlsx": "DATA_CLASSIFICATION_REPORT",
    "column_inspection.xlsx": "COLUMN_INSPECTION_REPORT",
    "preprocesses_1.xlsx": "PREPROCESS_1_REPORT",
    "model_selection.xlsx": "MODEL_SELECTION_REPORT",
    "preprocess_2.xlsx": "PREPROCESS_2_REPORT",
    "model_initialization.xlsx": "MODEL_INITIALIZATION_REPORT",
    "ml_experiments.xlsx": "ML_EXPERIMENTS_REPORT",
    "evaluation.xlsx": "EVALUATION_REPORT",

    # FINAL OUTPUTS
    "*preprocessed*final.csv": "FINAL PREPROCESSED DATASET",
    "*.joblib": "TRAINED MODEL",
}


# DISPLAY NAME RESOLUTION

def _resolve_display_name(file_name: str) -> str:
    for pattern, label in DISPLAY_NAME_MAPPING.items():
        if fnmatch.fnmatch(file_name, pattern):
            return label

    # fallback if not mapped
    return Path(file_name).stem.upper()


# FILE SEARCH

def _find_matching_files(root: Path, pattern: str) -> List[Path]:
    matches = []

    for rel in SEARCH_DIRS:
        directory = root / rel
        if not directory.exists():
            continue

        for file in directory.iterdir():
            if file.is_file() and fnmatch.fnmatch(file.name, pattern):
                matches.append(file)

    return matches


# MAIN REGISTRY

def list_pipeline_outputs() -> List[Dict]:
    root = Path(__file__).resolve().parents[1]
    results = []

    for pattern in FILES_IN_ORDER:
        matched = _find_matching_files(root, pattern)

        if not matched:
            continue

        matched.sort(key=lambda p: p.name)

        for file in matched:
            results.append({
                "display_name": _resolve_display_name(file.name),
                "file_name": file.name,
                "file_path": str(file.resolve()),
            })

    return results