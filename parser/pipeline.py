from pathlib import Path
import shutil
from parser.user_input import user_input
from parser.data_classification import data_classification
from parser.column_inspection import column_inspection
from parser.preprocess_1 import preprocess_1
from parser.model_selection import model_selection
from parser.preprocess_2 import preprocess_2
from parser.model_initialization import model_initialization
from parser.ml_experiment import ml_experiment
from parser.evaluation import evaluation
from parser.dataset_extractor import dataset_extractor


def clear_parser_data_dir():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "parser" / "data"

    if not data_dir.exists():
        print(f"Directory not found → {data_dir}")
        return

    for item in data_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def is_folder_exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    XLSX_PATH= PROJECT_ROOT / "parser" / "data" / "xlsx"
    CSV_PATH = PROJECT_ROOT / "parser" / "data" / "csv"
    JOBLIB_PATH= PROJECT_ROOT / "parser" / "data" / "joblib"

    XLSX_PATH.mkdir(parents=True, exist_ok=True)
    CSV_PATH.mkdir(parents=True, exist_ok=True)
    JOBLIB_PATH.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    clear_parser_data_dir()
    is_folder_exists()
    user_input()
    data_classification()
    column_inspection()
    preprocess_1()
    model_selection()
    preprocess_2()
    model_initialization()
    ml_experiment()
    evaluation()
    dataset_extractor()


if __name__ == "__main__":
    run_pipeline()