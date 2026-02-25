from pathlib import Path
import shutil


def clear_project_data_dir():
    project_root = Path(__file__).resolve().parents[1]  # adjust if needed
    data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"Directory not found → {data_dir}")
        return

    for item in data_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    print(f"Cleared → {data_dir}")