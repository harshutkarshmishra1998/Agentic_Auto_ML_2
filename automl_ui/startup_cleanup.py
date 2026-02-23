import shutil
from pathlib import Path


def clean_directory(path: Path):
    if not path.exists():
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def run_startup_cleanup():

    data_dir = Path("data")
    uploads_dir = Path("uploaded_files/user_uploads")

    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    clean_directory(data_dir)
    clean_directory(uploads_dir)