from pathlib import Path
import shutil


def copy_dataset_to_user_uploads(data_file: str):
    project_root = Path(__file__).resolve().parents[1]

    src = project_root / data_file

    if not src.exists():
        raise FileNotFoundError(f"Source file not found → {src}")

    dest_dir = project_root / "uploaded_files" / "user_uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    shutil.copy2(src, dest)

    print(f"Copied → {src} → {dest}")

    return dest  # optional but useful