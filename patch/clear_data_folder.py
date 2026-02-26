# from pathlib import Path
# import shutil


# def clear_project_data_dir():
#     project_root = Path(__file__).resolve().parents[1]  # adjust if needed
#     data_dir = project_root / "data"

#     if not data_dir.exists():
#         print(f"Directory not found → {data_dir}")
#         return

#     for item in data_dir.iterdir():
#         if item.is_dir():
#             shutil.rmtree(item)
#         else:
#             item.unlink()

#     print(f"Cleared → {data_dir}")

from pathlib import Path
import shutil


def clear_project_data_dir():
    project_root = Path(__file__).resolve().parents[1]  # adjust if needed

    paths_to_clear = [
        project_root / "data",
        project_root / "uploaded_files" / "user_uploads",
        project_root / "parser" / "data",
    ]

    for target_dir in paths_to_clear:

        if not target_dir.exists():
            print(f"Directory not found → {target_dir}")
            continue

        for item in target_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        print(f"Cleared → {target_dir}")