from pathlib import Path


def latest_joblib(data_dir):
    files = list(Path(data_dir).rglob("*.joblib"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def latest_preprocessed_final(data_dir):
    matches = list(Path(data_dir).rglob("*preprocessed*final*"))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None