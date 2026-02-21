from datetime import datetime
from pathlib import Path

def now_iso():
    return datetime.utcnow().isoformat()

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)