import json
from pathlib import Path
from groq import Groq
import api_keys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data"
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
]
MODEL_NAME = "llama-3.3-70b-versatile"


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _get_client():
    return Groq()

def _format_nested(obj, indent=0):
    space = "  " * indent

    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{space}{k}:")
                lines.append(_format_nested(v, indent + 1))
            else:
                lines.append(f"{space}{k}: {v}")
        return "\n".join(lines)

    elif isinstance(obj, list):
        lines = []
        for i, item in enumerate(obj):
            lines.append(f"{space}- item {i}:")
            lines.append(_format_nested(item, indent + 1))
        return "\n".join(lines)

    else:
        return f"{space}{obj}"

def _read_jsonl_stream(path: Path):
    blocks = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                formatted = _format_nested(obj)
            except Exception:
                formatted = line

            blocks.append(f"RECORD {i}\n{formatted}")

    return "\n\n".join(blocks)

def _build_prompt(file_name: str, content: str):
    return f"""
You are a senior ML pipeline auditor.

Analyze this file deeply and technically.

FILE: {file_name}

Explain:
- role in ML pipeline
- structure and semantics
- important signals
- anomalies / risks
- downstream impact
- improvements

DATA:
{content}
"""


# ---------------------------------------------------------
# main function (THIS is what you use)
# ---------------------------------------------------------

def analyze_pipeline_directory() -> str:
    """
    Reads all pipeline JSONL files in order and returns one combined LLM analysis.
    """

    client = _get_client()
    base = Path(INPUT_DIR)

    if not base.exists():
        raise FileNotFoundError(base)

    all_results = []

    for file_name in FILES_IN_ORDER:
        path = base / file_name

        if not path.exists():
            all_results.append(f"### {file_name}\nFile not found.\n")
            continue

        content = _read_jsonl_stream(path)
        prompt = _build_prompt(file_name, content)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert ML system auditor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        response = completion.choices[0].message.content

        name_clean = Path(file_name).stem.upper()
        all_results.append(f"## {name_clean}\n{response}\n")

    return "\n".join(all_results)