from __future__ import annotations

import json
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from run_agent import build_graph

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_files" / "user_uploads"


def _clean_data_dir_once() -> None:
    """Delete all files/folders in data directory once per Streamlit session."""
    if st.session_state.get("data_dir_cleaned", False):
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)

    st.session_state["data_dir_cleaned"] = True


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_columns(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]


def _save_uploaded_file(uploaded_file) -> tuple[Path, pd.DataFrame]:
    """Save uploaded tabular file as timestamped CSV inside uploaded_files/user_uploads."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name)
    suffix = original_name.suffix.lower()
    time_suffix = _timestamp()
    base_name = f"{original_name.stem}_{time_suffix}"

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    output_csv = UPLOAD_DIR / f"{base_name}.csv"
    df.to_csv(output_csv, index=False)

    return output_csv, df


def _run_pipeline(data_path: Path, categorical_cols: list[str], target_col: str | None) -> dict[str, Any]:
    graph = build_graph()
    initial_state: dict[str, Any] = {
        "data_path": str(data_path),
        "categorical_columns": categorical_cols,
        "target_column": target_col or None,
        "preprocess_last_n": 1,
        "evaluation_last_n": 1,
    }
    return graph.invoke(initial_state)


def _collect_data_outputs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.rglob("*") if p.is_file()])


def _jsonl_to_xlsx_bytes(jsonl_path: Path) -> bytes:
    records: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    records.append({"value": obj})
            except json.JSONDecodeError:
                records.append({"raw_line": line, "parse_error": True})

    if not records:
        records = [{"info": "No valid records found."}]

    df = pd.json_normalize(records, sep=".")
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="records", index=False)
    buff.seek(0)
    return buff.read()


def _latest_joblib() -> Path | None:
    artifacts_dir = DATA_DIR / "artifacts"
    if not artifacts_dir.exists():
        return None

    models = sorted(artifacts_dir.rglob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def _usage_doc_for_model(model_path: Path, target_col: str | None) -> str:
    return (
        "# Model Usage Guide\n\n"
        f"**Model file:** `{model_path}`\n\n"
        "```python\n"
        "import joblib\n"
        "import pandas as pd\n\n"
        f"model = joblib.load(r\"{model_path}\")\n"
        "new_data = pd.read_csv('new_data.csv')\n"
        f"# If target column exists, drop it before prediction: {target_col!r}\n"
        "# new_data = new_data.drop(columns=['target_column_name'], errors='ignore')\n"
        "predictions = model.predict(new_data)\n"
        "print(predictions[:10])\n"
        "```\n"
    )


def _build_ai_summary(final_state: dict[str, Any]) -> str:
    """Simple LLM-like narrative summary from graph outputs."""
    schema = final_state.get("schema_result", {})
    eval_result = final_state.get("evaluation_result", [])
    model_init = final_state.get("model_initialization_result", [])

    n_rows = schema.get("n_rows", "unknown")
    n_cols = schema.get("n_columns", "unknown")
    target = schema.get("target") or "not specified"

    model_family = "unknown"
    if model_init and isinstance(model_init, list):
        last = model_init[-1]
        model_family = str(last.get("model_name") or last.get("estimator") or "unknown")

    metric_hint = "Evaluation records generated."
    if eval_result and isinstance(eval_result, list):
        metric_hint = f"Evaluation generated {len(eval_result)} record(s)."

    return (
        f"Dataset has **{n_rows} rows** and **{n_cols} columns** with target **{target}**. "
        f"Selected/initialized model appears to be **{model_family}**. {metric_hint} "
        "You can audit all pipeline stages via downloadable JSONL and XLSX exports below."
    )


def main() -> None:
    st.set_page_config(page_title="Agentic AutoML", layout="wide")
    _clean_data_dir_once()

    st.title("🤖 Agentic AutoML Studio")
    st.caption("Upload data, configure schema hints, run the full pipeline, and download all artifacts.")

    with st.sidebar:
        st.header("Run Configuration")
        uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
        target_col = st.text_input("Target column (optional)")
        categorical_raw = st.text_area("Categorical columns (comma-separated)")
        numerical_raw = st.text_area("Numerical columns (comma-separated)")
        run_btn = st.button("🚀 Run AutoML", type="primary")

    if run_btn and uploaded_file is None:
        st.error("Please upload a file first.")
        return

    if run_btn and uploaded_file is not None:
        saved_path, df = _save_uploaded_file(uploaded_file)
        categorical_cols = _parse_columns(categorical_raw)
        numerical_cols = _parse_columns(numerical_raw)

        missing = set(categorical_cols + numerical_cols + ([target_col] if target_col else [])) - set(df.columns)
        if missing:
            st.error(f"These columns were not found in dataset: {sorted(missing)}")
            return

        with st.spinner("Running end-to-end agent graph..."):
            final_state = _run_pipeline(saved_path, categorical_cols, target_col or None)

        st.success("Pipeline execution completed.")

        st.subheader("Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])

        st.subheader("AI Summary")
        st.markdown(_build_ai_summary(final_state))

        st.subheader("Pipeline State Output")
        st.json(final_state)

        st.subheader("Generated Files (data folder)")
        outputs = _collect_data_outputs()
        if not outputs:
            st.info("No output files found in data folder yet.")
        else:
            for p in outputs:
                relative = p.relative_to(PROJECT_ROOT)
                with p.open("rb") as f:
                    st.download_button(
                        label=f"Download {relative}",
                        data=f.read(),
                        file_name=p.name,
                        mime="application/octet-stream",
                        key=f"dl_{relative}",
                    )

        st.subheader("JSONL → XLSX Audit Exports")
        jsonl_files = [p for p in outputs if p.suffix.lower() == ".jsonl"]
        if not jsonl_files:
            st.info("No JSONL files found for conversion.")
        else:
            for jf in jsonl_files:
                xlsx_data = _jsonl_to_xlsx_bytes(jf)
                out_name = f"{jf.stem}_audit.xlsx"
                st.download_button(
                    label=f"Download {jf.name} as XLSX",
                    data=xlsx_data,
                    file_name=out_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xlsx_{jf}",
                )

        st.subheader("Latest Model Artifact")
        latest_model = _latest_joblib()
        if latest_model is None:
            st.warning("No .joblib model artifact found in data/artifacts.")
        else:
            with latest_model.open("rb") as f:
                st.download_button(
                    label=f"Download latest model: {latest_model.name}",
                    data=f.read(),
                    file_name=latest_model.name,
                    mime="application/octet-stream",
                    key="latest_joblib",
                )

            usage_doc = _usage_doc_for_model(latest_model, target_col or None)
            st.download_button(
                label="Download model usage documentation (.md)",
                data=usage_doc,
                file_name="model_usage_guide.md",
                mime="text/markdown",
                key="model_usage_doc",
            )
            st.markdown("### Quick Usage")
            st.markdown(usage_doc)


if __name__ == "__main__":
    main()
