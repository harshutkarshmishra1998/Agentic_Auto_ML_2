import pickle
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from run_agent import build_graph

from automl_ui.artifact_finder import latest_joblib, latest_preprocessed_final
from automl_ui.audit_excel import export_full_audit
from automl_ui.llm_summary import generate_summary
from automl_ui.schema_resolver import resolve_schema
from automl_ui.startup_cleanup import run_startup_cleanup


def safe_download_button(label, data, filename, mime=None):
    if data is None:
        return

    if isinstance(data, (bytes, bytearray)):
        st.download_button(label, data, filename, mime=mime)
        return

    try:
        if isinstance(data, str):
            st.download_button(label, data.encode(), filename, mime=mime)
            return
    except Exception:
        pass

    st.warning(f"Download '{filename}' not available")


def reset_pipeline_state():
    st.session_state.pipeline_done = False
    st.session_state.df = None
    st.session_state.result = None
    st.session_state.summary = None
    st.session_state.audit_excel = None
    st.session_state.model_pickle = None
    st.session_state.preprocessed = None
    st.session_state.preprocessed_name = None


SESSION_DEFAULTS = {
    "pipeline_done": False,
    "df": None,
    "result": None,
    "summary": None,
    "audit_excel": None,
    "model_pickle": None,
    "preprocessed": None,
    "preprocessed_name": None,
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "startup_done" not in st.session_state:
    run_startup_cleanup()
    st.session_state.startup_done = True

DATA_DIR = Path("data")
UPLOAD_DIR = Path("uploaded_files/user_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Agentic AutoML Studio")

if st.session_state.pipeline_done:
    if st.button("Run New AutoML Pipeline"):
        reset_pipeline_state()
        st.rerun()
else:
    st.caption("Upload your dataset and optional schema hints. This panel hides automatically once execution starts.")

run_clicked = False
uploaded_file = None
cat_text = ""
num_text = ""
tgt_text = ""
input_container = st.container()

if not st.session_state.pipeline_done:
    with input_container:
        with st.form("pipeline_form"):
            uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
            col1, col2, col3 = st.columns(3)
            with col1:
                cat_text = st.text_area("Categorical columns", placeholder="e.g. gender, city")
            with col2:
                num_text = st.text_area("Numerical columns", placeholder="e.g. age, income")
            with col3:
                tgt_text = st.text_input("Target column (optional)")

            run_clicked = st.form_submit_button("Run AutoML", use_container_width=True)

if run_clicked and uploaded_file is not None:
    input_container.empty()

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    cat = [c.strip() for c in cat_text.split(",") if c.strip()]
    num = [c.strip() for c in num_text.split(",") if c.strip()]
    tgt = tgt_text.strip() if tgt_text else None

    cat, num, tgt = resolve_schema(df.columns, cat, num, tgt)

    st.subheader("Resolved Schema")
    schema_col1, schema_col2, schema_col3 = st.columns(3)
    schema_col1.write("**Categorical**")
    schema_col1.write(cat)
    schema_col2.write("**Numerical**")
    schema_col2.write(num)
    schema_col3.write("**Target**")
    schema_col3.write(tgt if tgt else "None (unsupervised mode)")

    data_path = UPLOAD_DIR / "data.csv"
    df.to_csv(data_path, index=False)

    with st.spinner("Running pipeline..."):
        graph = build_graph()
        result = graph.invoke(
            {
                "data_path": str(data_path),
                "categorical_columns": cat,
                "target_column": tgt,
                "preprocess_last_n": 1,
                "evaluation_last_n": 1,
            }
        )

    st.session_state.pipeline_done = True
    st.session_state.df = df
    st.session_state.result = result
    st.session_state.summary = generate_summary(DATA_DIR)

    audit_file = DATA_DIR / "pipeline_audit.xlsx"
    export_full_audit(DATA_DIR, audit_file)
    st.session_state.audit_excel = audit_file.read_bytes()

    model_file = latest_joblib(DATA_DIR)
    if model_file:
        model = joblib.load(model_file)
        pkl_path = DATA_DIR / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        st.session_state.model_pickle = pkl_path.read_bytes()

    preprocessed = latest_preprocessed_final(DATA_DIR)
    if preprocessed:
        st.session_state.preprocessed = preprocessed.read_bytes()
        st.session_state.preprocessed_name = preprocessed.name

if run_clicked and uploaded_file is None:
    st.warning("Please upload a file before running the pipeline.")

if st.session_state.pipeline_done:
    st.success("Pipeline complete")

    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())

    if st.session_state.summary:
        st.subheader("LLM Audit Summary")
        st.write(st.session_state.summary)

    safe_download_button(
        "Download Pipeline Audit Excel",
        st.session_state.audit_excel,
        "pipeline_audit.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    safe_download_button(
        "Download Model (pickle)",
        st.session_state.model_pickle,
        "model.pkl",
        "application/octet-stream",
    )

    if st.session_state.preprocessed_name:
        safe_download_button(
            "Download Preprocessed Final",
            st.session_state.preprocessed,
            st.session_state.preprocessed_name,
            "application/octet-stream",
        )
