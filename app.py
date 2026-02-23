import streamlit as st
import pandas as pd
import pickle
import joblib
from pathlib import Path

from run_agent import build_graph

from automl_ui.startup_cleanup import run_startup_cleanup
from automl_ui.schema_resolver import resolve_schema
from automl_ui.audit_excel import export_full_audit
from automl_ui.llm_summary import generate_summary
from automl_ui.artifact_finder import latest_joblib, latest_preprocessed_final


def safe_download_button(label, data, filename, mime=None):
    """
    Render download button only if data is valid binary.
    Prevents Streamlit NoneType errors.
    """

    if data is None:
        return

    if isinstance(data, (bytes, bytearray)):
        st.download_button(label, data, filename, mime=mime)
        return

    # fallback — try convert to bytes
    try:
        if isinstance(data, str):
            st.download_button(label, data.encode(), filename, mime=mime)
            return
    except Exception:
        pass

    # invalid type — do not render
    st.warning(f"Download '{filename}' not available")

# --------------------------------------------------
# SESSION STATE DEFAULTS (CRITICAL)
# --------------------------------------------------
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

# --------------------------------------------------
# STARTUP CLEAN (runs once per server start)
# --------------------------------------------------
if "startup_done" not in st.session_state:
    run_startup_cleanup()
    st.session_state.startup_done = True


# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_DIR = Path("data")
UPLOAD_DIR = Path("uploaded_files/user_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False


# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Agentic AutoML Studio")

uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

cat_text = st.text_input("Categorical columns (comma separated)")
num_text = st.text_input("Numerical columns (comma separated)")
tgt_text = st.text_input("Target column (optional)")


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------
if st.button("Run AutoML") and uploaded_file:

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    cat = [c.strip() for c in cat_text.split(",") if c.strip()]
    num = [c.strip() for c in num_text.split(",") if c.strip()]
    tgt = tgt_text.strip() if tgt_text else None

    # resolve schema (safe inference)
    cat, num, tgt = resolve_schema(df.columns, cat, num, tgt)

    st.subheader("Resolved Schema")
    st.write("Categorical:", cat)
    st.write("Numerical:", num)
    st.write("Target:", tgt if tgt else "None (unsupervised mode)")

    # save dataset
    data_path = UPLOAD_DIR / "data.csv"
    df.to_csv(data_path, index=False)

    # run pipeline
    with st.spinner("Running pipeline..."):
        graph = build_graph()
        result = graph.invoke({
            "data_path": str(data_path),
            "categorical_columns": cat,
            "target_column": tgt,
            "preprocess_last_n": 1,
            "evaluation_last_n": 1,
        })

    # ---------------------------------------------
    # STORE EVERYTHING IN SESSION
    # ---------------------------------------------
    st.session_state.pipeline_done = True
    st.session_state.df = df
    st.session_state.result = result

    # LLM summary
    st.session_state.summary = generate_summary(DATA_DIR)

    # audit excel
    audit_file = DATA_DIR / "pipeline_audit.xlsx"
    export_full_audit(DATA_DIR, audit_file)
    st.session_state.audit_excel = audit_file.read_bytes()

    # model
    model_file = latest_joblib(DATA_DIR)
    if model_file:
        model = joblib.load(model_file)
        pkl_path = DATA_DIR / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        st.session_state.model_pickle = pkl_path.read_bytes()

    # preprocessed data
    preprocessed = latest_preprocessed_final(DATA_DIR)
    if preprocessed:
        st.session_state.preprocessed = preprocessed.read_bytes()
        st.session_state.preprocessed_name = preprocessed.name


# # --------------------------------------------------
# # DISPLAY RESULTS (PERSISTENT)
# # --------------------------------------------------
# if st.session_state.pipeline_done:

#     st.success("Pipeline complete")

#     st.subheader("Data Preview")
#     st.dataframe(st.session_state.df.head())

#     st.subheader("LLM Audit Summary")
#     st.write(st.session_state.summary)

#     st.download_button(
#         "Download Pipeline Audit Excel",
#         st.session_state.audit_excel,
#         "pipeline_audit.xlsx"
#     )

#     if "model_pickle" in st.session_state:
#         st.download_button(
#             "Download Model (pickle)",
#             st.session_state.model_pickle,
#             "model.pkl"
#         )

#     if "preprocessed" in st.session_state:
#         st.download_button(
#             "Download Preprocessed Final",
#             st.session_state.preprocessed,
#             st.session_state.preprocessed_name
#         )

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
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
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    safe_download_button(
        "Download Model (pickle)",
        st.session_state.model_pickle,
        "model.pkl",
        "application/octet-stream"
    )

    if st.session_state.preprocessed_name:
        safe_download_button(
            "Download Preprocessed Final",
            st.session_state.preprocessed,
            st.session_state.preprocessed_name,
            "application/octet-stream"
        )