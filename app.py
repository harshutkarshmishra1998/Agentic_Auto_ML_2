import streamlit as st
import os
import uuid
import zipfile
import time
from pathlib import Path

from run_agent import run_process
from patch.clear_data_folder import clear_project_data_dir


# CONFIG
UPLOAD_DIR = Path("uploaded_files/user_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Agentic Auto ML",
    page_icon="🤖",
    layout="wide"
)


# SESSION INITIALIZATION (RUN ONLY ONCE PER SESSION)
if "initialized" not in st.session_state:
    clear_project_data_dir()
    st.session_state.initialized = True

if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

if "llm_response" not in st.session_state:
    st.session_state.llm_response = None

if "files_registry" not in st.session_state:
    st.session_state.files_registry = None

if "saved_path" not in st.session_state:
    st.session_state.saved_path = None


# UI STYLING
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    font-size: 18px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)


# HELPERS
def save_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1]
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_DIR / unique_name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(save_path)


def collect_files_by_extension(files_registry, ext):
    return [
        f["file_path"]
        for f in files_registry
        if f["file_name"].endswith(ext)
    ]


def create_zip(file_list, zip_name="jsonl_logs.zip"):
    zip_path = UPLOAD_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w") as z:
        for file in file_list:
            if os.path.exists(file):
                z.write(file, arcname=os.path.basename(file))
    return zip_path


# def collect_parser_csv(files_registry):
#     return [
#         f for f in files_registry
#         if f["file_name"].lower().endswith(".csv")
#         and "parser\\data\\csv" in f["file_path"].lower().replace("/", "\\")
#     ]

def collect_parser_csv(files_registry):
    result = []

    for f in files_registry:
        if not f["file_name"].lower().endswith(".csv"):
            continue

        parts = Path(f["file_path"]).parts

        # check directory structure safely
        if ("parser" in parts) and ("data" in parts) and ("csv" in parts):
            result.append(f)

    return result


# HEADER
st.markdown('<div class="main-title">🚀 Agentic Auto ML</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload → Analyze → Train → Evaluate → Download</div>', unsafe_allow_html=True)
st.divider()


# INPUT SECTION
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

with col2:
    st.markdown("### Optional Inputs")

    categorical_input = st.text_input(
        "Categorical Columns",
        placeholder="state, area_code"
    )

    target_input = st.text_input(
        "Target Column",
        placeholder=""
    )

st.divider()


# RUN PIPELINE BUTTON
if st.button("🚀 Run Auto ML", use_container_width=True):

    if uploaded_file is None:
        st.error("Please upload a dataset.")
        st.stop()

    saved_path = save_uploaded_file(uploaded_file)
    st.session_state.saved_path = saved_path

    cats = [c.strip() for c in categorical_input.split(",")] if categorical_input.strip() else []
    target = target_input.strip() if target_input.strip() else None

    progress = st.progress(0)
    status = st.empty()

    for i in range(40):
        progress.progress(i)
        status.info("Initializing intelligent pipeline...")
        time.sleep(0.02)

    with st.spinner("Running ML pipeline..."):
        result, llm_response, files_registry = run_process(
            DATA_FILE=saved_path,
            cats=cats,
            target=target
        )

    for i in range(40, 100):
        progress.progress(i)
        status.info("Finalizing reports...")
        time.sleep(0.01)

    progress.progress(100)
    status.success("Pipeline completed successfully!")

    st.session_state.llm_response = llm_response
    st.session_state.files_registry = files_registry
    st.session_state.pipeline_ran = True


# SHOW RESULTS (PERSISTENT)
if st.session_state.pipeline_ran:

    st.divider()
    st.subheader("🧠 LLM Analysis")
    st.code(st.session_state.llm_response, language="markdown")

    st.divider()
    # st.subheader("📦 Download Outputs")

    # files_registry = st.session_state.files_registry

    # # FILTER FILES
    # csv_files = collect_parser_csv(files_registry) if files_registry else []

    # xlsx_files = [
    #     f for f in files_registry
    #     if f["file_name"].lower().endswith(".xlsx")
    # ] if files_registry else []

    # jsonl_files = [
    #     f["file_path"] for f in files_registry
    #     if f["file_name"].lower().endswith(".jsonl")
    # ] if files_registry else []

    # colA, colB, colC = st.columns(3)


    # # CSV
    # with colA:
    #     st.markdown("### CSV Files")

    #     # fresh compute every rerun
    #     csv_files = [
    #         f for f in files_registry #type: ignore
    #         if f["file_name"].lower().endswith(".csv")
    #         and "parser\\data\\csv" in f["file_path"].lower()
    #     ]

    #     if not csv_files:
    #         st.warning("No parser CSV found.")
    #     else:
    #         # enforce exactly ONE button
    #         fobj = csv_files[0]

    #         path = fobj["file_path"]

    #         if os.path.exists(path):
    #             with open(path, "rb") as file_data:
    #                 st.download_button(
    #                     label=fobj["display_name"],
    #                     data=file_data,
    #                     file_name=fobj["file_name"],
    #                     key=f"csv_{hash(path)}",
    #                     use_container_width=True
    #                 )


    # # XLSX
    # with colB:
    #     st.markdown("### XLSX Reports")

    #     for fobj in xlsx_files:
    #         path = fobj["file_path"]
    #         label = fobj["display_name"]

    #         if os.path.exists(path):
    #             with open(path, "rb") as file_data:
    #                 st.download_button(
    #                     label=label,  # ✅ display name
    #                     data=file_data,
    #                     file_name=fobj["file_name"],
    #                     key=f"xlsx_{hash(path)}",
    #                     use_container_width=True
    #                 )


    # # JSONL
    # with colC:
    #     st.markdown("### JSONL Logs")

    #     if jsonl_files:
    #         zip_path = create_zip(jsonl_files)

    #         with open(zip_path, "rb") as f:
    #             st.download_button(
    #                 label="Download All JSONL Logs",
    #                 data=f,
    #                 file_name="pipeline_logs.zip",
    #                 key="jsonl_zip_download",
    #                 use_container_width=True
    #             )
    st.subheader("📦 Download Outputs")

    files_registry = st.session_state.files_registry

    # Categorize Files
    parser_csv = None
    joblib_file = None
    jsonl_files = []
    xlsx_files = []

    for f in files_registry: #type: ignore

        path = f["file_path"]
        name = f["file_name"].lower()

        # Final Preprocessed CSV (parser only)
        if name.endswith(".csv") and "\\parser\\data\\csv\\" in path.lower():
            parser_csv = f

        # Trained Model
        elif name.endswith(".joblib"):
            joblib_file = f

        # JSON Logs
        elif name.endswith(".jsonl"):
            jsonl_files.append(path)

        # Excel Reports
        elif name.endswith(".xlsx"):
            xlsx_files.append(f)


    col1, col2 = st.columns(2)


    # COLUMN 1
    with col1:
        st.markdown("### Core Artifacts")

        # Preprocessed CSV
        if parser_csv:
            path = parser_csv["file_path"]
            if os.path.exists(path):
                with open(path, "rb") as file_data:
                    st.download_button(
                        label=parser_csv["display_name"],
                        data=file_data,
                        file_name=parser_csv["file_name"],
                        key="download_final_csv",
                        use_container_width=True
                    )

        # Trained Model
        if joblib_file:
            path = joblib_file["file_path"]
            if os.path.exists(path):
                with open(path, "rb") as file_data:
                    st.download_button(
                        label=joblib_file["display_name"],
                        data=file_data,
                        file_name=joblib_file["file_name"],
                        key="download_model_joblib",
                        use_container_width=True
                    )

        # JSONL Logs ZIP
        if jsonl_files:
            zip_path = create_zip(jsonl_files)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="PIPELINE JSON LOGS (ZIP)",
                    data=f,
                    file_name="pipeline_logs.zip",
                    key="download_json_zip",
                    use_container_width=True
                )


    # COLUMN 2
    with col2:
        st.markdown("### Excel Reports")

        for fobj in xlsx_files:
            path = fobj["file_path"]

            if os.path.exists(path):
                with open(path, "rb") as file_data:
                    st.download_button(
                        label=fobj["display_name"].replace("_", " "),
                        data=file_data,
                        file_name=fobj["file_name"],
                        key=f"xlsx_{hash(path)}",
                        use_container_width=True
                    )
