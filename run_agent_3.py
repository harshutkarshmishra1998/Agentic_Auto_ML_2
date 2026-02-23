from langgraph.graph import StateGraph, END

from agent_state import AgentState

from schema_engine.langgraph_node import schema_inference_node
from data_understanding.langgraph_node import data_understanding_node
from preprocess_1.langgraph_node import preprocess_1_node
from model_selector.langgraph_node import model_selector_node

from preprocess_2.langgraph_node import preprocess_2_node
from model_intializer.langgraph_node import model_initializer_node
from ml_engine.langgraph_node import ml_training_node
from evaluation_engine.langgraph_node import evaluation_node
from retrain.langgraph_node import retrain_node

from tests.schema_mapping import extract_schema

from pathlib import Path
import traceback


# --------------------------------------------------
# DATASET SELECTION (single or range)
# --------------------------------------------------
def select_dataset_folders(base_dir="uploaded_files"):
    base = Path(base_dir)

    if not base.exists():
        raise FileNotFoundError(f"{base_dir} not found")

    # deterministic ordering
    folders = sorted(
        [f for f in base.iterdir() if f.is_dir()],
        key=lambda x: x.name.lower()
    )

    if not folders:
        raise ValueError("No dataset folders found")

    print("\nAvailable datasets:\n")
    for i, folder in enumerate(folders, start=1):
        print(f"{i}. {folder.name}")

    while True:
        raw = input("\nSelect dataset number OR range (e.g. 3 or 1-5): ").strip()

        try:
            # ---------- single ----------
            if "-" not in raw:
                idx = int(raw)
                if 1 <= idx <= len(folders):
                    return [folders[idx - 1]]
                else:
                    print("Invalid number")
                    continue

            # ---------- range ----------
            start, end = map(int, raw.split("-"))

            if start > end:
                print("Invalid range")
                continue

            if start < 1 or end > len(folders):
                print("Range out of bounds")
                continue

            return folders[start - 1 : end]

        except Exception:
            print("Invalid input. Try again.")


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------
def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node("schema_inference", schema_inference_node)
    builder.add_node("data_understanding", data_understanding_node)
    builder.add_node("preprocess_1", preprocess_1_node)
    builder.add_node("model_selector", model_selector_node)
    builder.add_node("preprocess_2", preprocess_2_node)
    builder.add_node("model_initializer", model_initializer_node)
    builder.add_node("ml_training", ml_training_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("retrain", retrain_node)

    builder.set_entry_point("schema_inference")
    builder.add_edge("schema_inference", "data_understanding")
    builder.add_edge("data_understanding", "preprocess_1")
    builder.add_edge("preprocess_1", "model_selector")
    builder.add_edge("model_selector", "preprocess_2")
    builder.add_edge("preprocess_2", "model_initializer")
    builder.add_edge("model_initializer", "ml_training")
    builder.add_edge("ml_training", "evaluation")
    builder.add_edge("evaluation", "retrain")
    builder.add_edge("retrain", END)

    return builder.compile()


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------
if __name__ == "__main__":

    selected_folders = select_dataset_folders()
    graph = build_graph()

    total = len(selected_folders)

    for idx, folder in enumerate(selected_folders, start=1):

        print("\n" + "=" * 70)
        print(f"RUNNING DATASET {idx}/{total}")
        print(f"DATASET NAME: {folder.name}")
        print("=" * 70 + "\n")

        data_file = folder / "data.csv"
        metadata_file = folder / "metadata.jsonl"

        if not data_file.exists():
            raise FileNotFoundError(f"{data_file} not found")

        if not metadata_file.exists():
            raise FileNotFoundError(f"{metadata_file} not found")

        try:
            # ---------------- schema extraction ----------------
            cats, target = extract_schema(str(metadata_file), str(data_file))

            # ---------------- initial state ----------------
            initial_state: AgentState = {
                "data_path": str(data_file),
                "categorical_columns": cats,
                "target_column": target,
                "preprocess_last_n": 1,
                "evaluation_last_n": 1,
            }

            # ---------------- run graph ----------------
            result = graph.invoke(initial_state)

            # print("\n=== FINAL STATE ===\n")
            # print(result)

            print("\n✅ DATASET COMPLETED SUCCESSFULLY\n")

        except Exception:
            print("\n❌ PIPELINE FAILED")
            print(f"DATASET: {folder.name}")
            print("\nFULL ERROR TRACEBACK:\n")
            traceback.print_exc()

            print("\nStopping execution.")
            break

    else:
        print("\n🎉 ALL SELECTED DATASETS COMPLETED SUCCESSFULLY\n")