from langgraph.graph import StateGraph, END

from agent_state import AgentState

from schema_engine.langgraph_node import schema_inference_node
from data_understanding.langgraph_node import data_understanding_node
from preprocess_1.langgraph_node import preprocess_1_node
from model_selector.langgraph_node import model_selector_node

# NEW
from preprocess_2.langgraph_node import preprocess_2_node
from model_intializer.langgraph_node import model_initializer_node
from ml_engine.langgraph_node import ml_training_node
from evaluation_engine.langgraph_node import evaluation_node
from retrain.langgraph_node import retrain_node

from tests.schema_mapping import extract_schema
from tests.json_printer import print_last_n_role_constants


from pathlib import Path


def select_dataset_folder(base_dir="uploaded_files"):
    base = Path(base_dir)

    if not base.exists():
        raise FileNotFoundError(f"{base_dir} not found")

    folders = [f for f in base.iterdir() if f.is_dir()]

    if not folders:
        raise ValueError("No dataset folders found")

    print("\nAvailable datasets:\n")

    for i, folder in enumerate(folders, start=1):
        print(f"{i}. {folder.name}")

    while True:
        try:
            choice = int(input("\nSelect dataset number: "))
            if 1 <= choice <= len(folders):
                selected = folders[choice - 1]
                break
            else:
                print("Invalid number")
        except ValueError:
            print("Enter a valid number")

    data_file = selected / "data.csv"
    metadata_file = selected / "metadata.jsonl"

    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found")

    if not metadata_file.exists():
        raise FileNotFoundError(f"{metadata_file} not found")

    print(f"\nSelected dataset: {selected.name}\n")

    return str(metadata_file), str(data_file)

METADATA_FILE, DATA_FILE = select_dataset_folder()

cats, target = extract_schema(METADATA_FILE, DATA_FILE)

# print("CATEGORICAL_COLUMNS = ", cats)
# if target:
#     print(f'TARGET_COLUMN = "{target}"')
# else:
#     print("TARGET_COLUMN = null")


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
# RUN
# --------------------------------------------------
if __name__ == "__main__":

    graph = build_graph()

    initial_state: AgentState = {
        "data_path": DATA_FILE,
        "categorical_columns": cats,
        "target_column": target,
        "preprocess_last_n": 1,
        "evaluation_last_n": 1,
    }

    result = graph.invoke(initial_state)

    print("\n=== FINAL STATE ===\n")
    print(result)

    # print("\n=== LAST CLASSIFICATION ===\n")
    # print_last_n_role_constants("data/data_classification.jsonl", n=1)
