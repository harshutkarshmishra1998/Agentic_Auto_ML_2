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

from patch.clear_data_folder import clear_project_data_dir
from patch.copy_dataset import copy_dataset_to_user_uploads
from patch.llm_response import analyze_pipeline_directory
from parser.pipeline import run_pipeline

DATA_FILE = "uploaded_files/churn/data.csv"

cats = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
target = 'CustomerChurned'

clear_project_data_dir()
copy_dataset_to_user_uploads(DATA_FILE)

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

    run_pipeline()

    llm_response = analyze_pipeline_directory()

    print("\n=== LLM RESPONSE ===\n")
    print(llm_response)
