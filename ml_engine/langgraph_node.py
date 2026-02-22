from typing import Dict, Any

from agent_state import AgentState
from ml_engine.pipeline import run_training


def ml_training_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node that executes ML training pipeline.
    """

    # how many recent experiments to train
    last_n = state.get("preprocess_last_n", 1)

    results = run_training(last_n) #type: ignore

    return {
        "ml_training_result": results
    }