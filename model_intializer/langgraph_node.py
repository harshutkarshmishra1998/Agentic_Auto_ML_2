from typing import Dict, Any

from agent_state import AgentState
from model_intializer.pipeline import run_initializer


def model_initializer_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for model initialization pipeline.
    """

    last_n = state.get("preprocess_last_n", 1)

    results = run_initializer(last_n) #type: ignore

    return {
        "model_initialization_result": results
    }