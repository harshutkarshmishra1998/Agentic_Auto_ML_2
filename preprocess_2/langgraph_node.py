from typing import Dict, Any
from agent_state import AgentState

from preprocess_2.pipeline import run_preprocess_2


def preprocess_2_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for preprocess_2 pipeline.
    """

    last_n = state.get("preprocess_last_n", 1)

    outputs = run_preprocess_2(last_n) #type: ignore

    return {
        "preprocess_2_result": outputs
    }