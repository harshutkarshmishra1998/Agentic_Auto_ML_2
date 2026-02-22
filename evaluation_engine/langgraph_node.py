from typing import Dict, Any

from agent_state import AgentState
from evaluation_engine.pipeline import run_evaluation_pipeline


def evaluation_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for evaluation pipeline.

    Uses preprocess_last_n as evaluation batch size.
    """

    last_n = state.get("preprocess_last_n")

    if last_n is None:
        return {
            "evaluation_result": None,
            "evaluation_error": "preprocess_last_n not provided"
        }

    results = run_evaluation_pipeline(last_n=last_n)

    return {
        "evaluation_result": results
    }