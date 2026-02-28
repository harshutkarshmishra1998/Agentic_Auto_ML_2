from typing import Dict, Any

from agent_state import AgentState
from evaluation_engine.pipeline import run_evaluation_pipeline


def evaluation_node(state: AgentState) -> Dict[str, Any]:
    last_n = (
        state.get("evaluation_last_n")
        or state.get("ml_training_last_n")
        or state.get("preprocess_last_n")
    )

    if last_n is None:
        return {
            "evaluation_result": None,
            "evaluation_error": "No evaluation batch size provided"
        }

    results = run_evaluation_pipeline(last_n=last_n)

    return {
        "evaluation_result": results
    }