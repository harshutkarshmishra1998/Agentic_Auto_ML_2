from typing import Any, Dict

from agent_state import AgentState
from retrain.pipeline import run_retraining_loop


def retrain_node(state: AgentState) -> Dict[str, Any]:
    max_rounds = state.get("retrain_max_rounds", 5)
    result = run_retraining_loop(max_rounds=max_rounds)

    return {"retrain_result": result}
