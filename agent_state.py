from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Global state shared across graph nodes.
    """

    # -------- user inputs --------
    data_path: str
    categorical_columns: Optional[List[str]]
    target_column: Optional[str]

    # -------- outputs --------
    schema_result: Optional[Dict[str, Any]]
    data_understanding_result: Optional[Dict[str, Any]]

    # -------- preprocess 1 --------
    preprocess_last_n: Optional[int]
    preprocess_1_result: Optional[List[Dict[str, Any]]]

    # -------- model selector --------
    model_selector_last_n: Optional[int]
    model_selection_result: Optional[List[Dict[str, Any]]]

    preprocess_2_last_n: Optional[int]
    preprocess_2_result: Optional[List[Dict[str, Any]]]

    # -------- model initializer --------
    model_initializer_last_n: Optional[int]
    model_initialization_result: Optional[List[Dict[str, Any]]]

    # -------- ML training --------
    ml_training_last_n: Optional[int]
    ml_training_result: Optional[List[Dict[str, Any]]]

    # -------- evaluation --------
    evaluation_last_n: Optional[int]
    evaluation_result: Optional[List[Dict[str, Any]]]
    evaluation_error: Optional[str]
