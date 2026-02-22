def analyze_retraining_need(context: dict) -> dict:
    """
    Decide whether retraining is required.

    context may contain:
        model
        task
        metrics
        training_time
        validation_gap (can be None)
    """

    val_gap = context.get("validation_gap")

    # -----------------------------
    # Case 1 — no validation info
    # -----------------------------
    if val_gap is None:
        return {
            "should_retrain": False,
            "confidence": 0.3,
            "reasons": ["validation gap unavailable"],
            "llm_summary": (
                "Cannot assess overfitting because validation performance "
                "is not available. Monitoring recommended."
            )
        }

    # -----------------------------
    # Case 2 — overfitting detected
    # -----------------------------
    if val_gap > 0.1:
        return {
            "should_retrain": True,
            "confidence": 0.7,
            "reasons": ["validation performance significantly worse than training"],
            "llm_summary": "Model shows signs of overfitting."
        }

    # -----------------------------
    # Case 3 — stable performance
    # -----------------------------
    return {
        "should_retrain": False,
        "confidence": 0.6,
        "reasons": ["training and validation performance aligned"],
        "llm_summary": "Model performance stable. No retraining needed."
    }