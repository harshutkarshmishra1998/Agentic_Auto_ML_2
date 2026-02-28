def analyze_retraining_need(context: dict) -> dict:
    """
    Decide whether retraining is required.

    context may contain:
        model
        task
        metrics
        training_time
        validation_gap (can be None)
        validation_strategy
    """

    task = (context.get("task") or "").lower()
    strategy = (context.get("validation_strategy") or {}).get("type")
    val_gap = context.get("validation_gap")

    # Unsupervised tasks: no train/val gap expected
    if task in {"clustering", "unsupervised"}:
        return {
            "validation_gap": val_gap,
            "should_retrain": False,
            "confidence": 0.85,
            "reasons": [
                "unsupervised task does not use train-validation generalization gap"
            ],
            "llm_summary": (
                "Validation gap is not applicable for clustering runs "
                f"(strategy={strategy or 'fit_all'})."
            )
        }

    # Case 1 — no validation info
    if val_gap is None:
        return {
            "validation_gap": val_gap,
            "should_retrain": False,
            "confidence": 0.3,
            "reasons": ["validation metrics unavailable"],
            "llm_summary": (
                "Cannot assess overfitting because train/validation scores "
                "are unavailable. Monitoring recommended."
            )
        }

    # Case 2 — overfitting detected
    if val_gap > 0.1:
        return {
            "validation_gap": val_gap,
            "should_retrain": True,
            "confidence": 0.7,
            "reasons": ["validation performance significantly worse than training"],
            "llm_summary": "Model shows signs of overfitting."
        }

    # Case 3 — stable performance
    return {
        "validation_gap": val_gap,
        "should_retrain": False,
        "confidence": 0.6,
        "reasons": ["training and validation performance aligned"],
        "llm_summary": "Model performance stable. No retraining needed."
    }
