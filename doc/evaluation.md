# Evaluation Module (`evaluation_engine`)

## What it is
`evaluation_engine` is the post-training analysis module that computes derived metrics and determines retraining need.

## What it does
- Loads latest experiment records.
- Extracts raw training/validation metrics from flexible schemas.
- Derives additional metrics (currently classification-centric from confusion matrix).
- Produces retraining recommendation logic.
- Appends evaluation outputs to evaluation log.

## How it works
1. `run_evaluation_pipeline(last_n)` reads latest experiments.
2. It resolves model/dataset/artifact metadata through tolerant key lookup.
3. `derive_metrics(...)` computes derived metrics.
4. Validation gap is computed when possible.
5. `analyze_retraining_need(...)` returns decision payload (`should_retrain`, confidence, reasons).
6. Evaluation record is appended via `append_evaluation(...)`.

## Important functions
- `pipeline.run_evaluation_pipeline(...)`.
- `metric_computation.derive_metrics(...)` and `compute_classification(...)`.
- `retraining_llm.analyze_retraining_need(...)`: rule-based retrain trigger logic.
- `experiment_loader.load_last_n_experiments(...)`.
- `evaluation_logger.append_evaluation(...)`.
- `utils.get_first_existing(...)`: robust schema compatibility helper.
