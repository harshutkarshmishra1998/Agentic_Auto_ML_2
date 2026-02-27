# ML Engine Module

## What it is
`ml_engine` is the training runtime. It instantiates selected models, performs validation, persists artifacts, and logs full experiment metadata.

## What it does
- Loads model initializations and aligned target columns.
- Resolves task type with safety overrides.
- Uses template registry to construct model family estimators.
- Runs validation strategy (holdout/kfold/stratified/LOO/fit-all).
- Saves model artifacts and appends evaluator-ready experiment logs.

## How it works
1. `run_training(last_n)` iterates over initialization + target pairs.
2. Dataset is loaded via `load_dataset(...)`; task resolved with `_infer_task(...)` + `_resolve_task(...)`.
3. Template class from `TEMPLATE_REGISTRY` is run, returning model + results + strategy.
4. Trained model artifact is written using `save_model_artifact(...)`.
5. Full experiment record is appended by `log_experiment(...)` to `data/ml_experiments.jsonl`.

## Important functions
- `pipeline.run_training(...)`.
- `validation_strategy.choose_strategy(...)` and `run_validation(...)`.
- `validation_strategy._prepare_features(...)`: categorical-safe numeric preparation.
- `logger.log_experiment(...)`: unified task-aware metrics schema + lineage.
- `artifacts.save_model_artifact(...)`: structured model persistence.
- Template classes in `ml_engine/templates/*` (RandomForest, XGBoost, LightGBM, Logistic, Linear, KMeans, DBSCAN).
