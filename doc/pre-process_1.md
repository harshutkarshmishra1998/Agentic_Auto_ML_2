# Pre-process 1 Module (`preprocess_1`)

## What it is
`preprocess_1` is the model-independent preprocessing stage. It applies universally safe transformations before model-specific preprocessing.

## What it does
- Builds a deterministic preprocessing plan from column inspection outputs.
- Executes common operations (drop identifiers, missing value imputation, log transforms).
- Produces uniquely-versioned preprocessed datasets.
- Logs executed steps + deferred model-dependent actions.

## How it works
1. `run_preprocess_1(last_n)` loads latest `column_inspection.jsonl` records.
2. `build_plan(...)` converts column profile signals into structured `PreprocessPlan` and `Step`s.
3. `execute_plan(...)` applies each planned operation with mapped step executors.
4. Outputs are saved as `data/<dataset>_preprocessed_1_<uid>.csv`.
5. Execution metadata is appended to `data/preprocesses_1.jsonl`.

## Important functions
- `pipeline.run_preprocess_1(...)`: main driver.
- `planner.build_plan(...)`: plan generation logic.
- `executor.execute_plan(...)`: transformation execution engine.
- Step executors: `_drop_columns`, `_impute_numeric_median`, `_impute_categorical_mode`, `_log_transform`.
- `logger.append_log(...)`: logs steps + deferred strategy-by-column requirements.
