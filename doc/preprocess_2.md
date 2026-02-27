# Preprocess 2 Module (`preprocess_2`)

## What it is
`preprocess_2` is the model-dependent preprocessing stage. It consumes deferred actions from preprocess_1 and aligns transformations with selected model context.

## What it does
- Loads model selection decisions.
- Resolves deferred strategy columns.
- Executes strategies in strict order.
- Produces final training dataset and logs applied strategy metadata.

## How it works
1. `run_preprocess_2(last_n)` loads recent model selection records.
2. `PreprocessContext` resolves dataset path, primary model, deferred actions, and model-specific requirements.
3. `run_strategies(...)` executes in order:
   - `data_sanity`
   - `encoding`
   - `scaling`
   - `correlation_resolution`
   - `feature_engineering`
4. Final dataset is written as `<preprocessed_1>_final.csv` and tracked in `data/preprocess_2.jsonl`.

## Important functions
- `pipeline.run_preprocess_2(...)` / `process_one(...)`.
- `context.PreprocessContext.get_columns(...)`: deferred strategy routing.
- `dispatcher.run_strategies(...)`: ordered execution with column existence safety.
- Strategy implementations:
  - `data_sanity.apply(...)`
  - `encoding.apply(...)` (hybrid one-hot + frequency)
  - `scaling.apply(...)` (numeric validation/coercion)
  - `correlation.apply(...)` (quality-score-based pruning)
  - `text_vectorization.apply(...)` (TF-IDF)
