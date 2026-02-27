# Model Initializer Module (`model_intializer`)

## What it is
`model_intializer` generates model-specific initial hyperparameters from preprocessing outcomes and dataset signals.

## What it does
- Reads latest preprocess-2 logs.
- Extracts compact signals (size, dimensionality, encoding usage, overfit risk).
- Applies rule-based default parameter generators for each model family.
- Appends initialization records to `data/model_initialization.jsonl`.

## How it works
1. `run_initializer(last_n)` loads last preprocess records.
2. `extract_signals(exp)` computes `DatasetSignals` from rows/cols and strategy logs.
3. `MODEL_RULES[model](signals)` selects the initializer function.
4. Output record is persisted (append-only) with model, params, signals, task, and dataset path.

## Important functions
- `pipeline.run_initializer(...)`: public entrypoint.
- `signal_extractor.extract_signals(...)`: feature/signal derivation.
- `rules_engine.MODEL_RULES`: model-to-initializer mapping.
- `model_defaults.init_random_forest(...)`, `init_boosting(...)`, `init_linear(...)`, `init_kmeans(...)`, `init_dbscan(...)`.
- `pipeline._load_json_records(...)`: robust multi-format JSON reader.
