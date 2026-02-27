# Schema Engine Module

## What it is
`schema_engine` is the first inference layer that reads a dataset and assigns semantic roles to each column (for example: numeric, categorical, identifier, datetime, target).

## What it does
- Loads tabular data from multiple file formats.
- Profiles each column (type, uniqueness, missingness, sample values, datetime parseability, etc.).
- Performs deterministic role inference.
- Uses LLM fallback only for ambiguous columns.
- Persists outputs into `data/data_classification.jsonl` and user directives into `data/user_input.jsonl`.

## How it works
1. `run_schema_inference(...)` loads the dataset and validates user-provided categorical/target columns.
2. It profiles all columns with `profile_dataframe(...)`.
3. For each feature, deterministic rules assign role/confidence (`deterministic_role(...)`).
4. If ambiguity is detected (`is_ambiguous(...)`), it calls Groq-based resolver (`resolve_with_llm(...)`).
5. It exports an append-only classification record.

## Important functions
- `pipeline.run_schema_inference(...)`: module entrypoint.
- `pipeline._validate_user_inputs(...)`: prevents invalid user column declarations.
- `profiler.profile_dataframe(...)`: builds per-column profiling objects.
- `deterministic.deterministic_role(...)`: fast rule engine for semantic roles.
- `llm_resolver.resolve_with_llm(...)`: LLM arbitration for uncertain roles.
- `exporter.export_schema_result(...)` and `export_user_inputs(...)`: lineage logging.
