# Model Selection Module (`model_selector`)

## What it is
`model_selector` is the arbitration layer that decides task type and ranks candidate model families using both rules and LLM reasoning.

## What it does
- Detects/normalizes problem type (classification, regression, unsupervised, semi-supervised).
- Computes dataset characteristics for ranking.
- Calls LLM for model recommendations and preprocessing suggestions.
- Fuses data-driven score with LLM support score.
- Logs final experiment manifest to `data/model_selection.jsonl`.

## How it works
1. `run_model_selection(last_n)` loads recent datasets prepared by earlier stages.
2. `build_model_plan(dataset)` performs:
   - rule-based problem detection,
   - semi-supervised detection,
   - LLM recommendation call,
   - ontology normalization + validation,
   - model ranking and preprocessing reconciliation.
3. Manifest includes hashes, top-k models, canonical problem definition, and LLM analysis.

## Important functions
- `pipeline.run_model_selection(...)`.
- `selector.build_model_plan(...)`: core orchestrator.
- `selector.canonical_problem_type(...)`, `validate_problem_configuration(...)`.
- `problem_type_detector.detect_problem_type(...)` and `detect_semi_supervised(...)`.
- `model_rules.rank_models(...)`: weighted fusion ranking.
- `llm_reasoner.llm_model_selection(...)`: Groq JSON reasoning call.
