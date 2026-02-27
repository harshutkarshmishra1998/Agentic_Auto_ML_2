# Retrain Module (`retrain`)

## What it is
`retrain` is the closed-loop orchestration module that keeps training-evaluation cycles running when evaluation indicates retraining is required.

## What it does
- Reads latest evaluation decision.
- If retraining is needed, seeds a new preprocess-2 compatible entry.
- Regenerates initialization, adjusts hyperparameters per round, retrains, and re-evaluates.
- Repeats until convergence or max round limit.

## How it works
1. `run_retraining_loop(max_rounds)` loads latest `evaluation.jsonl` entry.
2. If `should_retrain=True`, it:
   - resolves best model from model selection history,
   - appends preprocess_2 seed entry,
   - runs initializer,
   - appends adjusted initialization (`_adjust_init_params(...)`),
   - runs training + evaluation again.
3. Loop exits when retraining not needed anymore or `max_rounds` reached.

## Important functions
- `pipeline.run_retraining_loop(...)`: iterative control loop.
- `_resolve_best_model(...)`: best-model lookup from prior selections.
- `_adjust_init_params(...)`: model-family-specific tuning updates.
- `_append_adjusted_initialization(...)`: stores round metadata + adjusted params.
- `_append_preprocess_2_seed_entry(...)`: compatibility bridge into downstream stages.
