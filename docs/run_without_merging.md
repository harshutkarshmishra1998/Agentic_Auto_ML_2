# Running the fix without merging to `main`

Use one of these options to run the latest fix before merging to `main`.

## Option 1: Check out the branch directly (recommended)

```bash
git fetch origin
# replace <branch-name> with the PR/feature branch
git switch <branch-name>
python -m run_agent
```

## Option 2: Test a specific commit in detached HEAD

```bash
git fetch origin
# commit from this fix
git switch --detach 42e4b60
python -m run_agent
```

## Option 3: Apply only this fix onto your current branch

```bash
git fetch origin
# from your working branch (can be main or any other branch)
git cherry-pick 42e4b60
python -m run_agent
```

## Option 4: Pull only this PR branch into a local temp branch

```bash
git fetch origin
# start from current main state, but keep work isolated
git switch -c test-bool-fix origin/main
git merge --no-ff <branch-name>
python -m run_agent
```

## Verify the issue is gone

If your previous crash was from boolean quantile handling in `data_understanding/column_profiler.py`, rerun:

```bash
python -m run_agent
```

You can also run the regression test:

```bash
pytest -q tests/test_column_profiler.py
```
