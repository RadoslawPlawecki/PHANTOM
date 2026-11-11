# Regression harness (menu -> run files)

Proves that each `phantom` stage reproduces, file for file, what the old
interactive menu produced on the real data.

```
python tools/regression/regress.py list
python tools/regression/regress.py baseline <case|prefix> ... [--repeat]   # legacy path
python tools/regression/regress.py check    <case|prefix> ...              # new stage
```

- A **case** (`cases.py`) is one pipeline action, answered identically on both
  paths: scripted answers to the menu's prompts (legacy) and the equivalent
  `[[steps]]` table (new). Choices deliberately take non-default branches.
- Both paths run in a sandbox workspace, `<workspace>/.phantom/regression/sandbox`;
  real inputs are symlinked in and fingerprinted before/after -- the run aborts
  if anything under them changed.
- Outputs are reduced to a manifest (path -> SHA-256) under
  `.phantom/regression/baselines/`; small text outputs are kept beside it for
  diffing. PDFs are compared by their rendered pages (matplotlib embeds usetex
  fonts under random subset names); text files also by a hash with numbers
  rounded to 10 significant digits (summation-order noise).
- Determinism shims, applied identically to both paths: a seeded Optuna
  sampler, a seeded PERMANOVA, reduced Optuna trial budgets,
  `SOURCE_DATE_EPOCH`. `--repeat` refuses a baseline whose two legacy runs
  differ.

The legacy half disappears with the menu; recorded baselines remain the
reference `check` compares against. Baselines are tied to the real data on
the machine they were recorded on and are not committed.
