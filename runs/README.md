# Run files

A run file says **what** to compute; `config.toml` says **where** data, tools
and results live. Run files are versioned with the code, so every result can
be traced back to (and recomputed from) the exact file that produced it.

```
phantom stages                         # list stages
phantom stages classification.single   # parameters, defaults, available values
phantom new classification.single -o runs/my_run.toml   # template to fill in
phantom validate runs/my_run.toml      # check without running
phantom plan runs/my_run.toml          # what would run, what is up to date
phantom run runs/my_run.toml           # run it
```

## Format

```toml
[run]
name = "host_catboost"

[[steps]]
stage = "classification.single"
version = "3.0"
modality = "host"
vtool = ["geN", "VIB", "VS2"]   # a list = one job per value
model = "catboost"
```

- Several `[[steps]]` run in order; a failure stops the run.
- A **list** for a parameter marked "list = one job per value" expands into a
  grid; several such lists give every combination.
- Omitted parameters take their defaults (`phantom stages <stage>` shows them).
- Unknown parameters, wrong types and values not in `config.toml` are reported
  all at once, before anything runs.

## Re-running

`phantom run` skips jobs that are already up to date: same parameters, same
inputs (including the class labels in the metadata CSV), outputs still present.
Change one parameter and only the affected jobs run.

```
phantom run runs/my_run.toml --set model=rf          # one-off override
phantom run runs/my_run.toml --set model='["rf","xgb"]'
phantom run runs/my_run.toml --only classification.single
phantom run runs/my_run.toml --from 2                # from step 2 on
phantom run runs/my_run.toml --force                 # ignore stamps
```

Every output directory gets a `run.toml` with that job's resolved parameters
plus provenance (git commit, inputs). To recompute a result exactly:

```
phantom run <result dir>/run.toml --force
```

## Layout

- `runs/examples/` -- reference run files.
- Keep stable, rarely changed pipelines (`pipeline_v3.toml`) apart from
  small, frequently varied experiments (`experiments/*.toml`).
- Changing feature-engineering parameters means a **new feature version**
  (e.g. `3.1`), never overwriting an existing one.
