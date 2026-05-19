# Markov Button Behavior Default-Off Plan

## Context

`button_usage.jsonl` currently carries two separate kinds of behavior data:

- Marginal button probabilities and rating costs.
- `long_term_transition`, a Markov transition model for review button behavior.

The simulator's plain `StochasticBehavior` default is Markov-off, but several
CLI and batched experiment paths default to loading `button_usage.jsonl` and
then implicitly pass `long_term_transition` into event or batched behavior. This
made many full-deck `rl_scheduler` results Markov-on while single-card
teacher/student/distill work stayed Markov-off. The new default must make
Markov review transitions opt-in everywhere while preserving per-user marginal
button probabilities and costs.

## Outcome

When the work is done, the following must be true:

- Default behavior is Markov-off in event, vectorized, batched sweep, portfolio
  training, portfolio evaluation, and single-card tradeoff entry points.
- `--button-usage` still loads per-user first/review/learning/relearning rating
  probabilities and state-specific costs by default.
- `long_term_transition` is used only when an explicit option is set, for
  example `--review-markov-transition` or
  `simulation.review_markov_transition = true`.
- Experiment metadata, config snapshots, CSV/JSON summaries, and logs record
  whether review Markov transitions were enabled.
- New Markov-off results are never mixed with legacy Markov-on results in
  baseline staging, Pareto building, tradeoff summaries, or published reports.
- The current ADR-vs-476 comparison is rerun under one consistent behavior
  model: Markov-off baseline, Markov-off ADR, and Markov-off 476-param distill
  evaluation.
- Legacy `rl_scheduler` formal reports remain interpretable as Markov-on
  results until rerun; they must not be cited as evidence for the new default.

## Verification Surface

Use these evidence surfaces to prove the work:

- Source checks:
  - `rg -n "review_markov|long_term_transition|load_usage|button_usage" simulate.py experiments simulator tests`
  - Confirm every behavior-construction path gates `long_term_transition` behind
    an explicit Markov flag.
- Unit/static checks:
  - `uv run python -m unittest tests.test_button_usage`
  - `uv run python -m unittest tests.test_scheduler_catalog`
  - Add and run focused tests for batched behavior-cost loading and relevant
    config parsing.
  - `uv run pyright`
  - `uv run ruff format`
- Regression diagnostics:
  - User 2 FSRS6 DR=0.96 and DR=0.98 with Markov-off should return close to the
    historical single-card baseline scale. Earlier diagnostics were about
    `0.98 -> ~720 min/day` Markov-off versus `~480 min/day` Markov-on.
  - The diagnostic output must include the command, seed, user id, desired
    retention, minutes/day, reviews/day, and `review_markov_transition=false`.
- Formal experiment artifacts:
  - New baseline DR manifest and baseline sweep logs with
    `review_markov_transition=false`.
  - New run roots for ADR, 476-param distill portfolio, and r4d1/e512 distill
    portfolio. Their `config_snapshot.toml`, `resolved_config.json`,
    `performance_summary.json`, `analysis_summary.json`, report Markdown, and
    GPU monitor summaries must all exist.
  - New `adr_vs_476_tradeoff_first8_users` outputs: per-user CSVs, combined
    CSVs, summary CSVs, and all configured PNG visualizations.
- Report/source material:
  - Published docs must state the Markov mode.
  - Any report still based on old artifacts must say legacy Markov-on or be
    excluded from the current conclusion.

## Constraints

Codex must preserve these constraints while implementing and rerunning:

- Do not disable `button_usage` as a whole. Only disable implicit use of
  `long_term_transition`.
- Do not change single-card teacher/student policy semantics unless a separate
  experiment explicitly opts into Markov state. Existing 476-param and r4d1/e512
  distill checkpoints should remain valid Markov-off single-card policies.
- Do not silently reuse old retention sweep logs whose metadata lacks the new
  Markov mode field.
- Do not delete shared `logs/retention_sweep` or historical artifacts as a
  cleanup shortcut. Write new run ids or new output roots instead.
- Do not push to remotes.
- Respect GPU guard evidence. Judge OOM/spill from GPU monitor artifacts, not
  only manual `nvidia-smi` screenshots.
- Keep implementation scoped. Avoid unrelated simulator refactors or scheduler
  math changes.
- Use `uv` for Python commands. Tests use `unittest`, not pytest.
- If README examples or user-facing CLI examples change, update the relevant
  README files.
- For any performance-impacting code path, record a baseline performance
  result by affected engine or state why the change is semantic/config-only and
  does not require a performance comparison.

## Boundaries

Codex may use and modify:

- This repository only:
  `/home/jarrett/open-spaced-repetition/srs-simulator`.
- Existing local data repositories referenced by current configs:
  `../Anki-button-usage` and `../srs-benchmark`.
- Existing experiment output roots:
  - `artifacts/rl_scheduler`
  - `artifacts/single_card_tradeoff`
  - `logs/retention_sweep`
- GPU/CUDA on this machine, with automatic GPU monitor artifacts enabled.
- Checked-in experiment TOML under `experiments/rl_scheduler/configs` and
  `experiments/single_card_tradeoff/configs`.
- Documentation under `docs/rl_scheduler` and `docs/single_card_tradeoff`.

Codex must avoid:

- Remote pushes.
- Destructive git operations such as `git reset --hard` or `git checkout --`
  unless explicitly requested.
- Manual edits to generated historical result files unless the task is to
  publish a new report or mark a report as legacy.
- Cross-repository edits outside the local data inputs named above.

## Implementation Plan

### 1. Add an explicit Markov review-transition option

Introduce one canonical boolean meaning:

```toml
[simulation]
review_markov_transition = false
```

CLI entry points should expose the same concept with an explicit opt-in flag,
for example:

```bash
--review-markov-transition
```

Default is always false. If false, ignore `usage["long_term_transition"]` even
when `--button-usage` is present.

Code paths to update:

- `simulate.py`
- `experiments/retention_sweep/run_sweep.py`
- `experiments/retention_sweep/run_sweep_users_batched.py`
- `experiments/single_card_tradeoff/tradeoff.py`
- `experiments/single_card_tradeoff/run_tradeoff_config.py`
- `simulator/batched_sweep/config.py`
- `simulator/batched_sweep/behavior_cost.py`
- `simulator/batched_sweep/runner.py`
- `simulator/batched_sweep/logging.py`
- `experiments/rl_scheduler/policy_search_common.py`
- `experiments/rl_scheduler/portfolio_training_common.py`
- `simulator/experiment_infra/runner.py`
- `simulator/experiment_infra/training_batch.py`
- Trainer entry points that default `--button-usage` to
  `DEFAULT_BUTTON_USAGE_PATH`.

Implementation rule: `load_usage()` should either accept a
`review_markov_transition: bool = False` parameter or return marginal data
separately from optional Markov weights. `build_behavior_cost()` should accept
`review_markov_success_weights: torch.Tensor | None`.

### 2. Record Markov mode in metadata and filters

Add `review_markov_transition` to relevant config snapshots, resolved configs,
simulation metadata, CSV summaries, and JSONL log metadata.

Any scanner that reads shared `logs/retention_sweep` must treat Markov mode as
part of the semantic identity of a log. After this change:

- New logs must include `review_markov_transition=false` or `true`.
- Missing field means legacy and must not match a new Markov-off run unless an
  explicit legacy mode is requested.
- `LogFilenameFilter` should still be used before JSONL metadata reads; Markov
  mode validation happens after filename filtering unless a filename dimension
  is added.

### 3. Update TOML profiles for new formal runs

Add `simulation.review_markov_transition = false` to the profiles that will be
rerun for the current conclusion:

- `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml`
- `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml`
- `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.toml`
- `experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml`

For reruns, prefer new run ids or output roots with a clear suffix, such as
`_markov_off`, rather than overwriting legacy artifacts.

### 4. Validate the default with a small diagnostic

Before full reruns, run a targeted diagnostic on user 2:

- FSRS6 baseline, user 2, DR=0.96, Markov-off.
- FSRS6 baseline, user 2, DR=0.98, Markov-off.
- Optional opt-in check: same two points with Markov-on to confirm the old
  faster baseline remains reproducible when explicitly requested.

This diagnostic decides whether implementation is correct before spending GPU
time on portfolios.

### 5. Rerun the required experiment chain

Run order:

1. Baseline DR selection under Markov-off.
2. Manifest-driven FSRS6 baseline sweep under Markov-off.
3. Formal ADR portfolio run under Markov-off.
4. Formal 476-param stationary finite distill portfolio run under Markov-off.
5. Formal r4d1/e512 stationary finite distill portfolio run under Markov-off.
6. `adr_vs_476_tradeoff_first8_users` under Markov-off using the new ADR run
   root and the existing single-card distill checkpoints.
7. Generate or refresh reports and visualizations.

Use `run_portfolio_workflow.py` when a profile has a baseline DR manifest and
the full formal flow is desired. Use `run_experiment.py --stage ...` when
debugging an individual formal stage.

### 6. Mark legacy results

Do not rerun every historical branch immediately. Instead:

- Treat existing formal `rl_scheduler` results as legacy Markov-on unless their
  metadata proves otherwise.
- Any report cited in the new conclusion must be rerun or explicitly excluded.
- If a legacy result remains useful as an ablation, document it as
  `review_markov_transition=true`.

## Rerun Matrix

### Required for the current conclusion

These must be rerun before claiming ADR versus 476-param distill under the new
default:

- Shared FSRS6 baseline DR selection and baseline sweep for users 1-8.
- `fsrs6_adr_portfolio_users_1_8_pop16_v1`.
- `fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1`.
- `fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1`.
- `adr_vs_476_tradeoff_first8_users`.

### Rerun only if cited again

These can remain legacy until needed:

- ADR gen10/gen30/v3/baseline16x5 variants.
- AP portfolio and AP CMA-ES runs.
- Anki SM2 AP portfolio.
- FSRS3 scheduler comparison.
- ADR-time and default-ADR portfolios.
- 1-128 user portfolio runs.
- LSTM-trained ADR portfolio.

### Not required for this default change

These do not need rerun solely because Markov defaults changed:

- Single-card 476-param teacher/student/distill checkpoint training.
- Single-card r4d1/e512 teacher/student/distill checkpoint training.
- `first8_exact_vs_distill` single-card evaluator runs.
- `low_param_direct_policy_search_*` single-card low-parameter searches.
- Sampling benchmarks, GPU probes, and smoke/debug runs, unless their reported
  performance numbers are being republished.

## Suggested Commands

After implementation, run focused validation first:

```bash
uv run python -m unittest tests.test_button_usage
uv run python -m unittest tests.test_scheduler_catalog
uv run pyright
uv run ruff format
```

Dry-run a formal profile:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml \
  --run-id fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off \
  --dry-run
```

Run the formal workflow after the diagnostic passes:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml \
  --run-id fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off
```

Repeat with the two distill portfolio profiles, using matching `_markov_off`
run ids.

Run the final tradeoff config after the new ADR run root is available:

```bash
uv run python experiments/single_card_tradeoff/run_tradeoff_config.py \
  --config experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml \
  --force
```

If the tradeoff config still points at the legacy ADR run root, update it before
running.

## Iteration Policy

After each attempt, Codex should decide the next step by this policy:

- If tests fail, fix the smallest implementation slice that explains the
  failure and rerun the focused test before broad checks.
- If a diagnostic FSRS6 point still matches the Markov-on scale, inspect
  metadata and behavior construction for that exact command before running more
  experiments.
- If a baseline staging step finds only legacy logs, generate fresh Markov-off
  baseline logs instead of relaxing metadata validation.
- If a GPU run spills shared memory above the configured threshold, reduce
  lanes according to environment-specific caps and rerun the same stage.
- If a formal stage fails, inspect `command_record.json`, stdout/stderr,
  `gate_summary.json`, and `gpu_monitor/summary.json` before changing code.
- If a report conclusion changes direction, prefer adding a short diagnostic or
  ablation over editing prose to fit prior expectations.
- If an optional legacy experiment becomes relevant to the conclusion, rerun
  that exact experiment under Markov-off rather than comparing against its
  Markov-on report.

## Blocked Stop Condition

Stop and report that no defensible path remains under the current limits when
any of these conditions holds:

- The code cannot make Markov transitions opt-in without changing single-card
  teacher/student policy semantics or invalidating existing distill checkpoints.
- New Markov-off metadata cannot be distinguished from old Markov-on logs in
  shared retention sweep roots.
- The user 2 FSRS6 diagnostic cannot reproduce the historical Markov-off scale
  after the behavior path and metadata are verified.
- Required baseline logs or source data from `../Anki-button-usage` or
  `../srs-benchmark` are missing and cannot be regenerated from available local
  inputs.
- GPU monitor artifacts show repeated spill/OOM even after applying safe lane
  caps (`fsrs6` around 8192 lanes, `lstm` around 1024 lanes) and there is no
  acceptable smaller batch plan.
- Formal artifacts are internally inconsistent: config snapshots, metadata,
  command records, and reports disagree on Markov mode, user ids, seeds, or
  run ids.
- Continuing would require deleting historical artifacts, pushing to remotes,
  or editing repositories outside the declared boundaries.

When stopping, include the exact failing command, artifact paths inspected,
observed evidence, and the smallest next decision needed from the user.
