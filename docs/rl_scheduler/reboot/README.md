# RL Scheduler Reboot Documentation

This directory captures the restart plan for rebuilding RL scheduler experiment
infrastructure from a clean base branch.

Local note: this repository currently exposes `master`, not `main`, as the clean
base branch name. If the upstream branch is renamed later, read `main` and
`master` below as the same clean-base concept.

## Documents

- [Roadmap](./roadmap.md): phased implementation plan, deliverables, gates, and
  stop conditions.
- [SOP](./sop.md): standard operating procedure for infrastructure changes and
  formal experiment execution.
- [Best Practices](./best-practices.md): architecture, research-methodology,
  reproducibility, GPU, and reporting rules.
- [GPU Utilization Plan](./gpu-utilization-plan.md): concrete profiling,
  batching, and GPU evidence tasks.
- [Build vs Buy](./build-vs-buy.md): which custom infrastructure should be
  replaced by mature libraries, and which domain logic should remain custom.
- [Agent Skill Draft](./agent-skill/SKILL.md): a concise Codex skill draft for
  future agents working on this experiment infrastructure.

## Operating Principle

The restart should treat schema, provenance, gates, and reproducibility as
first-class infrastructure. Scheduler research code can iterate quickly only
after experiment identity, artifact contracts, baseline reuse, GPU guard
evidence, and external Pareto gates are machine-checkable.

The first implementation slice provides a stdlib typed schema scaffold under
`simulator/experiment_infra/` and keeps retention_sweep CSV output diagnostic
only by default.

The second slice adds a minimal TOML runner:

```bash
uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_linear_cmaes_users_1_8.toml --stage dry-run --run-id fsrs6_adr_linear_cmaes_users_1_8_v1
uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_linear_cmaes_users_1_8.toml --stage all --run-id fsrs6_adr_linear_cmaes_users_1_8_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_linear_cmaes_users_1_8/fsrs6_adr_linear_cmaes_users_1_8_v1
uv run python experiments/rl_scheduler/validate_artifact.py --metadata <artifact_metadata.json> --require-files
```

For any run that needs a current Pareto chart, the profile must include
`stage-baseline`, `sweep`, and `build-pareto`. Run `build-pareto` after the candidate sweep;
the PNG path is recorded in
`<output_root>/<run_id>/pareto/pareto_summary.json` as `plot_paths`, and the
default plot directory is `<output_root>/<run_id>/pareto/pareto_outputs/`.

`dry-run`, `preflight`, `stage-baseline`, `train-overfit`, `sweep`, `pareto`,
`select`, `aggregate`, and `reserved-test` are implemented. `train-overfit`,
`sweep`, `pareto`, `select`, `aggregate`, and `reserved-test` require
profile-specific command templates; the checked-in smoke TOML does not include
real trainer, sweep, Pareto, selector, aggregate, or reserved-test commands, so
`all` stops at the first missing command template until the profile is wired to
commands that emit valid scheduler metadata, JSONL logs, Pareto artifacts,
selection JSON, aggregate gate JSON, and reserved-test logs.

`all` runs configured stages in order, writes `all_summary.json`, and stops at
the first non-zero stage result.
