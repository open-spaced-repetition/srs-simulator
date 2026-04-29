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
uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/reboot_smoke.toml --stage dry-run --run-id smoke
uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/reboot_smoke.toml --stage preflight --run-id smoke
```

Only `dry-run` and `preflight` are implemented at this point. Later stages must
remain explicit non-zero unsupported stages until their artifact contracts and
gates are implemented.
