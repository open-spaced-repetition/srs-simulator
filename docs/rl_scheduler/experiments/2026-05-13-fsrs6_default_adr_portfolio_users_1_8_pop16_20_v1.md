# FSRS6 default ADR pop16 gen20 experiment report

Date: 2026-05-13

## Question

Evaluate whether `fsrs6_default_adr` improves or preserves the ADR Pareto
frontier when the ADR scheduler uses default FSRS6 weights internally, while
the training environment remains fitted FSRS6 and the formal evaluation
environments are FSRS6 and LSTM.

This isolates scheduler-side FSRS6 weight source. The matched comparison is the
ordinary `fsrs6_adr_portfolio_users_1_8_pop16_v1` run, which uses the same
pop16/off16/gen20/portfolio16 budget but keeps user-fitted FSRS6 scheduler
weights.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1` | `fsrs6_default_adr` | `fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The default-ADR run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml \
  --run-id fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1 \
  --skip-manifest \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Git commit recorded by artifacts: `92e3912d057dab7eb7aca3c6c3657f80b7b62425`
- Worktree recorded by artifacts: `dirty=true`
- Python: 3.13.11
- PyTorch: 2.9.1+cu126
- CUDA device: NVIDIA GeForce RTX 4090 D
- Training elapsed: 360.0 seconds
- Sweep elapsed: 43.2 seconds
- Training peak allocated CUDA memory: 174,585,344 bytes
- Training peak reserved CUDA memory: 199,229,440 bytes
- Sweep peak allocated CUDA memory: 1,130,132,480 bytes
- Sweep peak reserved CUDA memory: 1,153,433,600 bytes
- Windows shared GPU memory monitor max single adapter sample: 155.6 MiB
- Windows shared GPU memory monitor max summed sample: 173.8 MiB

Shared GPU memory stayed far below 1 GiB, so the run did not show a VRAM spill
signal.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user `HV delta` from
`analyze-pareto`, comparing the scheduler against the same FSRS6 baseline
manifest. Positive budget-memory gain AUC means more remembered cards at the
same time budget. Negative memory-target regret AUC means reaching the same
memory target faster.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | budget-memory gain AUC | budget coverage | memory-target regret AUC | target coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| FSRS6 | default ADR | -17,411 | -0.626% | 127 | -10.8 | 96/115, 98.029% span | -0.47 | 96/115, 93.432% span |
| FSRS6 | ADR | +96,880 | +3.482% | 128 | +65.4 | 101/115, 98.365% span | -2.92 | 106/115, 98.094% span |
| FSRS6 | default ADR - ADR | -114,291 | -4.108 pp | -1 | -76.2 | -5 budgets | +2.45 | -10 targets |
| LSTM | default ADR | -72,379 | -2.547% | 123 | -53.6 | 91/111, 98.458% span | +4.17 | 91/111, 91.953% span |
| LSTM | ADR | +55,849 | +1.966% | 125 | +28.1 | 92/111, 98.428% span | -1.02 | 102/111, 97.854% span |
| LSTM | default ADR - ADR | -128,227 | -4.513 pp | -2 | -81.7 | -1 budget | +5.19 | -11 targets |

The result is not close. Default-ADR loses to ordinary ADR on every primary
external metric in both environments:

- FSRS6 HV moves from `+96,880` for ADR to `-17,411` for default ADR.
- LSTM HV moves from `+55,849` for ADR to `-72,379` for default ADR.
- LSTM budget-memory gain AUC moves from `+28.1` to `-53.6`.
- LSTM memory-target regret AUC moves from `-1.02` to `+4.17`.

## Per-User HV Delta

Default ADR is behind ordinary ADR for every user in both formal environments.
The largest losses are users 2 and 4.

| user | FSRS6 default ADR HV delta | FSRS6 ADR HV delta | FSRS6 default - ADR | LSTM default ADR HV delta | LSTM ADR HV delta | LSTM default - ADR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +1,185 | +9,894 | -8,709 | -2,563 | +7,161 | -9,723 |
| 2 | -32,449 | +27,715 | -60,164 | -46,917 | +13,315 | -60,232 |
| 3 | -286 | +3,745 | -4,031 | -1,697 | +3,364 | -5,061 |
| 4 | +2,893 | +37,079 | -34,185 | -15,092 | +19,119 | -34,211 |
| 5 | +8,528 | +8,964 | -435 | +411 | +4,110 | -3,699 |
| 6 | +2,582 | +6,327 | -3,744 | -4,814 | +6,492 | -11,306 |
| 7 | -726 | +1,238 | -1,965 | -860 | +733 | -1,594 |
| 8 | +860 | +1,918 | -1,058 | -846 | +1,556 | -2,402 |

## Diagnostics

Default-ADR policy points are generally lower-time points, but the lower time
does not translate into a better Pareto frontier.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | ---: | ---: | ---: | ---: |
| FSRS6 | default ADR | 6530.7 | 39.92 | 23.37 | 141.44 |
| FSRS6 | ADR | 6553.0 | 50.42 | 25.42 | 195.90 |
| LSTM | default ADR | 6385.5 | 46.39 | 19.57 | 173.53 |
| LSTM | ADR | 6431.5 | 60.86 | 22.88 | 255.29 |

The training objective itself still found apparent gains in the default-weight
scheduler state space: final summed training HV gain was `209,475.7` across
users. That training HV should not be compared directly to ordinary ADR's
training HV as a quality signal, because the scheduler-side state and baseline
geometry differ. The external FSRS6/LSTM Pareto results are the relevant test,
and they reject the default-weight variant.

Final training HV gain by user:

| user | final training HV gain |
| ---: | ---: |
| 1 | 61,487 |
| 2 | 33,452 |
| 3 | 6,311 |
| 4 | 84,648 |
| 5 | 11,917 |
| 6 | 7,746 |
| 7 | 1,222 |
| 8 | 2,693 |

## Conclusion

Do not promote `fsrs6_default_adr`.

Changing only the ADR scheduler-side FSRS6 weights from user-fitted weights to
default FSRS6 weights is a material regression. It turns the matched pop16
portfolio from a positive external Pareto result into a negative one:

- FSRS6: `-114,291` HV versus ordinary ADR.
- LSTM: `-128,227` HV versus ordinary ADR.
- LSTM budget-memory gain AUC: `-81.7` versus ordinary ADR.
- LSTM memory-target regret AUC: `+5.19` minutes versus ordinary ADR.

The ordinary `fsrs6_adr` variant remains the correct pop16/off16/gen20 baseline.
`fsrs6_default_adr` is useful as a negative control showing that ADR's
scheduler-side FSRS6 state scale matters; replacing it with global default
weights loses the main Pareto advantage.

## Artifact Paths

- Default-ADR run root:
  `artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1`
- Default-ADR analysis:
  `artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Default-ADR GPU shared-memory monitor:
  `artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1/gpu_monitor/shared_gpu_memory.log`
- Ordinary ADR comparison analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
