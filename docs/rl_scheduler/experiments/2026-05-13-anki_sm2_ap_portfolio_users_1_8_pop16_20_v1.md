# Anki SM2 AP pop16 gen20 experiment report

Date: 2026-05-13

## Question

Evaluate whether a no-DR `anki_sm2_ap` portfolio can match or improve the
matched-budget `fsrs6_adr_portfolio_users_1_8_pop16_v1` result. The Anki SM2
adaptive-parameter scheduler searches the 7 runtime parameters from
`../srs-benchmark/models/anki.py`, with the two interval upper bounds capped at
`100` instead of `9999`.

The training budget is the same as the matched ADR run: users 1-8,
`population=16`, `offspring=16`, `generations=20`, `portfolio=16`, seed 42.
Training uses the fitted FSRS6 environment. Formal evaluation uses FSRS6 and
LSTM environments.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `anki_sm2_ap_portfolio_users_1_8_pop16_20_v1` | `anki_sm2_ap` | `anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The Anki SM2 AP run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml \
  --run-id anki_sm2_ap_portfolio_users_1_8_pop16_20_v1 \
  --skip-manifest \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Git commit recorded by artifacts: `275052e0d8f7eb3f9bd9f97b31ff128bf794de21`
- Worktree recorded by artifacts: `dirty=true`
- Python: 3.13.11
- PyTorch: 2.9.1+cu126
- CUDA device: NVIDIA GeForce RTX 4090 D
- Training elapsed: 330.8 seconds
- Sweep elapsed: 45.6 seconds
- Training peak allocated CUDA memory: 174,816,256 bytes
- Training peak reserved CUDA memory: 205,520,896 bytes
- Sweep peak allocated CUDA memory: 1,096,558,080 bytes
- Sweep peak reserved CUDA memory: 1,132,462,080 bytes
- Manual Windows shared GPU memory samples stayed below 1 GiB; max sampled
  single-adapter shared usage was 361.4 MiB, with 379.6 MiB summed across
  reported adapters.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user `HV delta` from
`analyze-pareto`, comparing each scheduler against the same staged FSRS6
baseline manifest. Positive budget-memory gain AUC means more remembered cards
at the same time budget. Negative memory-target regret AUC means reaching the
same memory target faster.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | budget-memory gain AUC | budget coverage | memory-target regret AUC | target coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| FSRS6 | Anki SM2 AP | -257,971 | -9.272% | 121 | -99.4 | 61/115, 90.562% span | +6.86 | 89/115, 88.426% span |
| FSRS6 | ADR | +96,880 | +3.482% | 128 | +65.4 | 101/115, 98.365% span | -2.92 | 106/115, 98.094% span |
| FSRS6 | Anki SM2 AP - ADR | -354,851 | -12.754 pp | -7 | -164.8 | -40 budgets | +9.78 | -17 targets |
| LSTM | Anki SM2 AP | -344,345 | -12.120% | 107 | -128.4 | 61/111, 92.620% span | +7.18 | 83/111, 86.108% span |
| LSTM | ADR | +55,849 | +1.966% | 125 | +28.1 | 92/111, 98.428% span | -1.02 | 102/111, 97.854% span |
| LSTM | Anki SM2 AP - ADR | -400,193 | -14.086 pp | -18 | -156.5 | -31 budgets | +8.20 | -19 targets |

The result is decisive. Anki SM2 AP loses to matched-budget ADR on every primary
external metric in both formal environments:

- FSRS6 HV moves from `+96,880` for ADR to `-257,971` for Anki SM2 AP.
- LSTM HV moves from `+55,849` for ADR to `-344,345` for Anki SM2 AP.
- FSRS6 budget-memory gain AUC moves from `+65.4` to `-99.4`.
- LSTM memory-target regret AUC moves from `-1.02` to `+7.18`.

## Per-User HV Delta

Anki SM2 AP is behind ordinary ADR for every user in both formal environments.
The largest losses are users 2 and 4.

| user | FSRS6 Anki SM2 AP HV delta | FSRS6 ADR HV delta | FSRS6 Anki - ADR | LSTM Anki SM2 AP HV delta | LSTM ADR HV delta | LSTM Anki - ADR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -59,270 | +9,894 | -69,164 | -60,225 | +7,161 | -67,385 |
| 2 | -102,383 | +27,715 | -130,098 | -152,432 | +13,315 | -165,747 |
| 3 | -7,074 | +3,745 | -10,820 | -8,201 | +3,364 | -11,565 |
| 4 | -72,924 | +37,079 | -110,002 | -88,553 | +19,119 | -107,671 |
| 5 | +216 | +8,964 | -8,747 | -5,331 | +4,110 | -9,441 |
| 6 | -8,504 | +6,327 | -14,831 | -19,056 | +6,492 | -25,547 |
| 7 | -6,827 | +1,238 | -8,065 | -6,997 | +733 | -7,731 |
| 8 | -1,206 | +1,918 | -3,124 | -3,550 | +1,556 | -5,105 |

## Diagnostics

The unweighted policy-point averages show why this variant can look tempting:
Anki SM2 AP often samples lower-time points and higher average memorized cards
than ADR. Those points are not efficient enough to form a competitive external
Pareto frontier.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | ---: | ---: | ---: | ---: |
| FSRS6 | Anki SM2 AP | 6709.1 | 41.90 | 21.77 | 137.54 |
| FSRS6 | ADR | 6553.0 | 50.42 | 25.42 | 195.90 |
| LSTM | Anki SM2 AP | 6563.5 | 45.66 | 17.99 | 155.27 |
| LSTM | ADR | 6431.5 | 60.86 | 22.88 | 255.29 |

Train-overfit passed for all users, and the final summed training HV gain was
`28,900` across users. That does not survive external Pareto evaluation.

Final training HV gain by user:

| user | final training HV gain |
| ---: | ---: |
| 1 | 4,368 |
| 2 | 5,974 |
| 3 | 2,780 |
| 4 | 7,921 |
| 5 | 4,715 |
| 6 | 1,923 |
| 7 | 83 |
| 8 | 1,136 |

## Conclusion

Do not promote `anki_sm2_ap`.

With the same pop16/off16/gen20/portfolio16 budget, Anki SM2 AP is a large
regression versus ordinary ADR:

- FSRS6: `-354,851` HV versus ADR.
- LSTM: `-400,193` HV versus ADR.
- FSRS6 budget-memory gain AUC: `-164.8` versus ADR.
- LSTM memory-target regret AUC: `+8.20` versus ADR.

The 7-parameter Anki SM2 search space can overfit the training simulator enough
to pass the local gate, but it does not provide a useful external Pareto
frontier against the matched FSRS6 ADR portfolio.

## Artifact Paths

- Anki SM2 AP run root:
  `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1`
- Anki SM2 AP analysis:
  `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Ordinary ADR comparison analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
