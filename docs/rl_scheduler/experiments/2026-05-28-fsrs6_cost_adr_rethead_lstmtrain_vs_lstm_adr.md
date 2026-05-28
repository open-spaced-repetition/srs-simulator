# LSTM-trained Cost-ADR Retention Head vs LSTM-trained ADR

Date: 2026-05-28

## Question

Train `fsrs6_cost_adr` with the default desired-retention head directly in the
LSTM environment, using no coefficient preconditioning, pop16/gen20, and
`sigma0 = 1.0`, then compare it against the ADR portfolio trained in the LSTM
environment.

## Runs

| run | config | training env | budget |
| --- | --- | --- | --- |
| Cost-ADR R head | `fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml` | `lstm` | CMA-ES pop16/gen20, 16 cost weights |
| ADR | `fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml` | `lstm` | SMS-EMOA pop16/off16/gen20, portfolio16 |

Commands:

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml \
  --stage all \
  --run-id fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1
```

Both runs passed `dry-run`, `preflight`, `stage-baseline`, `train-overfit`,
`sweep`, `build-pareto`, and `analyze-pareto`.

## External Pareto Results

Primary metric is scheduler-only hypervolume delta against the FSRS6 baseline
frontier. Time-save and memory-lift AUCs are interpolation diagnostics over the
common covered spans reported by `analysis_summary.json`.

| eval env | scheduler | HV delta sum | HV delta / baseline | same-target time-save AUC | relative time-save AUC | target span coverage | same-budget memory-lift AUC | relative memory-lift AUC | budget span coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | Cost-ADR R head | 42,273 | 1.367% | 1.49 | 0.491% | 87.629% | -49.75 | -0.634% | 94.553% |
| FSRS6 | ADR | 36,378 | 1.177% | 1.18 | 4.329% | 71.702% | 44.54 | 0.670% | 83.258% |
| LSTM | Cost-ADR R head | 122,021 | 3.790% | 9.67 | 16.181% | 94.307% | 129.82 | 2.028% | 88.688% |
| LSTM | ADR | 106,540 | 3.309% | 10.38 | 11.961% | 84.108% | 66.10 | 0.998% | 80.996% |

On the target LSTM environment, Cost-ADR beats LSTM-trained ADR on primary
hypervolume by 15,481 HV (+14.53% relative to ADR). It also has higher
relative time-save AUC and broader target coverage, although ADR has a slightly
higher raw time-save AUC over its narrower covered span.

## Per-user LSTM HV Delta

| user | Cost-ADR R head | ADR | Cost-ADR minus ADR |
| ---: | ---: | ---: | ---: |
| 1 | 24,922 | 18,011 | +6,912 |
| 2 | 45,705 | 50,110 | -4,405 |
| 3 | 3,001 | 3,995 | -994 |
| 4 | 27,396 | 19,608 | +7,788 |
| 5 | 3,920 | 3,829 | +91 |
| 6 | 11,020 | 8,975 | +2,045 |
| 7 | 927 | 859 | +68 |
| 8 | 5,130 | 1,154 | +3,977 |

Cost-ADR wins 6 of 8 users. ADR is stronger on users 2 and 3.

## GPU Monitor

| run | stage | shared-memory spill | peak shared memory | peak `nvidia-smi` memory |
| --- | --- | --- | ---: | ---: |
| Cost-ADR R head | train-overfit | false | 268,001,280 bytes | 15,295 MiB |
| Cost-ADR R head | sweep | false | 221,855,744 bytes | 15,105 MiB |
| ADR | train-overfit | false | 229,023,744 bytes | 10,942 MiB |
| ADR | sweep | false | 222,515,200 bytes | 10,835 MiB |

No stage exceeded the 1 GiB shared-memory spill threshold.
