# Native ADR vs Stationary Finite Distill, First Eight Users

## Question

When each first-eight benchmark user gets an independently trained native
single-card FSRS6 ADR policy for each default evaluation cost weight, how does
that frontier compare with the per-user `fsrs6_oracle_stationary_finite_distill`
frontier?

## Configuration

Reproduction config:
`experiments/single_card_tradeoff/configs/native_adr_vs_stationary_finite_distill_first8_default_eval_weights.toml`

Environment and evaluation:

- Environment: `fsrs6`
- Users: `1,2,3,4,5,6,7,8`
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Lifecycle: `1825` days
- Particles: `10000`
- Target retentions for `fsrs6`: `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`
- Evaluation cost weights: `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Stationary finite distill teacher cost weights: `0,4,16,64,256,1024`
- Review Markov transitions: off
- Device: CUDA

Training:

- Scheduler: native `fsrs6_adr`
- Policies: `8 users * 17 cost weights = 136`
- CEM population: `32`
- Elite count: `8`
- Generations: `64`
- Train particles per candidate: `64`
- Eval particles per policy: `10000`
- `job_batch_size`: `8192`, which batches all 136 jobs together. This is
  `278,528` lanes during CEM population evaluation and `1,360,000` lanes during
  final policy evaluation.

Comparison policy:
`artifacts/single_card_tradeoff/stationary_finite_distill_train_weights_add_4_only_first8_markov_off/user_{user_id}_policy.pt`

## Artifacts

ADR training outputs:
`artifacts/single_card_tradeoff/native_adr_first8_default_eval_weights_markov_off`

Evaluation outputs:
`artifacts/single_card_tradeoff/native_adr_vs_stationary_finite_distill_first8_default_eval_weights_markov_off`

Key files:

- `combined_results.csv`
- `combined_regret_auc.csv`
- `summary.csv`
- `mean_summary.csv`
- `combined_results_by_user/user_{user_id}.png`
- `same_target_time_saved_auc_by_user.png`
- `relative_time_saved_by_user.png`
- `span_coverage_by_user.png`

## Results

Mean same-target AUC vs `fsrs6`:

| scheduler | users | positive users | mean time-saved AUC | mean relative time saved | mean span coverage | min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fsrs6_adr` | 8 | 8 | 3.8414 | 9.18% | 83.29% | 34.09% |
| `fsrs6_oracle_stationary_finite_distill` | 8 | 8 | 4.9154 | 12.31% | 97.63% | 90.65% |

Per-user AUC vs `fsrs6`:

| user | ADR AUC | ADR relative | ADR coverage | distill AUC | distill relative | distill coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.3963 | 5.54% | 91.81% | 5.3292 | 8.75% | 92.74% |
| 2 | 5.8174 | 8.96% | 99.41% | 5.5756 | 8.81% | 99.80% |
| 3 | 1.9691 | 11.55% | 74.06% | 1.7829 | 12.54% | 99.92% |
| 4 | 13.3671 | 12.85% | 86.79% | 18.7485 | 18.53% | 90.65% |
| 5 | 3.1783 | 12.04% | 83.24% | 4.0024 | 16.13% | 97.93% |
| 6 | 2.2865 | 13.02% | 96.93% | 2.8098 | 16.32% | 99.98% |
| 7 | 0.3427 | 4.78% | 34.09% | 0.6282 | 11.79% | 100.00% |
| 8 | 0.3736 | 4.67% | 99.96% | 0.4467 | 5.58% | 99.98% |

Direct distill-vs-ADR shared-span comparison:

- Mean distill time-saved AUC vs ADR: `1.2011`
- Mean relative distill time saved vs ADR: `2.92%`
- Distill is ahead of ADR on users 1, 3, 4, 5, 6, 7, and 8 in direct shared-span
  AUC. User 2 is a small shared-span deficit at `-0.1900` AUC.

## GPU Monitor

ADR training:

- Runtime: `544.41s`
- Peak dedicated GPU memory: `2694 MiB`
- Peak shared GPU memory, single adapter: `202,182,656 bytes`
- Shared-memory spill detected: `false`

Tradeoff evaluation:

- Runtime: `173.84s`
- Peak dedicated GPU memory: `2683 MiB`
- Peak shared GPU memory, single adapter: `208,130,048 bytes`
- Shared-memory spill detected: `false`

## Conclusion

The corrected `add_4_only` teacher checkpoint changes the comparison
materially. The 476-parameter stationary finite distill has the stronger
default-grid frontier, with higher mean same-target time-saved AUC versus
`fsrs6` (`4.9154` vs `3.8414`), higher mean relative time saved (`12.31%` vs
`9.18%`), and wider coverage (`97.63%` mean, `90.65%` minimum).

Native single-card ADR remains a competitive direct per-cost-weight optimizer,
but its frontier is less complete: user 7 covers only `34.09%` of the `fsrs6`
memory span. The earlier user 2 negative distill result came from the older
sparse teacher checkpoint; with teacher cost weights `0,4,16,64,256,1024`, user
2 is positive versus `fsrs6`.

For this experiment, `job_batch_size=8192` is acceptable: it batches all 136 jobs
without shared-memory spill on the RTX 4090 D.
