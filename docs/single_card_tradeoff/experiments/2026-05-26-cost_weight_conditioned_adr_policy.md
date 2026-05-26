# Cost-Weight-Conditioned ADR Policy

## Question

Can the single-card tradeoff use one ADR-style policy per user:

```text
policy(S, D, cost_weight) -> interval or desired retention
```

instead of a portfolio of one ADR policy per cost weight, and can that smaller
representation beat native ADR?

## Implementation

Implemented a compact FSRS6 cost-conditioned ADR runtime:

- `FSRS6CostConditionedADRPolicy`
- `FSRS6CostConditionedADRScheduler`
- vectorized multi-user batch scheduler ops for `tradeoff.py`
- `fsrs6_cost_adr_train.py` for supervised fitting

The deployed policy supports two monotone heads:

- interval head: `cost_weight` can only increase predicted interval;
- desired-retention head: `cost_weight` can only decrease predicted DR.

The tested interval families were:

| family | state features | coefficient groups | params per user |
| --- | ---: | ---: | ---: |
| compact | 6 | 4 | 24 |
| hinge | 8 | 4 | 32 |

Training used the existing first-eight
`fsrs6_oracle_continuous_stationary_finite_distill` checkpoints as teacher
tables. This avoids rerunning the exact continuous stationary finite oracle for
every experiment while preserving the same S/D/cost geometry.

## Configuration

Training:

- Environment: `fsrs6`
- Users: `1..8`
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Teacher: `continuous_stationary_finite_distill_first8_eval_weights_add_025_05_markov_off/user_{user_id}_policy.pt`
- Teacher cost weights: `0,4,16,64,256,1024`
- S/D table: `64 x 32`
- Epochs: `4096`
- Device: CUDA

Evaluation:

- Lifecycle: `1825` days
- Particles: `10000`
- Cost weights: `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- FSRS6 target retentions: `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`
- Comparisons: `fsrs6`, native `fsrs6_adr`, `fsrs6_cost_adr`, and for the
  32-param run the 476-param continuous stationary finite distill teacher.

## Artifacts

Training:

- `artifacts/single_card_tradeoff/fsrs6_cost_adr_distill_first8_24p_from_continuous_distill/`
- `artifacts/single_card_tradeoff/fsrs6_cost_adr_distill_first8_32p_from_continuous_distill/`
- `artifacts/single_card_tradeoff/fsrs6_cost_adr_distill_first8_32p_retention_from_continuous_distill/`

Evaluation:

- `artifacts/single_card_tradeoff/fsrs6_cost_adr_24p_first8_eval_vs_adr/`
- `artifacts/single_card_tradeoff/fsrs6_cost_adr_32p_first8_eval_vs_adr/`
- `artifacts/single_card_tradeoff/fsrs6_cost_adr_first8_comparison_summary.csv`
- `artifacts/single_card_tradeoff/fsrs6_cost_adr_policy_distribution_summary.csv`

GPU monitor summaries reported no shared-memory spill. The 32-param training
run took `47.8s`; the 32-param evaluation took `184.6s`; the 24-param
evaluation took `157.3s`.

## Fit Results

The interval head fit was much better than the direct desired-retention head.

| policy | action head | mean loss | mean abs error | p95 abs error |
| --- | --- | ---: | ---: | ---: |
| 24p | interval | 0.2723 | 0.5218 | 2.1111 |
| 32p | interval | 0.2700 | 0.5212 | 2.0758 |
| 32p | desired retention | 1.2982 | 1.6550 | 6.9097 |

The 24p and 32p interval fits are nearly identical. The extra hinge features did
not materially improve this teacher-distillation objective.

## Rollout Results

Mean same-target time-saved AUC vs `fsrs6`:

| scheduler | params/user | positive users | mean AUC | mean relative | mean coverage | min coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native `fsrs6_adr` | 19 x 6 = 114 | 8/8 | 4.9303 | 11.20% | 95.5% | 83.8% |
| cost-conditioned ADR 24p | 24 | 8/8 | 5.6469 | 11.36% | 80.3% | 41.4% |
| cost-conditioned ADR 32p | 32 | 8/8 | 5.5568 | 11.22% | 81.3% | 42.8% |
| continuous distill teacher | 476 | 8/8 | 5.4696 | 12.49% | 93.2% | 63.2% |

Direct shared-span AUC vs native ADR:

| scheduler | positive users | mean AUC | mean relative | mean coverage | min coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| cost-conditioned ADR 24p | 3/8 | 0.7811 | 0.06% | 82.5% | 41.6% |
| cost-conditioned ADR 32p | 3/8 | 0.7030 | -0.08% | 83.2% | 43.0% |

The compact single policy does beat native ADR on mean AUC versus `fsrs6`, even
with only 24 parameters per user. It is not yet a clean replacement for native
ADR because the win is concentrated in users 1, 2, and 4; users 3, 5, 6, 7, and
8 are behind ADR in direct shared-span comparison. Coverage is also materially
worse than ADR.

## Cost-Weight Distribution

For the 24-param interval policy, increasing cost weight produces the intended
aggregate shape:

| cost weight | mean retention | mean interval | lowest-S mean | highest-S mean | D=1 mean | D=10 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9324 | 89.65 | 0.6661 | 0.9837 | 0.9292 | 0.9335 |
| 4 | 0.9154 | 145.75 | 0.6660 | 0.9753 | 0.9160 | 0.9071 |
| 16 | 0.8916 | 331.20 | 0.6656 | 0.9477 | 0.8972 | 0.8714 |
| 64 | 0.8393 | 1403.36 | 0.6648 | 0.8539 | 0.8534 | 0.8022 |
| 256 | 0.7486 | 12930.32 | 0.6637 | 0.6491 | 0.7738 | 0.6954 |
| 1024 | 0.6364 | 283393.12 | 0.6625 | 0.4125 | 0.6718 | 0.5737 |

This matches the teacher geometry qualitatively:

- low stability remains pinned near the one-day interval retention;
- high stability carries most of the cost tradeoff;
- high difficulty accepts lower retention once cost is nontrivial;
- the cost input needs state-dependent slope, not a scalar offset.

The main mismatch is high-cost extrapolation. The interval head is not bounded
by `retention_min`, so it can produce extremely long intervals and canonical
retention below 0.5 in high-S states. That helps scalarized cost in some users
but narrows target coverage.

## Conclusion

The policy interface is feasible. A single 24-parameter policy per user can
cover the full cost-weight grid and beat native ADR on mean same-target AUC vs
`fsrs6` (`5.6469` vs `4.9303`) while using far fewer parameters than a 19-point
ADR portfolio (`24` vs `114` parameters per user).

It is not yet robust enough to declare victory over ADR. Direct vs ADR is
positive on only `3/8` users, and min coverage falls to about `41%`. The next
iteration should keep the interval head but add a coverage guard or rollout
fine-tuning objective, and probably penalize high-cost over-extension so the
frontier spans the same memory range as ADR.
