# Stationary Finite Distill Teacher-Occupancy Ablation

Config: `experiments/single_card_tradeoff/configs/stationary_finite_distill_teacher_occupancy_ablation.toml`

## Question

Does sampling the discrete `fsrs6_oracle_stationary_finite_distill` table by
exact stationary teacher occupancy improve the first-eight per-user distill
frontier compared with uniform exact-table supervision?

## Design

Both arms use the same budget and model:

- Users: `1,2,3,4,5,6,7,8`
- Teacher weights: `0,4,16,64,256,1024`
- Eval weights: `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Student: per-user residual `8x2`, 476 parameters per user
- Training: 128 epochs, 64 steps per epoch, 256 table samples per teacher weight
- Evaluation: 10,000 particles, CUDA, Markov review transition off

The control arm is `--per-user-supervision uniform_table`. The treatment arm is
`--per-user-supervision teacher_occupancy`, which samples `(S,D)` states from
the exact stationary teacher occupancy distribution independently within each
teacher cost weight. The loss, action grid, model, and rollout semantics are
unchanged.

## Results

| supervision | mean relative AUC | mean AUC | mean coverage | covered/target | train agreement | full-table agreement | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_table` | 12.02% | 4.7639 | 98.31% | 66/77 | 83.29% | 83.21% | 121.2s |
| `teacher_occupancy` | 9.45% | 3.2466 | 98.38% | 67/77 | 90.93% | 62.85% | 142.4s |

Exact-vs-distill evaluation gives the same distill frontier numbers and the
same exact teacher reference for both arms:

| scheduler | mean relative AUC | mean AUC | mean coverage | covered/target |
| --- | ---: | ---: | ---: | ---: |
| exact stationary finite | 13.13% | 5.1976 | 98.90% | 68/77 |
| distill, `uniform_table` | 12.02% | 4.7639 | 98.31% | 66/77 |
| distill, `teacher_occupancy` | 9.45% | 3.2466 | 98.38% | 67/77 |

Per-user relative AUC deltas are treatment minus control:

| user | uniform | teacher occupancy | delta |
| --- | ---: | ---: | ---: |
| 1 | 8.95% | 1.09% | -7.85 pp |
| 2 | 8.45% | -2.01% | -10.47 pp |
| 3 | 12.90% | 13.67% | +0.77 pp |
| 4 | 17.87% | 17.25% | -0.62 pp |
| 5 | 16.56% | 18.21% | +1.65 pp |
| 6 | 15.71% | 16.47% | +0.76 pp |
| 7 | 9.87% | 5.93% | -3.94 pp |
| 8 | 5.87% | 5.01% | -0.86 pp |

## Interpretation

Teacher-occupancy sampling improves agreement on the states it samples
directly, but it hurts the table-wide policy fit and lowers the actual tradeoff
frontier. The main regressions are users 1 and 2; user 2 falls below the FSRS6
baseline on mean relative AUC. Coverage is essentially unchanged, so this is not
a coverage/overlap artifact.

For the discrete stationary finite distill default, keep `uniform_table`.
Occupancy-weighted sampling is still useful as a diagnostic and may be useful in
a mixed objective, but using it alone overweights the on-policy high-density
states enough to damage off-density interpolation and cost-weight generalization.

## Evidence

- Comparison summary:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/comparison_summary.csv`
- Per-user deltas:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/by_user_delta.csv`
- Uniform train/eval:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/uniform_table/`
- Teacher-occupancy train/eval:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/teacher_occupancy/`
- Uniform exact-vs-distill:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/uniform_table_exact_vs_distill/`
- Teacher-occupancy exact-vs-distill:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/teacher_occupancy_exact_vs_distill/`

GPU monitor summaries reported no shared-memory spill in all four full-budget
train/eval tasks.
