# FSRS6 Cost-ADR Retention-Head Improvement

Date: 2026-05-27

This note analyzes the implied-retention distribution of the direct interval-head Cost-ADR policy and validates a retention-head improvement.

## Question

Can the weaker Cost-ADR `desired_retention` action head be improved by matching the interval head's implied retention distribution over `(S, D, cost)`?

## Distribution Analysis

Source interval-head run:

`artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off`

For users 1..8, 16 cost weights, 64 log-spaced stability points, and 32 difficulty points, interval-head outputs were converted back to retention with each user's FSRS6 forgetting curve.

Overall implied-retention distribution:

| metric | value |
|---|---:|
| min | 0.1885 |
| q0.1% | 0.2996 |
| q0.5% | 0.3816 |
| q1% | 0.4124 |
| q5% | 0.5232 |
| median | 0.8416 |
| q95% | 0.9830 |
| q99% | 0.9909 |
| q99.5% | 0.9920 |
| q99.9% | 0.9936 |
| max | 0.9953 |
| below 0.50 | 3.84% |
| above 0.98 | 6.05% |

Selected cost slices:

| cost weight | min | q1% | q5% | median | q95% | q99% | max | below 0.50 | above 0.98 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3912 | 0.5094 | 0.7097 | 0.9544 | 0.9906 | 0.9929 | 0.9953 | 0.87% | 17.22% |
| 16 | 0.3903 | 0.5078 | 0.6637 | 0.8878 | 0.9799 | 0.9897 | 0.9945 | 0.89% | 4.90% |
| 128 | 0.2944 | 0.4239 | 0.5301 | 0.7860 | 0.9516 | 0.9893 | 0.9944 | 2.98% | 2.72% |
| 384 | 0.2278 | 0.3816 | 0.4584 | 0.7308 | 0.9383 | 0.9891 | 0.9944 | 8.72% | 2.51% |
| 1024 | 0.1885 | 0.3228 | 0.4086 | 0.6828 | 0.9283 | 0.9889 | 0.9943 | 15.77% | 2.36% |

The old retention-head bounds `[0.50, 0.98]` cut off both ends of the interval-head behavior. The high-cost tail needs retention below 0.50; the low-cost tail often needs retention above 0.98. A wider `[0.30, 0.995]` range covers about the 0.1% to 99.9% implied-retention range.

## Method

Added script:

`experiments/rl_scheduler/fit_fsrs6_cost_adr_retention_init_from_interval.py`

The script:

- Reads per-user interval-head Cost-ADR policies.
- Loads each user's FSRS6 parameters.
- Samples the same 16 cost weights over an `(S, D)` grid.
- Converts interval outputs to implied retention through the FSRS6 forgetting curve.
- Fits a 24-parameter retention-head policy to those implied retentions.
- Writes one retention-head `policy.json` per user for use as `initial_policy_root`.

Generated initializer artifact:

`artifacts/rl_scheduler/fsrs6_cost_adr_retention_init_from_interval_first8_stdpre_r030_0995`

Fit settings:

- users: 1..8
- cost weights: formal 16-point grid
- grid: 64 stability x 32 difficulty
- retention bounds: `[0.30, 0.995]`
- epochs: 4096
- state feature count: 6, so 24 parameters

Initializer generation command:

```bash
uv run python experiments/rl_scheduler/fit_fsrs6_cost_adr_retention_init_from_interval.py \
  --interval-train-run-root artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off \
  --users 1,2,3,4,5,6,7,8 \
  --retention-min 0.30 \
  --retention-max 0.995 \
  --s-points 64 \
  --d-points 32 \
  --epochs 4096 \
  --learning-rate 0.03 \
  --torch-device cpu \
  --out-dir artifacts/rl_scheduler/fsrs6_cost_adr_retention_init_from_interval_first8_stdpre_r030_0995
```

Fit diagnostics:

| user | final loss | mean abs logit error | p95 abs logit error | below 0.30 | above 0.995 | below 0.50 | above 0.98 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0066 | 0.0478 | 0.1268 | 0.00% | 0.04% | 3.09% | 4.28% |
| 2 | 0.0047 | 0.0690 | 0.1864 | 0.00% | 0.00% | 0.00% | 25.12% |
| 3 | 0.0310 | 0.1851 | 0.5278 | 0.00% | 0.00% | 6.10% | 0.02% |
| 4 | 0.0091 | 0.0755 | 0.2505 | 0.00% | 0.00% | 5.09% | 14.17% |
| 5 | 0.1071 | 0.2639 | 0.6059 | 0.81% | 0.00% | 9.62% | 0.00% |
| 6 | 0.0039 | 0.0698 | 0.1700 | 0.00% | 0.00% | 0.27% | 0.00% |
| 7 | 0.0516 | 0.2419 | 0.6554 | 0.00% | 0.00% | 1.33% | 0.00% |
| 8 | 0.0580 | 0.2317 | 0.7925 | 0.00% | 0.00% | 5.22% | 4.78% |

## Validation Run

Config:

`experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml`

Run root:

`artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off`

Published report:

`docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.md`

Command:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off \
  --skip-manifest \
  --skip-baseline-sweep
```

Stages passed: dry-run, preflight, stage-baseline, train-overfit, sweep, build-pareto, analyze-pareto, report.

GPU monitor:

- train-overfit shared-memory spill: false, peak single-adapter shared memory 297.8 MiB
- sweep shared-memory spill: false, peak single-adapter shared memory 209.5 MiB

## Results

Against the old retention-head run:

| run | training HV sum | fsrs6 HV delta | fsrs6 relative time-save AUC | fsrs6 budget coverage | lstm HV delta | lstm relative time-save AUC | lstm budget coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| old retention head | 72,673 | 72,551 | 10.741% | 79.410% | 8,907 | 1.918% | 81.902% |
| interval-init wide retention head | 100,265 | 99,155 | 13.217% | 87.612% | 46,745 | 6.052% | 86.520% |
| delta | +27,591 | +26,604 | +2.476 pp | +8.202 pp | +37,839 | +4.135 pp | +4.619 pp |

Against the interval-head run:

| env | metric | interval-init wide retention head | interval head | delta |
|---|---:|---:|---:|---:|
| fsrs6 | training HV sum | 100,265 | 97,905 | +2,359 |
| fsrs6 | external HV delta | 99,155 | 97,711 | +1,444 |
| fsrs6 | relative same-budget memory lift AUC | 1.511% | 1.394% | +0.118 pp |
| fsrs6 | relative same-target time saved AUC | 13.217% | 12.793% | +0.424 pp |
| fsrs6 | budget span coverage | 87.612% | 87.137% | +0.475 pp |
| fsrs6 | target span coverage | 84.725% | 82.331% | +2.394 pp |
| lstm | external HV delta | 46,745 | 40,052 | +6,694 |
| lstm | relative same-budget memory lift AUC | 1.190% | 0.814% | +0.376 pp |
| lstm | relative same-target time saved AUC | 6.052% | 4.590% | +1.462 pp |
| lstm | budget span coverage | 86.520% | 86.716% | -0.196 pp |
| lstm | target span coverage | 94.002% | 91.662% | +2.340 pp |

## Interpretation

The method is effective for the retention head.

It fixes the two concrete failures found in the previous diagnosis:

- Bounds mismatch: `[0.30, 0.995]` covers almost all interval-head implied-retention mass, unlike `[0.50, 0.98]`.
- Weak starting point: fitting retention coefficients to interval-implied retention turns generation 0 from deeply negative HV into a strong candidate. Final training HV improves from 72.7k to 100.3k.

The experiment also shows that the remaining CMA-ES search contributed little beyond the fitted initializer. Most users kept their best policy from generation 0 or close to it. The improvement is therefore best understood as a better retention-head parameterization/initialization, not as evidence that the original retention-head CMA-ES search setup was adequate.

The result does not prove a standalone retention head is superior in a clean-slate setting, because the initializer is distilled from an interval-head policy. It does prove that a retention head can match or slightly exceed the interval-head external Pareto results when initialized from the interval policy's implied-retention surface and given retention bounds that match the observed distribution.

## Recommendation

Use interval-implied-retention fitting as the default way to initialize any future Cost-ADR retention-head ablation. Keep `[0.30, 0.995]` bounds for this family unless a broader distribution analysis contradicts it.

For a stricter retention-head study, the next ablation should remove the interval-derived preconditioning or add a retention-specific diagonal preconditioner, because this run still uses `first8_distill24_std_v1`.
