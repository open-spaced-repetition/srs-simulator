# Stationary Finite Policy Visualization

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

What action patterns does the exact stationary finite policy table learn after removing action retentions below 0.5?

## Evidence

The visualization report is generated from `action_summary.csv` and the exact-vs-distill table comparison CSV.

Source artifacts:
- `stationary_finite_policy_action_summary`: `artifacts/single_card_tradeoff/stationary_finite_policy_viz/action_summary.csv`
- `stationary_finite_policy_distill_comparison`: `artifacts/single_card_tradeoff/stationary_finite_policy_viz/distill_exact_comparison.csv`
- `stationary_finite_policy_findings`: `artifacts/single_card_tradeoff/stationary_finite_policy_viz/findings.md`

## Results

### Exact table action summary

| weight | modal retention | modal share | mean action retention | normalized entropy | iterations |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.98 | 55.37% | 0.9238 | 0.646 | 3 |
| 16 | 0.93 | 21.53% | 0.8929 | 0.833 | 2 |
| 64 | 0.93 | 23.10% | 0.8638 | 0.853 | 3 |
| 256 | 0.50 | 24.85% | 0.7608 | 0.774 | 5 |
| 1,024 | 0.50 | 52.64% | 0.6541 | 0.651 | 8 |

### Distill vs exact table

| weight | exact match | distill lower | distill higher | mean abs retention diff | mean retention diff |
| --- | --- | --- | --- | --- | --- |
| 0 | 71.19% | 6.84% | 21.97% | 0.0218 | 0.0171 |
| 16 | 59.47% | 16.46% | 24.07% | 0.0369 | 0.0262 |
| 64 | 48.68% | 28.22% | 23.10% | 0.0791 | -0.0441 |
| 256 | 54.64% | 17.48% | 27.88% | 0.0480 | -0.0034 |
| 1,024 | 63.43% | 17.97% | 18.60% | 0.0578 | -0.0135 |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| visualize_stationary_finite_policy | 7/7 |

## Conclusion

The exact policy remains state-sensitive after clipping. The distill mostly tracks the table, but exact-cell agreement is lowest around the mixed `w=64` region.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-stationary_finite_policy_viz.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/stationary_finite_policy_viz.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
