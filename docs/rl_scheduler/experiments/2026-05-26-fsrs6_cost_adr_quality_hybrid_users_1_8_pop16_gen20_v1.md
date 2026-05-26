# FSRS6 Cost-ADR quality-aware hybrid first-eight report

Date: 2026-05-26

Machine summaries:

- Coverage baseline summary: `artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- Quality v2 summary: `artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- Hybrid summary: `artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- Hybrid selection manifest: `artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off/hybrid_selection_manifest.json`
- Hybrid all summary: `artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off/all/all_summary.json`

## Question

Improve Cost-ADR after the coverage-aware run showed high coverage but weak
Pareto quality. The main failure mode was that the coverage penalty rewarded
wide curves even when many points were dominated by the FSRS6 baseline.

## Iterations

### Quality v1

Code change: add quality-aware coverage diagnostics to
`train_cmaes_fsrs6_cost_adr.py`:

- optionally exclude baseline-dominated candidate points from coverage
  computation;
- optionally subtract a baseline-HV-scaled dominated-point soft penalty;
- record dominated candidate counts and coverage candidate counts in progress and
  metrics.

Users 1-2 looked promising: FSRS6 HV delta improved from `28,991` to `34,190`
and relative same-target time-save AUC improved from `9.18%` to `12.36%`.

The same settings were too strict on users 1-8. Training completed but failed
the overfit gate at `6/8` users; users 5 and 8 never found positive training HV.
GPU monitor showed no spill, so this was an objective/search failure, not a CUDA
resource issue.

### Quality v2

Quality v2 kept dominated points in the coverage span calculation, but applied a
small dominated-point soft penalty:

- `coverage_filter_baseline_dominated = false`
- `coverage_dominated_point_penalty_weight = 0.005`

This restored training pass for all users, including user 5 and user 8, but did
not improve deployment metrics by itself. FSRS6 HV delta dropped to `36,004`,
and LSTM HV delta dropped to `-12,194`.

### Hybrid Selector

The hybrid selector uses only training-time evidence: for each user, choose the
artifact with the larger training `best_hypervolume_delta` between the prior
coverage-aware run and quality v2. It does not use FSRS6/LSTM sweep results.

| user | selected artifact | coverage train HV | quality v2 train HV |
| ---: | --- | ---: | ---: |
| 1 | quality v2 | 12,240 | 12,609 |
| 2 | coverage | 23,416 | 22,303 |
| 3 | coverage | 3,876 | 3,446 |
| 4 | quality v2 | 22,398 | 25,625 |
| 5 | quality v2 | 7,133 | 8,179 |
| 6 | quality v2 | 3,588 | 3,626 |
| 7 | coverage | 1,918 | 1,871 |
| 8 | coverage | 2,350 | 0 |

Hybrid verification command:

```bash
uv run python experiments/rl_scheduler/build_fsrs6_cost_adr_hybrid.py \
  --source coverage=artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off \
  --source quality_v2=artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1_markov_off \
  --baseline-run-root artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off \
  --output-run-root artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off \
  --users 1-8

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off
```

All hybrid stages passed: `dry-run`, `preflight`, `sweep`, `build-pareto`, and
`analyze-pareto`. The hybrid train root is assembled from existing per-user
artifacts, so it does not rerun `train-overfit`.

Hybrid sweep GPU monitor: `shared_memory_spill_detected=false`, peak
single-adapter shared memory `181,882,880` bytes, summed shared memory
`206,790,656` bytes, and `batch_lanes=256`.

## Direct Comparison

Positive HV, same-budget memory lift, and same-target time saved are better.
Coverage is the shared span used by the AUC metrics.

| env | run | HV delta | HV/base | budget lift AUC | budget coverage | time save AUC | relative time save | target coverage |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| fsrs6 | Cost-ADR coverage | 47,692 | +1.543% | -0.8 | 99/112, 98.670% | 1.81 | -0.198% | 103/112, 97.287% |
| fsrs6 | Cost-ADR quality v2 | 36,004 | +1.165% | -165.2 | 100/112, 99.423% | 1.56 | -11.933% | 100/112, 96.570% |
| fsrs6 | Cost-ADR hybrid | 57,918 | +1.874% | 4.8 | 103/112, 99.432% | 2.30 | 0.140% | 100/112, 95.397% |
| fsrs6 | ADR pop16 | 96,070 | +3.108% | 85.7 | 86/112, 83.234% | 4.56 | 11.371% | 79/112, 74.096% |
| fsrs6 | Oracle distill w11 | 105,660 | +3.418% | 113.7 | 95/112, 80.364% | 4.74 | 13.588% | 86/112, 77.913% |
| lstm | Cost-ADR coverage | -3,922 | -0.122% | -58.2 | 109/114, 99.001% | -0.15 | -11.434% | 100/114, 93.435% |
| lstm | Cost-ADR quality v2 | -12,194 | -0.379% | -83.4 | 108/114, 99.711% | -1.89 | -40.529% | 103/114, 96.173% |
| lstm | Cost-ADR hybrid | 12,753 | +0.396% | -45.0 | 109/114, 99.757% | 0.42 | -10.978% | 105/114, 94.773% |
| lstm | ADR pop16 | 52,881 | +1.642% | 50.2 | 81/114, 80.385% | 2.16 | 5.359% | 85/114, 83.872% |
| lstm | Oracle distill | 42,521 | +1.321% | 67.2 | 96/114, 84.598% | 0.62 | 5.575% | 96/114, 89.497% |

## Decision

The hybrid selector is a real Cost-ADR improvement over the prior coverage-aware
run:

- FSRS6 HV delta: `47,692 -> 57,918`
- FSRS6 relative time-save AUC: `-0.198% -> 0.140%`
- LSTM HV delta: `-3,922 -> 12,753`
- LSTM time-save AUC: `-0.15 -> 0.42`

It is not strong enough to replace ADR or oracle distill. ADR still has much
higher FSRS6 and LSTM HV and clearly better time-save AUC. The next Cost-ADR
iteration should make this selector first-class instead of manually assembled:
train multiple objective variants, choose per user by internal training HV, and
then address user 7/8 frontier collapse with either a user-specific fallback or
a quality target tied directly to same-target time saved.

## Artifact Paths

- Quality v1 config: `experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1.toml`
- Quality v2 config: `experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml`
- Hybrid config: `experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml`
- Hybrid run root: `artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off`
