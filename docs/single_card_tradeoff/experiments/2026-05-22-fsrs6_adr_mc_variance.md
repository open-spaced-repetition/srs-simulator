# FSRS6 ADR Monte Carlo Variance, First Eight Users

## Question

The native single-card ADR training uses only `64` simulated cards per
candidate. How large is the Monte Carlo noise in that training objective?

## Method

This experiment fixes the trained ADR policies from:

`artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off`

It does not retrain. The artifact was generated before the later 512- and
1024-particle ADR reruns overwrote that train-run root, so rerunning the command
below now will measure the current 1024-particle policy set unless the earlier
run root is restored. The reported variance numbers are still the archived
64-particle-estimator measurements used to decide that larger training
particle counts were warranted. For each of the `8 * 19 = 152` trained policies,
the script repeats the training-style rollout with different random seeds:

- Environment: `fsrs6`
- Users: `1,2,3,4,5,6,7,8`
- Cost weights:
  `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Repeated estimator: `128` repeats at `64` particles
- Reference estimator: one `10000`-particle rollout
- Objective:
  `card_expected_retrievability - cost_weight * card_minutes_per_day`
- Exact memory accumulation: off, matching default ADR training rollouts
- Review Markov transitions: off
- Device: CUDA

Command:

```bash
uv run python -m experiments.single_card_tradeoff.cli.fsrs6_adr_mc_variance \
  --env fsrs6 \
  --user-ids 1,2,3,4,5,6,7,8 \
  --cost-weights 0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024 \
  --train-run-root artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off \
  --out-dir artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --benchmark-partition 0 \
  --days 1825 \
  --particles 64 \
  --repeats 128 \
  --reference-particles 10000 \
  --job-batch-size 8192 \
  --torch-device cuda \
  --no-progress
```

## Results

Across the 152 policies, the standard deviation of a single 64-particle
objective estimate is:

| statistic | objective variance | objective std |
| --- | ---: | ---: |
| mean | 0.002063 | 0.02135 |
| median | 0.0000768 | 0.00876 |
| p90 | 0.002123 | 0.05262 |
| p95 | 0.006425 | 0.08164 |
| max | 0.116364 | 0.34112 |

The standard error of the 128-repeat mean is much smaller:

| statistic | objective SE of mean |
| --- | ---: |
| mean | 0.00189 |
| p90 | 0.00465 |

The 128-repeat mean is close to the 10000-particle reference:

| statistic | absolute mean-reference objective error |
| --- | ---: |
| mean | 0.00184 |
| p90 | 0.00518 |
| max | 0.01926 |

Noise grows strongly with the cost weight:

| cost weight | mean objective std | p90 objective std | max objective std |
| ---: | ---: | ---: | ---: |
| 0 | 0.00015 | 0.00034 | 0.00072 |
| 1 | 0.00118 | 0.00293 | 0.00309 |
| 8 | 0.00422 | 0.00726 | 0.00977 |
| 32 | 0.00911 | 0.01565 | 0.02090 |
| 128 | 0.02084 | 0.03427 | 0.05335 |
| 256 | 0.03319 | 0.05632 | 0.08345 |
| 512 | 0.05564 | 0.10322 | 0.15704 |
| 1024 | 0.11197 | 0.23888 | 0.34112 |

Noise also varies by user:

| user | mean objective std | p90 objective std | max objective std |
| ---: | ---: | ---: | ---: |
| 1 | 0.02983 | 0.06880 | 0.19506 |
| 2 | 0.01058 | 0.01581 | 0.03388 |
| 3 | 0.00872 | 0.01788 | 0.02199 |
| 4 | 0.05760 | 0.13295 | 0.34112 |
| 5 | 0.02947 | 0.07170 | 0.14590 |
| 6 | 0.01877 | 0.04152 | 0.08597 |
| 7 | 0.00822 | 0.02508 | 0.03468 |
| 8 | 0.00758 | 0.02002 | 0.04468 |

The noisiest single policies are high-cost policies:

| user | cost weight | objective std | SE of 128-repeat mean | abs mean-reference error |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1024 | 0.34112 | 0.03015 | 0.01926 |
| 1 | 1024 | 0.19506 | 0.01724 | 0.01018 |
| 4 | 512 | 0.15704 | 0.01388 | 0.00367 |
| 5 | 1024 | 0.14590 | 0.01290 | 0.01075 |
| 4 | 384 | 0.12693 | 0.01122 | 0.01121 |

Compared with the final training improvement recorded in the ADR training
summary:

- Median `objective_std / abs(objective_improvement)`: `0.071`
- p90 `objective_std / abs(objective_improvement)`: `0.908`
- Policies where one-shot 64-particle std exceeds the recorded improvement:
  `13/152`

Those 13 policies are mostly high-cost user 7 and user 8 policies, plus user 6
at `w=32`. Three of them are the same policies that failed the training
objective gate in the ADR training summary: user 7 at `w=192` and `w=320`, and
user 8 at `w=256`.

## Interpretation

The 64-card estimator is not uniformly too noisy, but it is noisy in the part of
the frontier where ADR was already weakest: high cost weights and users with
very low time budgets. For low and medium cost weights, objective std is small
relative to the training improvements. At high weights, especially `w >= 256`,
the noise is large enough that CEM candidate ranking can be materially noisy
when candidates are close.

A single comparison between two policies with independent 64-card estimates has
noise standard deviation roughly `sqrt(2)` times the per-policy objective std.
At the p90 policy, that is about `0.074`; at `w=1024`, it can be much larger.
This is large enough to explain unstable or weak ADR outcomes for some
high-cost points.

The 128-repeat means line up well with the 10000-particle reference, so the
issue is variance rather than an obvious estimator bias.

## Runtime And GPU Monitor

- Runtime: `816.51s`
- Peak `nvidia-smi` memory: `2714 MiB`
- Peak summed shared GPU memory: `205,414,400` bytes
- Shared-memory spill detected: `false`

## Artifacts

- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/aggregate_summary.json`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/policy_variance_summary.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/variance_by_lambda.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/variance_by_user.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/reference.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/samples.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_mc_variance/performance_summary.json`
