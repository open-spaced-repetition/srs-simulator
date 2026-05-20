# Stationary Finite Distill Train-Weight 1 and 4 Control Plan

## Question

Does adding low cost weights `1` and `4` to stationary finite distill training
repair the user-2 high-memory frontier regression, without changing network
capacity or the rest of the recipe?

## Context

The repaired stationary finite teacher exposes a sharp low-weight, high-memory
frontier bend for user 2. The current 476-parameter per-user student is trained
on sparse weights:

```text
0,16,64,256,1024
```

That means `w=1`, `w=2`, and `w=4` behavior is mostly interpolation, even though
the formal evaluation includes `1,2,4` and the user-2 failure is concentrated
near `w=0..2`. A dense low-weight eval partially recovers user 2, but before
changing architecture or training on a full dense grid, this plan tests the
smallest plausible supervision fix: add `1` and `4`.

## Hypothesis

Adding `1` and `4` should improve the shape of the high-memory frontier because
the student will see direct teacher labels on both sides of the current
low-weight interpolation gap. It should be much cheaper than a full
dense-low-weight training grid and should isolate whether the regression is
primarily missing low-weight supervision rather than model capacity.

## Treatments

Run these two rows first:

| label | training weights | purpose |
| --- | --- | --- |
| sparse_baseline | `0,16,64,256,1024` | current repaired-teacher baseline |
| add_1_4 | `0,1,4,16,64,256,1024` | minimal low-weight supervision treatment |

Optional diagnostic rows, only if `add_1_4` improves but does not pass gates:

| label | training weights | purpose |
| --- | --- | --- |
| add_1_only | `0,1,16,64,256,1024` | test whether `w=1` is the critical point |
| add_4_only | `0,4,16,64,256,1024` | test whether `w=4` stabilizes mid-low weights |
| add_1_2_4 | `0,1,2,4,16,64,256,1024` | test whether `w=2` is still missing |

Do not include dense-low-weight training in this plan's primary comparison. That
belongs in a follow-up if the minimal treatment helps.

## Fixed Variables

- Environment: `fsrs6`
- Users: `1,2,3,4,5,6,7,8`
- Review Markov transition: `false`
- Teacher: repaired stationary finite, four-corner transition interpolation and
  bilinear value lookup
- Teacher grid: `64x32`
- Action retentions: clipped 11-action grid
- Student architecture: current `residual:8:2`, 476 parameters per user
- Cost-weight feature: fixed current normalized `log1p(cost_weight)`
- Supervision: `uniform_table`
- Epochs: `128`
- Steps per epoch: `64`
- Table samples per weight: `256`
- Eval particles: `10000`
- Seed: `42`
- Torch device: CUDA

## Evaluation Matrix

For each treatment:

- Evaluate against `fsrs6` on the formal sparse eval grid:
  `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Evaluate user 2 against `fsrs6` on dense-low-weight eval grid:
  `0,0.0625,0.125,0.25,0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,320,384,512,1024`
- Evaluate direct pairwise against repaired exact stationary finite.
- Evaluate direct pairwise against the Markov-off ADR portfolio.
- Report the high-memory segment contribution for user 2 over `9400-9750`
  memorized cards.

## Required Artifacts

Use new roots; do not overwrite current first-eight artifacts.

Suggested training roots:

- `artifacts/single_card_tradeoff/stationary_finite_distill_train_weights_sparse_first8_markov_off/`
- `artifacts/single_card_tradeoff/stationary_finite_distill_train_weights_add_1_4_first8_markov_off/`

Each root must include:

- `user_<id>_policy.pt`
- `train_summary.csv`
- `summary.csv`
- `regret_auc.csv`
- `performance_summary.json`
- `gpu_monitor/summary.json`
- A command or config snapshot containing the exact training weights

Suggested comparison root:

- `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/`

Comparison outputs should include:

- `train_weight_summary.csv`
- `train_weight_by_user.csv`
- `train_weight_direct_vs_exact.csv`
- `train_weight_user2_dense_loww.csv`
- `train_weight_user2_segment_auc.csv`
- Frontier plots for user 2 comparing `fsrs6`, exact, sparse baseline,
  `add_1_4`, and ADR

## Gates

The `add_1_4` treatment is useful if it meets all of these versus the sparse
baseline:

- User 2 formal sparse-grid AUC versus `fsrs6` is no longer negative, or improves
  by at least `+3.0` deck-minutes/day.
- User 2 dense-low-weight AUC versus `fsrs6` improves and is at least `+1.0`
  deck-minutes/day.
- Mean direct AUC versus repaired exact stationary finite improves.
- Mean coverage versus `fsrs6` drops by less than `1.0` percentage point.
- Minimum user coverage remains at least `90%`.
- No other user loses more than `2.0` relative-time-saved percentage points.
- GPU monitor reports no shared-memory spill.

If `add_1_4` passes user 2 but worsens other users, run the optional rows to
identify whether `1`, `4`, or the combination causes the regression.

## Suggested Commands

The exact commands should be committed as TOML configs before formal runs. The
treatment command should follow this shape:

```bash
uv run experiments/single_card_tradeoff/oracle_stationary_finite_distill_multiuser.py \
  --env fsrs6 \
  --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --per-user-models \
  --per-user-supervision uniform_table \
  --cost-weights 0,1,4,16,64,256,1024 \
  --out-dir artifacts/single_card_tradeoff/stationary_finite_distill_train_weights_add_1_4_first8_markov_off \
  --torch-device cuda \
  --no-progress
```

If the CLI does not currently persist training cost weights clearly enough in
metadata, add that metadata before running the formal comparison.

## Interpretation Rules

- If `add_1_4` fixes user 2, prioritize training-weight density before network
  capacity.
- If `add_1_4` helps but remains below exact teacher, proceed to either
  `add_1_2_4` or the capacity sweep.
- If `add_1_4` does not help, the failure is less likely to be a missing-label
  issue at `w=1/4`; prioritize capacity, loss weighting, or frontier-aware
  objectives.
- Always report `same_target_time_saved_auc`, relative AUC, and coverage
  together.
