# Cost-Weight Conditioned ADR-Style Single Policy

## 1. Problem

The current single-card ADR setup usually trains or evaluates a separate policy
point for each scalarization weight:

```text
maximize M(policy) - lambda * T(policy)
```

where:

```text
M(policy) = card_expected_retrievability
T(policy) = card_minutes_per_day
lambda = cost_weight
```

This produces a useful frontier, but the deployed representation is a portfolio:

```text
(user, lambda) -> policy_lambda(S, D) -> desired retention
```

The goal of this document is to specify an ADR-style single policy:

```text
policy_user(S, D, cost_weight) -> desired retention or interval
```

The policy should keep the low-parameter and interpretable flavor of ADR while
covering a full cost-weight range with one artifact per user. This is a
family-constrained policy class, in the terminology of
`docs/single_card_tradeoff/constrained_target_policy_search.md`: it can be
searched, distilled, and evaluated against a frontier, but it does not receive
oracle optimality certificates unless it is compared to an exact oracle search.

## 2. Relationship to Constrained Target Search

The constrained target search document defines a general pipeline:

```text
theta -> policy(theta) -> evaluate(policy) -> (M, T)
```

For the cost-weight-conditioned single policy, the trainable object is:

```text
theta_user -> policy_user(S, D, lambda)
```

The cost weight is no longer an external policy coordinate that selects one
artifact. It becomes an input to the policy. The target-search layer can still
use the policy in two ways:

- scalarized sweep: evaluate `policy_user(..., lambda)` over many lambda values;
- constrained query: build the induced `(M, T)` frontier and select the best
  deterministic feasible lambda for a target memory or time budget.

This changes the artifact shape but not the evaluation contract. A single policy
still induces many executable policy points:

```text
lambda_grid -> policy_user_lambda -> (M_lambda, T_lambda)
```

The resulting frontier is only optimal within this conditional policy family.
Exact stationary finite or continuous stationary finite oracles remain the
benchmark for certificate-based target-local refinement.

## 3. Empirical Policy Geometry

The strongest current evidence comes from the first-eight-user continuous
stationary finite S/D policy analysis. It averages policy-table cells uniformly
over users, stability grid cells, and difficulty grid cells.

Representative rows:

| cost weight | mean DR | min-DR share | max-DR share | lowest-S mean | highest-S mean | D=1 mean | D=10 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.934 | 0.0% | 36.5% | 0.667 | 0.980 | 0.933 | 0.935 |
| 16 | 0.892 | 0.4% | 11.9% | 0.667 | 0.860 | 0.896 | 0.867 |
| 64 | 0.840 | 5.0% | 5.0% | 0.667 | 0.620 | 0.854 | 0.784 |
| 256 | 0.728 | 23.2% | 2.4% | 0.666 | 0.500 | 0.769 | 0.638 |
| 1024 | 0.643 | 46.2% | 1.0% | 0.663 | 0.500 | 0.663 | 0.587 |

Observed structure:

1. Increasing cost weight lowers retention monotonically in aggregate.
2. The effect is not a uniform downward shift. It changes where the table
   saturates.
3. Low stability is almost cost-invariant because the shortest physical review
   interval is one day. The corresponding canonical retention is around `0.66`,
   above the configured lower retention bound.
4. High stability carries most of the tradeoff. As cost rises, high-stability
   states move toward low retention first because they can tolerate long
   intervals.
5. Difficulty matters mostly when cost is nontrivial. Higher difficulty lowers
   the value of a successful review, so high-difficulty states accept lower
   retention at the same stability.
6. The surface transitions from high-retention saturation at low cost to
   low-retention saturation at high cost.

The practical implication is that the policy needs state-dependent cost
sensitivity. A scalar offset such as:

```text
logit(DR) = base(S, D) - alpha * cost_weight
```

is too rigid. The slope with respect to cost must depend on `S` and `D`, and it
should allow saturation near both action bounds.

## 4. Policy Interface

The proposed deployed interface is:

```text
CostConditionedADRPolicy:
  input:
    stability S
    difficulty D
    cost_weight lambda
  output:
    interval_days or desired_retention
```

There are two viable action heads.

### 4.1 Recommended: Interval-Internal Head

The policy predicts a physical review interval in log space:

```text
log_interval = f_theta(S, D, lambda)
interval = round(clamp(exp(log_interval), 1, max_interval_days))
desired_retention = forgetting_curve(interval, S)
```

This is preferred for the single-card setting because the observed low-S edge is
caused by interval geometry. Predicting interval first lets the 1-day floor
appear naturally. The policy can still export desired retention for schedulers
that consume a retention target.

### 4.2 Compatible: Desired-Retention Head

The policy predicts a bounded desired retention directly:

```text
desired_retention =
  retention_min + (retention_max - retention_min) * sigmoid(f_theta(S, D, lambda))
```

This is closer to existing ADR policy artifacts. It is easier to plug into
`fsrs6_adr`, but it must learn interval-rounding geometry indirectly.

## 5. Feature Normalization

Use normalized state features consistent with existing ADR code:

```text
x = normalize(log(S)) in [0, 1]
d = normalize(D)      in [0, 1]
z = log1p(lambda) / log1p(lambda_max)
```

`z` should be the primary cost input. The tested lambda range spans several
orders of magnitude, and linear raw cost would overemphasize the high-cost end.

Recommended state basis:

```text
phi(S, D) = [
  1,
  x,
  d,
  x * d,
  x^2,
  d^2,
  relu(x - 0.5),
  relu(d - 0.5),
]
```

Recommended cost basis:

```text
h(lambda) = [
  sqrt(z),
  z,
  z^2,
]
```

The square-root term gives the policy resolution at low and medium cost weights.
The quadratic term gives it enough curvature to express high-cost saturation.

## 6. Parametric Families

### 6.1 Monotone Interval Family

The first implementation should use a monotone interval model:

```text
base(phi) = a0 . phi
slope_1(phi) = softplus(a1 . phi)
slope_2(phi) = softplus(a2 . phi)
slope_3(phi) = softplus(a3 . phi)

log_interval =
  base(phi)
  + slope_1(phi) * sqrt(z)
  + slope_2(phi) * z
  + slope_3(phi) * z^2
```

Increasing `lambda` can only increase the predicted interval. This matches the
intended scalarization semantics: higher review cost should not make a card more
aggressively reviewed at the same `(S, D)` state.

With the eight-feature `phi`, this family has:

```text
8 features * 4 coefficient groups = 32 parameters per user
```

A smaller six-feature version without hinge terms has 24 parameters per user.

### 6.2 Monotone Desired-Retention Family

The DR-head version mirrors the interval family with a sign flip:

```text
base(phi) = a0 . phi
slope_1(phi) = softplus(a1 . phi)
slope_2(phi) = softplus(a2 . phi)
slope_3(phi) = softplus(a3 . phi)

logit(DR) =
  base(phi)
  - slope_1(phi) * sqrt(z)
  - slope_2(phi) * z
  - slope_3(phi) * z^2

DR = retention_min + (retention_max - retention_min) * sigmoid(logit(DR))
```

This family is convenient when the scheduler must consume a desired retention.
It is less direct than the interval family but still captures state-dependent
cost sensitivity.

### 6.3 Non-Monotone Residual Extension

If the monotone family underfits, add a small residual with a bounded amplitude:

```text
log_interval =
  monotone_log_interval
  + residual_scale * tanh(r . [phi, z, z^2, x*z, d*z])
```

The residual should be disabled in the first pass. It weakens monotonicity and
makes target-search behavior harder to reason about. Use it only if exact-value
evaluation shows a clear systematic gap that cannot be fixed by the monotone
state basis.

## 7. Training Strategy

Direct black-box rollout search should not be the first training method. The
better path is teacher-first, then rollout fine-tuning.

### 7.1 Teacher Distillation

Use an exact stationary teacher:

```text
teacher(S, D, lambda) -> interval or desired_retention
```

Best initial teacher choices:

- `fsrs6_oracle_continuous_stationary_finite` for interval-internal training;
- `fsrs6_oracle_stationary_finite` for discrete retention-action training.

For each user, sample table cells from:

```text
lambda in train_cost_weights
S grid cells
D grid cells
```

Recommended loss for interval head:

```text
L = weighted_smooth_l1(
      predicted_log_interval,
      teacher_log_interval
    )
```

Recommended loss for DR head:

```text
L =
  interval_weight * smooth_l1(predicted_log_interval, teacher_log_interval)
  + retention_weight * smooth_l1(predicted_retention_logit, teacher_retention_logit)
```

The interval term should dominate because rollout behavior is determined by the
rounded physical interval.

### 7.2 Sampling Distribution

Use two sampling modes:

1. Uniform table sampling.
   This preserves full policy-surface coverage and prevents the small model from
   forgetting rare state regions.

2. Teacher-occupancy sampling.
   This emphasizes states that actually affect rollout metrics.

The first production run should use a mixed sampler:

```text
50% uniform table samples
50% teacher occupancy samples
```

If coverage degrades, increase the uniform share. If same-target time saved is
weak while coverage is high, increase the occupancy share.

### 7.3 Rollout Fine-Tuning

After distillation, run a small CEM/CMA-ES fine-tune initialized at the
distilled coefficients. The objective is the scalarized lifecycle objective:

```text
mean over lambda_grid of [M(policy_lambda) - lambda * T(policy_lambda)]
```

Use a coverage-preserving guardrail:

```text
penalize if frontier coverage vs fsrs6 falls below threshold
```

Fine-tuning should use low particles for exploration and high particles only for
final confirmation, following the discovery/confirmation split in constrained
target search.

## 8. Evaluation Plan

Evaluate every candidate as an ordinary policy family:

```text
lambda_grid -> policy_user(S, D, lambda) -> rollout points -> frontier
```

Required comparisons:

1. Against `fsrs6` desired-retention frontier.
2. Direct shared-span comparison against native single-card ADR.
3. Gap to exact stationary finite or continuous stationary finite teacher.
4. Coverage over the `fsrs6` memory span.
5. Parameter count per user and per evaluated frontier.

Primary metrics:

```text
same_target_time_saved_auc
relative_same_target_time_saved_auc_percent
span_coverage_percent
direct pairwise AUC vs fsrs6_adr
```

Suggested pass gates for the first strong candidate:

```text
mean relative time saved vs fsrs6 > native ADR
mean direct AUC vs native ADR > 0
positive direct AUC vs native ADR on at least 6/8 first users
mean span coverage >= 95%
min span coverage >= 85%
```

Stretch target:

```text
approach or exceed 476-param stationary finite distill
while using <= 64 parameters per user
```

## 9. Artifact Shape

The policy artifact should be one JSON file per user:

```json
{
  "policy_kind": "fsrs6-cost-conditioned-adr",
  "feature_version": "fsrs6_cost_adr_interval_mono_v1",
  "action_head": "interval",
  "coefficients": [...],
  "cost_weight_min": 0.0,
  "cost_weight_max": 1024.0,
  "retention_min": 0.5,
  "retention_max": 0.98,
  "bounds": {
    "s_min": 0.01,
    "s_max": 36500.0,
    "d_min": 1.0,
    "d_max": 10.0
  },
  "metadata": {
    "user_id": 1,
    "teacher": "fsrs6_oracle_continuous_stationary_finite",
    "train_cost_weights": [0, 4, 16, 64, 256, 1024],
    "eval_cost_weights": [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512, 1024]
  }
}
```

The scheduler should accept a runtime `goal_cost_weight` and call:

```text
policy.evaluate(stability=S, difficulty=D, cost_weight=goal_cost_weight)
```

If the artifact uses an interval head but the scheduler interface expects
desired retention, convert the selected interval to the canonical retention
before returning.

## 10. Target-Search Integration

This policy can be exposed to `target_search` as a conditional one-dimensional
family:

```text
theta_name = "goal_cost_weight"
theta_value = lambda
policy_ref = user policy JSON
```

For constrained memory targets:

```text
evaluate lambda grid
build deterministic frontier
select lowest T with M >= M0
confirm selected neighbors with high particles
```

For time targets:

```text
evaluate lambda grid
build deterministic frontier
select highest M with T <= T0
confirm selected neighbors with high particles
```

This gives a clean bridge from scalarized training to user-facing target
queries. The target-search output should still say that the selected point is
family-constrained, not oracle-certified.

## 11. Nonstationary Extension

The stationary policy intentionally omits remaining time:

```text
policy(S, D, lambda)
```

This is enough to target the stationary finite frontier and to beat ADR if the
family is expressive enough. It is not enough to match finite-horizon interval
or continuous-retention oracles, whose advantage partly comes from deadline
behavior.

If the goal changes from "beat ADR with a compact single policy" to "approach
finite-horizon oracle", add remaining time:

```text
policy(S, D, lambda, remaining_time_norm)
```

Use the existing ADR time-feature pattern as the conservative starting point:

```text
[phi(S, D), t, x*t, d*t, t^2]
```

This extension should be evaluated separately because it changes the policy
class and the deployment assumptions.

## 12. Recommended Rollout Plan

Phase 1: offline fit and exact-value diagnostic.

- Implement the monotone interval family.
- Fit one policy per user from continuous stationary finite teacher tables.
- Evaluate teacher-table loss and exact-value gap.
- Compare 24-param and 32-param variants.

Phase 2: single-card rollout comparison.

- Add the policy as a scheduler family in the single-card tradeoff runner.
- Evaluate on the first eight users and the 19-weight grid.
- Compare against `fsrs6`, native ADR, 476-param distill, and exact stationary
  finite teacher.

Phase 3: rollout fine-tune.

- Initialize from the distilled coefficients.
- Run short CEM/CMA-ES fine-tuning with batched users and cost weights.
- Confirm final frontiers with high particles.

Phase 4: target-search integration.

- Expose `goal_cost_weight` as the family coordinate.
- Reuse `points.csv`/`frontier.csv`/`target_answers.csv` output format.
- Validate memory and time target queries against exact oracle target search.

## 13. Success Criteria

The first useful version should satisfy:

- one policy artifact per user covers the full cost-weight grid;
- policy evaluation is deterministic and cheap;
- cost weight is monotone in interval or anti-monotone in desired retention;
- mean relative time saved exceeds native single-card ADR on the first-eight
  benchmark;
- direct shared-span AUC against ADR is positive on most users;
- mean span coverage stays at or above `95%`;
- final reports distinguish family-constrained performance from oracle
  certificate claims.

The core design principle is:

```text
learn the state-dependent cost sensitivity, not just a cost-dependent offset.
```

The empirical S/D surfaces show that cost weight changes where the policy
saturates. A compact single policy must therefore let the cost slope depend on
stability and difficulty.
