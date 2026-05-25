# Single-Card Tradeoff Constrained Target Policy Search

## 1. Problem

The single-card tradeoff benchmark evaluates a scheduling policy by two
rollout-level expectations:

```text
M(policy) = card_expected_retrievability
T(policy) = card_minutes_per_day
```

For the current 1825-day single-card lifecycle, `M` is the average expected
retrievability accumulated over days and particles, and `T` is the average
single-card review time in minutes per day. Both are induced by the policy. A
more aggressive policy usually increases both `M` and `T`, but the relationship
is not guaranteed to be strictly monotone because actions are rounded to review
intervals, rollouts are stochastic, and policy classes may be discrete or
capacity-limited.

Current oracle, ADR, and distillation comparisons usually optimize or evaluate a
scalarized objective:

```text
maximize M(policy) - lambda * T(policy)
```

The constrained target problem asks for the inverse form:

```text
given M0: minimize T(policy) subject to M(policy) >= M0
given T0: maximize M(policy) subject to T(policy) <= T0
```

The engineering goal is to support a batch of targets, not a single target:

```text
M0 values: 0.70, 0.75, 0.80, 0.85, 0.90, ...
T0 values: 0.001, 0.002, 0.004, ...
```

and return a corresponding set of best feasible deterministic policies, with
enough diagnostics to explain feasibility, target slack, and distance from an
oracle frontier when an oracle is available.

## 2. Application Background

In the single-card setting, each policy point is one scheduler behavior over the
same iid lifecycle:

- `fsrs6` static desired-retention policies use a target retention scalar.
- `fixed` policies use a fixed interval scalar.
- exact finite or stationary oracles use a cost weight `lambda` or
  `goal_cost_weight`.
- continuous desired-retention oracle variants output a retention but still
  execute through rounded day-level intervals.
- ADR and low-parameter direct search policies are parameterized functions over
  FSRS state, often trained per cost weight.
- distilled policies condition on a goal variable and approximate an exact
  teacher policy.

All of these policies can be viewed through a common interface:

```text
theta -> policy(theta) -> evaluate(policy) -> (M, T)
```

The interpretation of `theta` depends on the policy family:

| family | theta | theoretical role |
| --- | --- | --- |
| exact scalarized oracle | `lambda` | dual tradeoff variable |
| `fsrs6` baseline | desired retention | policy-family coordinate |
| fixed interval | interval days | policy-family coordinate |
| stationary finite distill | goal cost weight | student conditioning coordinate |
| ADR portfolio | training lambda or policy index | trained policy coordinate |

The same constrained target API can apply to all of them, but the strength of
the optimality claim differs. Exact oracles can provide certificates for the
supported frontier. Ordinary parameterized policy families can only provide
best feasible policies within the sampled or searched family.

## 3. Theoretical Foundation

### 3.1 Finite-Horizon CMDP View

The single-card process is a finite-horizon Markov decision process with known
transition dynamics. A state contains enough information to predict the next
review outcome distribution and future memory:

```text
state ~= (remaining day, stability S, difficulty D, pending cost, ...)
```

An action chooses a review interval directly or chooses a desired retention that
is converted into a rounded interval. The rollout metrics are expected sums:

```text
M(policy) = E[sum memory reward] / days
T(policy) = E[sum time cost] / days
```

The target-memory constrained problem is a constrained MDP:

```text
minimize T(policy)
subject to M(policy) >= M0
```

Using occupancy measures `x(s,a,t)`, both objectives are linear:

```text
M(x) = <m, x>
T(x) = <c, x>
```

The exact constrained problem can be written as a linear program:

```text
minimize <c, x>
subject to <m, x> >= M0
           x satisfies MDP flow constraints
           x >= 0
```

This LP is the cleanest theoretical formulation, but it is not the cheapest
engineering implementation for the current FSRS grid because the occupancy
space spans time, stability, difficulty, user parameters, and actions.

### 3.2 Scalarization and Supported Frontier

The current oracle solves the Lagrangian/scalarized problem:

```text
maximize M(policy) - lambda * T(policy)
```

Each `lambda` gives one supported point on the `(M,T)` frontier:

```text
policy_lambda -> (M_lambda, T_lambda)
```

If all supported frontier segments around a target are known, the constrained
query can be answered from those points:

- target `M0`: choose the feasible point with smallest `T`.
- target `T0`: choose the feasible point with largest `M`.

If randomized or mixed policies are allowed, the convex hull of frontier points
is also feasible because `M` and `T` are expectations. If only deterministic
policies are allowed, the solver should return the best deterministic feasible
point and report the target slack. This document focuses on deterministic output
as the default, while keeping mixed output as an optional diagnostic.

### 3.3 Why Sparse Lambda Points Are Not Enough

A sparse lambda set such as:

```text
lambda in {0, 1024}
```

may produce two extreme policies:

```text
(M=0.99, T=high)
(M=0.54, T=low)
```

For a target `M0=0.75`, mixing these endpoints is only optimal among the two
known policies. It is not a global claim. There may be a middle lambda that
produces:

```text
(M=0.76, T << endpoint mixture T)
```

Therefore the engineering problem is not just "interpolate known endpoints".
It is to find or certify the local frontier near each target.

### 3.4 Lambda_AB Oracle Certificate

For exact scalarized oracles, two candidate frontier points can be certified
locally. Let:

```text
A = (M_A, T_A)
B = (M_B, T_B)
M_A > M_B and T_A > T_B
```

The slope of the segment is:

```text
lambda_AB = (M_A - M_B) / (T_A - T_B)
```

Solve the scalarized oracle at `lambda_AB`:

```text
V*(lambda_AB) = max_policy M(policy) - lambda_AB * T(policy)
```

The segment value is:

```text
V_AB = M_A - lambda_AB * T_A
     = M_B - lambda_AB * T_B
```

Then:

```text
gap = V*(lambda_AB) - V_AB
```

- If `gap <= tolerance`, no policy lies above this segment in scalarized value.
  The segment is certified as a supported hull edge for the oracle
  approximation being solved.
- If `gap > tolerance`, the oracle has found a better point. Insert that point
  and split the segment.

This certificate is a local proof step. Adaptive frontier refinement is the
algorithm that repeatedly applies the certificate to all target-relevant
segments.

### 3.5 Ordinary Policy Families Do Not Get Oracle Certificates

For a non-oracle policy family such as desired retention:

```text
theta = desired_retention
policy = fsrs6(theta)
```

we can search for:

```text
min_theta T(theta) subject to M(theta) >= M0
```

but this is only optimal within that policy family. There is no global
certificate unless we can solve:

```text
max_theta M(theta) - lambda * T(theta)
```

globally for the family and prove no better `theta` exists. In practice we use
batched adaptive sampling and final high-particle validation.

## 4. Design Principles

The implementation should follow these principles:

1. Batch targets together.
   A set of `M0/T0` targets should share one frontier search, not run one
   independent search per target.

2. Keep the output deterministic by default.
   Mixed policies are useful for theoretical hull diagnostics, but the default
   artifact should identify one executable policy per target.

3. Separate discovery from confirmation.
   Use cheaper evaluations to find candidates, then high-particle rollouts or
   exact evaluation to confirm feasibility and ranking.

4. Cache every expensive solve.
   Oracle DP/cache keys must include oracle kind, user, lambda, grid, action
   semantics, interpolation version, horizon, retention bounds, and relevant
   solver version.

5. Prefer target-aware local refinement.
   Do not build the whole frontier if the requested targets occupy only a small
   region.

6. Report infeasibility explicitly.
   A target above `max(M)` or below available minimum cost constraints should
   not be hidden by an arbitrary fallback policy.

## 5. Core Algorithms

### 5.1 Shared Point Set

All constrained solvers should maintain a shared evaluated point set:

```text
Point:
  user_id
  family
  theta
  policy_ref or cache_ref
  M
  T
  eval_particles or exact_eval_version
  source: oracle | rollout | cache | imported
```

From these points, compute the empirical deterministic frontier by removing
dominated points:

```text
point A dominates point B when:
  M_A >= M_B
  T_A <= T_B
  and at least one inequality is strict
```

For a target-memory query:

```text
best_det(M0) = argmin T among frontier points with M >= M0
```

For a target-time query:

```text
best_det(T0) = argmax M among frontier points with T <= T0
```

The same point set can answer many targets.

### 5.2 Oracle Target Search

For exact oracle families, use batched adaptive frontier refinement.

Initial points:

```text
lambda = 0
lambda = high enough to reach low-memory/high-time-cost end
plus any cached/evaluated lambda values already available
```

For each target, locate the current bracket:

```text
M-high point: M >= M0 with smallest known T near the target
M-low point:  M <  M0 with largest known M below the target
```

For each uncertified segment that matters for at least one target, compute:

```text
lambda_AB = (M_A - M_B) / (T_A - T_B)
```

Batch solve the top-K `lambda_AB` values. Insert new points if they improve the
frontier; otherwise mark the segment certified. Stop when every target's local
deterministic answer is bounded by certified neighboring segments or by the
configured tolerance.

Pseudocode:

```text
points = load_cached_points(user, family)
points += solve_initial_lambdas(batch)
segments = frontier_segments(points)

while budget remains:
    target_segments = select_segments_covering_targets(segments, targets)
    uncertified = [s for s in target_segments if not s.certified]
    if not uncertified:
        break

    lambdas = [s.lambda_AB for s in top_k(uncertified)]
    new_points = solve_oracle_batch(lambdas)

    changed = insert_frontier_improvements(points, new_points)
    if not changed for a segment:
        segment.certified = true

    segments = frontier_segments(points)

answers = query_best_feasible(points, targets)
```

Segment priority should be target-aware:

- segments containing one or more requested `M0/T0` targets;
- segments whose deterministic feasible endpoint could still be improved;
- segments with high certificate gap or large `(M,T)` span;
- segments covering multiple targets.

### 5.3 Ordinary One-Dimensional Policy Family Search

For a policy family parameterized by a scalar `theta`, use batched boundary
search. Examples:

```text
theta = desired_retention
theta = fixed_interval
theta = distill goal cost weight
theta = ADR policy lambda
```

When the family is approximately monotone, bracket each target and sample
inside merged brackets. For target memory:

```text
feasible if M(theta) >= M0
best feasible minimizes T(theta)
```

For target time:

```text
feasible if T(theta) <= T0
best feasible maximizes M(theta)
```

Pseudocode:

```text
points = initial_grid(theta)
evaluate_batch(points)

for round in 1..max_rounds:
    frontier = empirical_frontier(points)
    answers = query_best_feasible(frontier, targets)
    brackets = target_brackets(points, targets)
    brackets = merge_overlapping(brackets)

    if all targets converged:
        break

    candidates = []
    for bracket in selected(brackets):
        candidates += sample_quantiles(bracket, k_per_bracket)

    evaluate_batch(candidates)
    points += candidates

finalists = select_frontier_neighbors(points, targets)
confirm_batch(finalists, high_particles)
answers = query_best_feasible(confirmed_points, targets)
```

This does not prove global optimality outside the family. It provides a fast
family-constrained best feasible policy.

### 5.4 Direct Constrained Training

Direct training can be useful after an oracle or family-search benchmark exists.
For CEM/direct policy search, train one job per `(user, target)` and rank
candidate policies lexicographically:

For target memory:

```text
if M >= M0:
    rank by lower T
else:
    rank by smaller (M0 - M)
```

For target time:

```text
if T <= T0:
    rank by higher M
else:
    rank by smaller (T - T0)
```

This is preferable to a fixed penalty such as:

```text
score = -T - penalty * max(0, M0 - M)^2
```

because the penalty scale is fragile. Direct constrained training should still
be evaluated against an oracle or family-search benchmark and must use
high-particle final confirmation.

## 6. Engineering Implementation Plan

### 6.1 Library Layer

Add a small internal library under `experiments/single_card_tradeoff/core/` or a
sub-package such as `target_search/`:

```text
target_search/
  types.py
  frontier.py
  oracle_refinement.py
  family_search.py
  confirmation.py
```

Suggested data structures:

```text
ConstrainedTarget:
  target_type: "memory" | "time"
  value: float
  user_id: int | None

EvaluatedPoint:
  user_id: int
  family: str
  theta_name: str
  theta_value: float
  M: float
  T: float
  policy_ref: str | None
  cache_key: str | None
  exact: bool

TargetAnswer:
  target: ConstrainedTarget
  feasible: bool
  point: EvaluatedPoint | None
  achieved_M: float | None
  achieved_T: float | None
  memory_slack: float | None
  time_slack: float | None
  certified: bool
  neighbor_low: EvaluatedPoint | None
  neighbor_high: EvaluatedPoint | None
  mixed_T: float | None
  mixed_probability_high: float | None
```

The point and answer types should be independent of any specific scheduler. This
makes the target-search layer reusable for oracle, desired-retention baseline,
fixed interval, and learned policy families.

### 6.2 Evaluator Interface

Define a common evaluator protocol:

```text
evaluate(user_ids, theta_values, particles_or_exact) -> list[EvaluatedPoint]
```

For oracle families:

```text
solve_oracle_batch(user_ids, lambdas) -> policy tables + exact/evaluated (M,T)
```

For rollout-only families:

```text
rollout_policy_family_batch(user_ids, theta_values, particles) -> (M,T)
```

The batching dimension should be:

```text
user x theta x particles
```

with chunking controls to avoid GPU shared-memory spill.

### 6.3 CLI Shape

Add a target-search CLI rather than overloading the existing tradeoff runner:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 \
  --user-ids 1,2,3,4,5,6,7,8 \
  --family fsrs6_oracle_stationary_finite \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --deterministic-only \
  --particles 10000 \
  --out-dir artifacts/single_card_tradeoff/target_search/oracle_stationary_first8
```

For time targets:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --family fsrs6 \
  --target-times 0.001,0.002,0.004,0.008 \
  --theta-grid 0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98
```

Key arguments:

```text
--family
--target-memories
--target-times
--deterministic-only
--allow-mixed-diagnostics
--initial-theta-grid / --initial-lambdas
--max-refinement-rounds
--candidates-per-round
--certificate-tolerance
--explore-particles
--confirm-particles
--eval-group-batch-size
--torch-device
```

### 6.4 Output Files

The run should write:

```text
points.csv
frontier.csv
target_answers.csv
segments.csv
metadata.json
target_search.png
```

`target_answers.csv` should contain:

```text
user_id
target_type
target_value
family
feasible
theta_name
theta_value
achieved_M
achieved_T
memory_slack
time_slack
certified
policy_ref
cache_key
neighbor_low_theta
neighbor_high_theta
mixed_available
mixed_probability_high
mixed_M
mixed_T
```

For deterministic-only mode, `mixed_*` fields are diagnostic and should not be
treated as the selected policy.

### 6.5 Caching and Warm Starts

Oracle target search depends on repeated nearby lambda solves. Cache behavior is
critical:

- Every solved lambda should be written immediately after DP completion.
- Stationary finite policy iteration should initialize from the nearest cached
  lambda policy when possible.
- Cache keys must include oracle kind, user/config identity, horizon, grid
  shape, action semantics, interpolation version, retention bounds, and solver
  version.
- `points.csv` should also be reusable as an initialization source for later
  target searches.

For rollout-only families, cache evaluated `(user, theta, seed, particles,
days, family)` rows if the result is expensive and deterministic enough to
reuse.

### 6.6 Parallelization

Parallelization is where the largest engineering win comes from:

- batch all users when memory allows;
- batch multiple theta/lambda values per round;
- batch multiple targets through a shared frontier;
- chunk by lane count to avoid GPU memory spill;
- use existing GPU monitor artifacts to detect spill rather than relying only
  on `nvidia-smi`.

Suggested first defaults:

```text
oracle top-K segments per round: 4 to 8
ordinary family candidates per bracket: 4
exploration particles: 512 to 2048
confirmation particles: 10000
max refinement rounds: 8 to 16
```

For exact DP or stationary oracle solves, particle counts apply only to
rollout/evaluation if exact lifecycle objective is not directly available.

### 6.7 Noise and Feasibility Margins

Monte Carlo evaluation can flip a boundary result. The target-search layer
should support a safety margin:

```text
memory target: require M >= M0 + memory_margin
time target:   require T <= T0 - time_margin
```

The final report should distinguish:

```text
exploration_feasible
confirmed_feasible
```

If a high-particle confirmation run fails the target, the solver should either
fall back to the next safer feasible policy or report the target as not
confirmed.

### 6.8 Deterministic and Mixed Outputs

The default selected policy should be deterministic:

```text
target M0 -> best frontier point with M >= M0 and minimum T
target T0 -> best frontier point with T <= T0 and maximum M
```

When the local oracle segment is certified, the runner can also report mixed
diagnostics:

```text
p = (M0 - M_low) / (M_high - M_low)
T_mix = (1-p) * T_low + p * T_high
```

Mixed output is useful for theory and for deck-level randomized policies, but
it should not replace the deterministic answer unless the user explicitly asks
for randomized policies.

## 7. Discussion and Tradeoffs

### 7.1 Oracle Search vs Direct Constrained Training

Oracle target search is the right first benchmark because it uses known model
dynamics and provides local certificates. Direct training can be layered on top
afterwards:

```text
oracle target answer -> benchmark
constrained CEM/ADR target answer -> compact executable policy
distill target answer -> small conditional network
```

Without the oracle benchmark, direct constrained training may appear successful
while being far from the true best feasible time.

### 7.2 Deterministic Policy Overshoot

For deterministic output, the selected policy often overshoots the target:

```text
M_selected > M0
```

This is not a bug. It is the cost of requiring one deterministic policy instead
of allowing a mixture. The report should show:

```text
memory_slack = M_selected - M0
```

and, when available:

```text
mixed_T <= deterministic_T
```

so the user can see the deterministic price.

### 7.3 Target-Memory and Target-Time Symmetry

The same evaluated frontier answers both problem forms:

```text
given M0 -> minimize T
given T0 -> maximize M
```

The refinement priority differs slightly. Memory targets care most about
segments near horizontal memory level `M0`; time targets care most about
segments near vertical time budget `T0`. A shared target set can contain both.

### 7.4 Non-Monotonic Families

Do not assume strict monotonicity in production code. Even if a family is
conceptually monotone, simulation and action rounding can create small
violations. The solver should always select from all evaluated feasible points,
not simply return the last bisection point.

### 7.5 When to Build the Whole Frontier

Build the whole frontier when:

- many targets cover most of the memory/time span;
- the result is a reusable benchmark artifact;
- the oracle solve is already mostly cached;
- the report needs a full AUC/frontier comparison.

Use local target-aware refinement when:

- the user asks for a small number of targets;
- DP is expensive;
- the target region is narrow;
- the policy family is only being used as a quick constrained baseline.

## 8. Recommended Rollout Plan

Phase 1: family-constrained target search for existing rollout families.

- Support `fsrs6` desired retention and `fixed` intervals.
- Use batched evaluation and deterministic best feasible selection.
- Output `points.csv`, `frontier.csv`, and `target_answers.csv`.
- No oracle certificates yet.

Phase 2: oracle target search.

- Add batched lambda solve integration for stationary finite and continuous
  stationary finite oracles.
- Add `lambda_AB` segment certificates.
- Reuse existing DP cache aggressively.
- Support deterministic selected policies plus mixed diagnostics.

Phase 3: constrained direct training.

- Add CEM jobs keyed by `(user, target_type, target_value)`.
- Use lexicographic feasible-first ranking.
- Compare against Phase 2 oracle answers.
- Save policy artifacts with target metadata.

Phase 4: conditional target distillation.

- Train compact policies conditioned on `target_memory` or `target_time`.
- Calibrate final outputs with high-particle confirmation.
- Report target feasibility, slack, and gap to oracle target frontier.

## 9. Success Criteria

The first production-quality implementation should satisfy:

- one command accepts a batch of `M0` and/or `T0` targets;
- all targets share evaluated points and frontier refinement;
- outputs identify the selected deterministic policy and its slack;
- oracle families can certify target-local segments when supported by exact
  scalarized solves;
- rollout-only families clearly state that results are family-constrained and
  not globally certified;
- high-particle confirmation is separated from exploration;
- GPU runs include monitor artifacts and avoid shared-memory spill;
- all expensive oracle solves and policy evaluations are reusable through cache
  or saved point CSVs.

The most important engineering principle is:

```text
solve/evaluate candidate policies in batches, build one shared frontier, then
query all targets from that frontier.
```

This avoids independent per-target search loops and gives a clean path from
fast family-constrained baselines to certified oracle target policies.
