# RL Scheduler And Single-Card Tradeoff Research Process Report

Date: 2026-05-31

## Strategy Primer

This report names many scheduler and policy families. The first appearance of
each family below states its interface and how it is trained, solved, or
selected.

| strategy or family | input | output | training, solve, or selection method |
| --- | --- | --- | --- |
| FSRS6 desired-retention baseline | User-fitted FSRS6 memory state, scheduler weights, and a fixed desired-retention scalar. | A review interval chosen by the standard FSRS6 scheduler, producing one memory/time frontier point per scalar. | The scheduler itself is not learned in these experiments; the 16 baseline DR values per user are selected with CMA-ES in the FSRS6 environment and then staged as the formal comparison frontier. |
| Simulated annealing FSRS6 policy scheduler | FSRS6 card state and policy parameters from the early DR-conditioned scheduler search. | Desired-retention or interval decisions for the simulator. | Trained with simulated annealing on batched simulator feedback; later removed after CMA-ES and portfolio search became the stronger path. |
| FSRS6 ADR portfolio (`fsrs6_adr`) | User-fitted FSRS6 scheduler state features, mainly stability/difficulty-derived log-polynomial features; each portfolio child carries learned coefficients. | A dynamic desired retention clipped to the configured bounds, then converted by FSRS6 into a review interval. | Trained per user as a portfolio with SMS-EMOA, using candidate points `(memorized_average, -time_average)` and maximizing hypervolume gain over the staged FSRS6 DR baseline. |
| LSTM-trained ADR | The same ADR policy interface as `fsrs6_adr`. | The same dynamic desired-retention output as ADR. | Trained with the ADR SMS-EMOA portfolio workflow, but candidate simulation uses the LSTM evaluation environment instead of the FSRS6 training environment. |
| FSRS6 AP (`fsrs6_ap`) | User-fitted FSRS6 state plus a child policy containing one desired-retention scalar and bounded deltas to the FSRS6 scheduler weights. | A review interval from FSRS6 using the adjusted weights and desired-retention target. | Trained per user as an SMS-EMOA portfolio under the same pop16/off16/gen20 budget as ADR. |
| Default-weight ADR (`fsrs6_default_adr`) | ADR features computed with default FSRS6 scheduler weights rather than user-fitted scheduler weights. | A dynamic desired-retention value and FSRS6 interval. | Trained with the same SMS-EMOA portfolio budget as ordinary ADR to isolate whether user-fitted scheduler state scale matters. |
| ADR with remaining-time feature (`fsrs6_adr_time`) | Ordinary ADR state features plus normalized remaining simulation time. | A dynamic desired-retention value and FSRS6 interval. | Trained with the same SMS-EMOA portfolio budget as ordinary ADR to test horizon information in the multi-card scheduler. |
| Anki SM2 AP (`anki_sm2_ap`) | Anki SM2 scheduling state and learned adaptive-parameter values; it has no desired-retention input. | SM2-style interval/ease scheduling decisions. | Trained per user as an SMS-EMOA portfolio against the same staged FSRS6 baseline frontier. |
| Single-Card finite oracle (`fsrs6_oracle`) | Single-card FSRS6 state, remaining days, and scalar cost weight on study time. | The optimal discrete desired-retention action from a finite grid. | Solved by finite-horizon dynamic programming/grid backup, then evaluated by rollout; it is an oracle table rather than a learned compact policy. |
| Single-Card interval oracle (`fsrs6_oracle_interval`) | Single-card FSRS6 state, remaining days, and cost weight. | The optimal integer review interval. | Solved as an exact finite-horizon interval-action oracle over the same single-card lifecycle. |
| Single-Card continuous-retention oracle | Single-card FSRS6 state, remaining days, and cost weight. | A continuous desired-retention value, internally mapped through attainable rounded intervals. | Solved as a finite-horizon continuous-retention oracle; stationary variants drop the remaining-day input and solve a stationary table. |
| Stationary finite oracle | Single-card FSRS6 stability/difficulty state and cost weight, without remaining days. | A stationary retention-action table entry. | Solved from the finite grid oracle as a stationary approximation to long-horizon behavior. |
| Oracle distill policies | FSRS6 stability, difficulty, and cost weight. | An interval or desired-retention value from a compact neural approximation of the oracle table. | Trained by supervised distillation from exact or continuous oracle tables; examples include the 476-parameter stationary finite distill and continuous stationary distill. |
| Native single-card ADR | Single-card FSRS6 state for one user and one fixed cost weight. | A dynamic desired-retention value from a six-parameter log-polynomial rule. | Trained directly with CEM per `(user, cost weight)` on the single-card scalar objective. |
| UVFA PPO | Single-card state plus a goal or cost-weight condition. | A learned goal-conditioned retention-action policy and value function. | Trained with PPO using the objective `card_expected_retrievability - goal_cost_weight * card_minutes_per_day`, with oracle or static-policy guidance depending on the run. |
| Target-memory direct or target-conditioned policies | Single-card state plus a requested memory target, or one policy per target in the direct-search form. | A retention decision intended to meet the target with minimal time. | Trained with feasible-first constrained CEM and, for the target-conditioned variant, supervised distillation into one target-conditioned policy; certified oracle target search audits feasibility. |
| Oracle stationary finite distill portfolio in RL Scheduler | Multi-card FSRS6 scheduler state evaluated through per-user single-card distill policies and searched goal-cost weights. | Card-level interval or desired-retention recommendations, assembled into multi-card portfolio frontier points. | The underlying distill is trained on Single-Card teachers; RL Scheduler experiments search/select cost-weighted portfolio points and then evaluate them with the formal multi-card Pareto workflow. |
| Cost-ADR (`fsrs6_cost_adr`) | FSRS6 stability/difficulty-derived features plus an explicit cost weight, with later variants also changing action head, bounds, feature set, or horizon. | Either an interval head or a desired-retention head; the retention-head variants output dynamic DR directly. | Trained either by supervised fitting from Single-Card distill teachers for initialization/prototypes or by RL Scheduler CMA-ES pop16/gen20 and related matched-budget objectives; coverage-aware, quality-aware, distill-initialized, std-preconditioned, retention-head, compressed, and horizon-specific runs differ in objective, initialization, head, and feature structure. |

## Scope And Evidence

This report summarizes the research process visible in the current repository
from 2026-04-23 through 2026-05-31. It focuses on two linked experiment lines:

- the RL Scheduler line, which evaluates multi-card schedulers against FSRS6
  desired-retention Pareto frontiers under formal retention-sweep workflows;
- the Single-Card Tradeoff line, which studies the same memory/time tradeoff in
  an iid single-card lifecycle and uses exact or approximate oracle policies to
  understand scheduler structure.

The summary is based on current git history and published experiment documents.
The relevant git history includes about 188 RL Scheduler related commits and
about 108 Single-Card related commits in this window. The main document sources
are:

- `README.md`
- `docs/rl_scheduler/reboot/*.md`
- `docs/rl_scheduler/experiments/*.md`
- `docs/single_card_tradeoff/experiments/*.md`
- `experiments/single_card_tradeoff/README.md`

The narrative below treats current machine-readable experiment reports as the
authoritative record when they exist. Commit messages are used to reconstruct
sequence and intent, not as substitute evidence for results.

## Executive Summary

The research arc moved through four stages.

First, the project rebuilt RL Scheduler experimentation around reproducible
TOML-driven stages, typed artifacts, GPU guardrails, batched retention sweeps,
and Pareto metrics. This was not just infrastructure work: it changed the
research standard from "training objective improved" to "external Pareto
frontier improved under a staged baseline with recorded GPU and provenance
evidence."

Second, the project established a strong multi-card baseline: FSRS6 ADR
portfolio training with a matched per-user FSRS6 desired-retention baseline
manifest. The best first-eight user default became `fsrs6_adr` pop16/gen20. It
beat the FSRS6 baseline on both FSRS6 and LSTM evaluation environments, while
AP, default-weight ADR, Anki SM2 AP, and several budget variants failed to beat
it under matched conditions.

Third, the project opened the Single-Card Tradeoff line as a microscope for the
same memory/time objective. Exact finite, stationary finite, interval, and
continuous-retention oracles exposed the structure of good policies: stability
dominates, high-stability states carry most cost tradeoff, finite-horizon
policies have a deadline/no-review mode, and sparse scalarization grids can
create misleading frontier gaps. Compact distillations were surprisingly strong,
especially the 476-parameter stationary finite distill and later continuous
stationary distill variants.

Fourth, Single-Card findings were fed back into RL Scheduler through
Cost-ADR: a compact cost-conditioned policy intended to replace a portfolio of
one ADR policy per cost weight. Early coverage-aware objectives improved span
coverage but hurt Pareto quality. Distill-initialized Cost-ADR showed the first
headline win against ADR, but later fairer and more targeted ablations showed
that initialization, action head, bounds, and feature structure mattered more
than raw policy capacity. The most useful direction became a compressed
retention-head Cost-ADR policy initialized from interval-implied retention, with
structure ablation reducing the policy to 18 or 15 parameters per user.

The main conclusion is that the project shifted from scheduler search as an
optimization problem to scheduler research as a controlled Pareto frontier
science. Single-Card experiments did not directly replace multi-card RL
Scheduler results, but they identified the policy geometry that made later
Cost-ADR iterations possible.

## Core Research Question

Across both lines, the underlying question is:

How can a scheduler use memory state, cost preference, and horizon information
to trade off remembered cards against review time more efficiently than a
static desired-retention FSRS baseline?

The RL Scheduler line studies this question in a realistic multi-card
simulation:

- many cards compete for daily new/review limits;
- schedulers are evaluated over 1825-day trajectories;
- results are aggregated by per-user Pareto frontiers;
- the primary metric is scheduler-only hypervolume delta against an FSRS6
  baseline frontier;
- same-budget memory lift AUC and same-target time saved AUC diagnose the
  common covered span.

The Single-Card line removes daily deck competition and asks a cleaner
structural question:

- for one iid card lifecycle, what review interval or desired-retention policy
  is optimal for a cost-weighted memory/time objective;
- how much of that exact policy can be distilled into compact schedulers;
- which state variables and action representations actually matter?

The two lines are complementary. Single-Card can expose the shape of a good
policy; RL Scheduler tests whether that shape survives multi-card workload
constraints, priority rules, baseline staging, and external environments.

## Stage 1: Rebooting The RL Scheduler Stack

The first visible phase begins with Pareto plotting and quickly turns into a
formal reboot of the experiment stack.

Important commits and documents:

- `3e40479` on 2026-04-23 added study-time plotting on Pareto frontiers.
- 2026-04-29 commits added reboot scaffolding, TOML preflight, baseline
  staging, fail-fast runner behavior, run inspection, artifact contracts,
  train/sweep/pareto/select/aggregate/reserved-test stages, and the GPU
  utilization plan.
- `docs/rl_scheduler/reboot/roadmap.md`
- `docs/rl_scheduler/reboot/gpu-utilization-plan.md`

The reboot roadmap defined the research discipline that later reports follow:

- formal runs must be reproducible from checked-in TOML and recorded artifacts;
- stage outputs must include config snapshots, command records, run records,
  manifests, gate summaries, environment summaries, and GPU evidence;
- training reward or feasibility counts are not promotion evidence;
- promotion requires external Pareto evidence.

The GPU plan was also a research enabler. The project needed thousands of
simulator lanes across users, desired-retention points, cost weights, and
optimizer candidates. The infrastructure therefore prioritized:

- batched tensor simulation;
- super-batching axes such as `(user, desired_retention)` and
  `(user, cost_weight, candidate)`;
- performance summaries with throughput and memory;
- GPU monitor artifacts to detect shared-memory spill.

This phase created the foundation for the rest of the month: every later
conclusion depends on being able to stage identical baselines, rerun sweeps, and
compare Pareto frontiers under controlled metadata filters.

## Stage 2: FSRS6 ADR Portfolio As The First Strong RL Baseline

The next phase developed `fsrs6_adr` and related portfolio trainers.

The initial direction included simulated annealing and DR-conditioned FSRS6
schedulers, but the line converged on FSRS6 ADR portfolio optimization:

- `f063432`, `1b60ddb`, and `106b1b1` added the early SA FSRS6 policy
  scheduler, batched sweeps, and annealing trainer.
- By 2026-05-07 and 2026-05-08, CMA-ES and portfolio training became the main
  path.
- `f202072` removed simulated annealing experiments.
- `fsrs6_adr_portfolio` and `fsrs6_ap_portfolio` then became the formal
  matched-budget experiment families.

The key baseline was `fsrs6_adr_portfolio_users_1_8_pop16_v1`, a portfolio of
16 child policies per user trained with SMS-EMOA under a pop16/off16/gen20
budget. It used a per-user FSRS6 baseline DR manifest selected with CMA-ES.

Baseline DR selection matters because the portfolio is judged against the
baseline frontier. The 2026-05-21 baseline DR report records:

- selection environment: `fsrs6`;
- target count: 16 per user;
- optimizer: CMA-ES, population 16, generations 5;
- retention range: `0.50..0.98`;
- total anchor HV: `3,655,082.462`;
- selected HV: `3,687,030.135`;
- gain: `31,947.674`, or `+0.874%`;
- all 8 users improved over the uniform anchor.

Once the baseline was staged, several scheduler families were compared.

### ADR Budget Studies

The gen10/gen20/gen30 studies clarified the optimizer budget.

For pop16 ADR:

- gen10 underfit the LSTM target:
  - LSTM HV delta: `42,718`;
  - LSTM same-budget memory lift AUC: `+35.4`;
  - LSTM same-target time saved AUC: `+0.91`.
- gen20 became the working default:
  - LSTM HV delta: `55,849`;
  - LSTM same-budget memory lift AUC: `+62.2`;
  - LSTM same-target time saved AUC: `+1.82`.
- gen30 improved FSRS6 training/external metrics but regressed LSTM relative to
  gen20:
  - LSTM HV delta: `52,781`;
  - LSTM same-target time saved AUC: `+1.64`.

The conclusion was subtle but important: more training HV in the FSRS6 training
environment did not monotonically improve LSTM external validation. Gen20 was
the best default for the LSTM-focused comparison.

### LSTM-Trained ADR

The LSTM-trained ADR run tested environment-specific optimization.

Compared with FSRS-trained pop16/gen20:

- LSTM HV improved from `55,849` to `91,189`;
- LSTM same-target time saved improved from `+1.82` to `+7.39`;
- FSRS6 HV dropped from `96,880` to `54,706`.

This proved that the ADR policy class could fit the target environment, but it
also showed environment specificity. Training in LSTM helped LSTM and hurt
FSRS6.

### AP, Default ADR, ADR Time, And Anki SM2 AP

Several side branches helped define what ADR's advantage was not.

FSRS6 AP, which searches bounded FSRS6 parameter deltas, was competitive but
behind ADR:

- FSRS6 HV: AP `80,676` vs ADR `96,880`;
- LSTM HV: AP `45,380` vs ADR `55,849`;
- LSTM same-target time saved: AP `+0.80` vs ADR `+1.82`.

Default-weight ADR isolated the scheduler-side FSRS6 weight source. It failed
strongly:

- FSRS6 HV moved from ADR `+96,880` to default ADR `-17,411`;
- LSTM HV moved from ADR `+55,849` to default ADR `-72,379`.

This showed that user-fitted FSRS6 scheduler state scale matters. ADR was not
just a generic formula over arbitrary default weights.

ADR with a remaining-time feature was inconclusive:

- FSRS6 delta vs ADR: `+520` HV;
- LSTM delta vs ADR: `-7,419` HV;
- same-budget memory lift improved, but same-target time saved weakened on
  LSTM.

Anki SM2 AP was rejected:

- FSRS6 delta vs ADR: `-354,851` HV;
- LSTM delta vs ADR: `-400,193` HV.

Together these branches left ordinary user-fitted FSRS6 ADR pop16/gen20 as the
strong first-eight-user RL baseline.

## Stage 3: Single-Card Tradeoff As A Policy Geometry Lab

The Single-Card line began around 2026-05-14 and rapidly expanded through
oracles, PPO, distillation, exact policy analysis, and target-memory search.

Important commits include:

- `de15cab`: single-card tradeoff sweep;
- `60c6a00`: UVFA PPO;
- `85982f2`: FSRS oracle frontier estimate;
- `9ea5ea8`, `c04af37`: oracle distillation and interval oracle
  distillation;
- `729648a`, `b7ae380`: infinite and stationary finite oracles;
- `e62e457`: low-parameter direct search;
- `72e34e7` and `a9d4420`: layered and modularized tradeoff package;
- `daf9a9e`: native FSRS6 ADR training;
- `6bbda2c`: continuous retention oracle distill;
- `1540749`, `fec9233`, `2c6a4c5`: implied cost weights, target search, and
  certified target-memory oracle frontier;
- `eda0db4`: cost-conditioned ADR policy.

The Single-Card experiments simulate an iid single-card lifecycle with no daily
study-budget constraint. Card-level metrics are scaled to a 10,000-card deck,
so the same memory/time frontier language can be used.

### Early Strong Baselines

The first structured report index records several strong baselines:

- UVFA PPO reached `24.89%` relative time saved at `99.16%` coverage, but used
  27,148 parameters.
- Discrete oracle distill reached `24.18%` relative time saved at `86.29%`
  coverage with 1,468 parameters.
- Interval distill reached `30.03%` relative time saved at `100.00%` coverage
  in the rerun comparison.
- First-eight per-user stationary finite distill reached `8.76%` relative time
  saved at `97.20%` coverage in the repaired rerun.

These were not direct replacements for RL Scheduler policies, because the
environment removes multi-card competition. But they provided upper bounds and
compact policy shapes.

### Exact Stationary Finite Versus Distill

The exact stationary finite teacher versus 476-parameter distill comparison
showed that the exact teacher was still ahead:

- exact stationary finite:
  - mean same-target time saved vs FSRS6: `5.1976`;
  - mean relative time saved: `13.13%`;
  - coverage: `98.90%`.
- per-user distill:
  - mean same-target time saved: `3.1652`;
  - mean relative time saved: `8.76%`;
  - coverage: `97.20%`.

On the shared exact-teacher span, distill was `-5.12%` relative time saved at
`98.04%` coverage. The teacher remained meaningful, but the distill was still
positive versus FSRS6.

The r4d1/e512 compression validation tested whether a 132-parameter student
could replace the 476-parameter model. It did not validate as a drop-in
replacement:

- 132 parameters at 512 epochs:
  - mean coverage `95.52%`;
  - mean relative time saved `10.67%`.
- 476-parameter reference:
  - mean coverage `97.55%`;
  - mean relative time saved `12.36%`.

The largest coverage loss was user 5, with `-11.25` coverage points. This
established a recurring theme: average compression can look good while a few
users lose frontier span.

### ADR Versus Single-Card Oracles

The native single-card ADR comparison showed that ADR is competitive but not at
the oracle frontier.

Against the 476-parameter stationary finite distill:

- ADR mean absolute AUC: `6.2885`;
- distill mean absolute AUC: `5.0314`;
- ADR mean coverage: `76.75%`;
- distill mean coverage: `97.52%`;
- distill mean relative time saved: `12.86%`, higher than ADR's `11.40%`.

Against exact stationary finite teacher, direct pairwise comparison favored the
exact teacher on 7 of 8 users. A user-2 dense scalarization diagnostic showed
that the apparent user-2 deficit came mostly from sparse exact cost-weight
grid sampling:

- formal coarse grid exact vs ADR AUC: `-4.0910`;
- dense grid: `-0.3168`;
- dense low-weight grid: `+1.2916`.

This was a major methodological finding: sparse scalarization grids can create
false deficits when the frontier bends sharply.

The broader 2026-05-22 frontier comparison placed the scheduler families in a
clear order on first-eight users:

- interval oracle: `16.97%` relative time saved, `99.99%` coverage;
- continuous finite oracle: `16.17%`, `98.05%`;
- discrete finite grid oracle: `15.19%`, `98.77%`;
- continuous stationary finite: `15.00%`, `98.38%`;
- stationary finite: `14.47%`, `99.00%`;
- continuous stationary distill: `13.23%`, `97.33%`;
- discrete stationary finite distill: `13.06%`, `99.09%`;
- native ADR with 1024 train particles: `11.20%`, `95.48%`.

Increasing ADR train particles from 64 to 512 to 1024 improved relative AUC
from `9.63%` to `10.64%` to `11.20%`, and all 152 training gates passed at
1024. The remaining gap was therefore not only estimator noise; the ADR policy
class or search setup was still below oracle/distill rows.

### Policy Geometry Findings

The Single-Card analysis produced several structural findings that later
Cost-ADR work reused.

Finite continuous-retention policies have a deadline mode:

- with one day remaining, every positive cost weight had mean retention near
  `0.5003`;
- about `99.4%` of table cells were at `retention_min`;
- even at zero review cost, `59.0%` of cells were at the lower bound.

This indicates a terminal/no-review behavior that stationary policies cannot
represent.

Stationary continuous finite policies are mostly stability policies:

- across representative weights, the mean-retention range over stability was
  roughly `0.31..0.34`;
- the difficulty range was much smaller, from `0.002` at `w=0` to `0.131` at
  `w=256`;
- low stability is pinned by one-day interval geometry;
- high stability carries most cost tradeoff.

Cost-weight training also mattered. The `add_4_only` train-weight control became
the selected default for 476-parameter stationary finite distill:

- sparse baseline mean relative AUC: `8.76%`;
- `add_4_only` mean relative AUC: `12.02%`;
- mean coverage: `98.31%`;
- all gates passed.

Adding both 1 and 4 repaired user 2 more strongly but regressed user 7, so
`add_4_only` was promoted.

The target-memory line provided another view of the same frontier. It certified
56/56 discrete stationary oracle target answers, then compared full-coverage
non-oracle schedulers by extra time:

- FSRS6 desired-retention baseline was closest to the certified oracle on mean
  extra time among non-oracle rows;
- continuous stationary distill was next;
- ADR and discrete stationary distill were close but slightly worse;
- direct target-conditioned policies looked fast on feasible rows but covered
  only 36/56 confirmed targets.

The lesson was again that feasibility and coverage must be audited, not
inferred from train-time objectives.

## Stage 4: Feeding Single-Card Structure Back Into RL Scheduler

The bridge back into RL Scheduler began with oracle stationary finite distill
portfolio experiments and then became Cost-ADR.

### Oracle Distill Portfolio In RL Scheduler

The first multi-card attempt used per-user FSRS6 oracle stationary finite
distill policies with searched goal-cost weights. It was not a clean promotion:

- versus ADR, fsrs6 delta: `+6,990` HV, but same-target time saved `-0.15`;
- versus ADR, lstm delta: `-10,360` HV, same-target time saved `-1.54`.

The w11 variant improved FSRS6 but still lagged LSTM:

- fsrs6 delta vs ADR: `+9,591` HV;
- lstm delta vs ADR: `-10,661` HV.

The r4d1/e512 variant was rejected:

- fsrs6 delta vs ADR: `-58,627` HV;
- lstm delta vs ADR: `-41,066` HV.

This established an important boundary. A good Single-Card policy is not
automatically a good multi-card scheduler. The multi-card setting has daily
card competition, scheduler priority interactions, and external environment
transfer.

### Cost-ADR: A Compact Cost-Conditioned Bridge

The Single-Card `cost_weight_conditioned_adr_policy` document proposed one
policy per user:

```text
policy(S, D, cost_weight) -> interval or desired retention
```

The initial Single-Card implementation showed feasibility:

- 24-parameter cost-conditioned ADR beat native single-card ADR on mean AUC vs
  FSRS6:
  - cost-conditioned ADR 24p: `5.6469`;
  - native ADR: `4.9303`;
- but direct shared-span comparison against ADR was positive for only 3/8 users;
- min coverage fell to about `41%`.

This was not enough for promotion, but it gave a compact policy family with
interpretable state/cost structure.

### Coverage-Aware Cost-ADR

The first formal multi-card Cost-ADR run used a coverage-aware objective. It
did what it was designed to do, but it did not produce a better scheduler:

- training used one 24-parameter policy per user;
- CMA-ES pop16/gen20;
- 16 cost weights;
- multi-user in-process training used `effective_lanes=2048`;
- every user satisfied the 90% training coverage floor.

External results were weak:

- FSRS6 HV: Cost-ADR coverage `47,692` vs ADR `96,070`;
- LSTM HV: Cost-ADR coverage `-3,922` vs ADR `52,881`;
- coverage improved, but same-budget memory lift and same-target time saved
  were worse.

The interpretation was precise: the coverage objective bought interpolation
span, not better Pareto quality.

### Quality And Hybrid Selectors

Quality-aware objective variants tried to punish baseline-dominated points.
Quality v1 helped users 1-2 but failed the all-user overfit gate. Quality v2
restored training pass but did not improve deployment metrics. A training-HV
hybrid selector improved over coverage-aware Cost-ADR:

- FSRS6 HV: `47,692 -> 57,918`;
- LSTM HV: `-3,922 -> 12,753`;
- FSRS6 relative time-save AUC: `-0.198% -> 0.140%`.

But ADR remained much stronger:

- ADR FSRS6 HV: `96,070`;
- ADR LSTM HV: `52,881`.

This branch showed that internal training selection could help, but the policy
family and initialization still needed work.

### Distill Initialization: First Headline Win And Later Caution

The `distill24 densew` run initialized Cost-ADR from Single-Card distill
policies and reported a strong matched-budget win over ADR:

- FSRS6 Cost-ADR: `122,307` HV vs ADR `96,070`;
- LSTM Cost-ADR: `75,782` HV vs ADR `52,881`;
- deltas vs ADR:
  - fsrs6: `+26,237` HV, `+28.7` memory lift AUC, `+0.97` time saved AUC;
  - lstm: `+22,901` HV, `+67.6` memory lift AUC, `+0.63` time saved AUC.

That was the first clear multi-card result where a Single-Card-inspired compact
Cost-ADR policy beat the ADR portfolio under the reported metrics.

However, later reports made the conclusion more nuanced. A mean-initialized
matched 16-weight run failed badly:

- FSRS6 HV: `-177,810`;
- LSTM HV: `-123,537`;
- behind ADR for every user in every formal environment.

The scheduler-HV std-preconditioned run was closer:

- FSRS6: `+1,641` HV vs ADR;
- LSTM: `-12,829` HV vs ADR.

The practical conclusion became: distill-informed Cost-ADR is promising, but
the result is sensitive to initialization, preconditioning, objective, and
action head. It is not enough to say "Cost-ADR works"; the precise policy
parameterization matters.

### Retention Head, Bounds, And Structure

The retention-head improvement report diagnosed why the desired-retention head
was weaker than the interval head. The interval head's implied retention
distribution extended beyond the old `[0.50, 0.98]` retention bounds:

- overall min: `0.1885`;
- q0.1%: `0.2996`;
- q99.9%: `0.9936`;
- below `0.50`: `3.84%`;
- above `0.98`: `6.05%`.

The fix was to initialize a retention-head policy by fitting interval-implied
retention and widening bounds to `[0.30, 0.995]`. This improved retention-head
results materially:

- training HV sum: `72,673 -> 100,265`;
- FSRS6 HV delta: `72,551 -> 99,155`;
- LSTM HV delta: `8,907 -> 46,745`;
- LSTM relative time-save AUC: `1.918% -> 6.052%`.

Against the interval-head run, the interval-initialized wide retention head was
slightly better:

- FSRS6 external HV: `99,155` vs `97,711`;
- LSTM external HV: `46,745` vs `40,052`.

The report concluded that retention head can match or slightly exceed interval
head when initialized from interval-implied retention and given appropriate
bounds. This is a direct example of Single-Card/interval geometry improving the
multi-card policy class.

The structure ablation then compressed the policy:

- 24-parameter full reference: FSRS6 HV `102,728`, LSTM HV `26,097`;
- 18-parameter drop `sqrt_z`: FSRS6 HV `104,175`, LSTM HV `47,522`;
- 20-parameter drop `x_d^2`: FSRS6 HV `103,019`, LSTM HV `51,870`;
- 15-parameter drop `sqrt_z + x_d^2`: FSRS6 HV `106,075`, LSTM HV `54,322`;
- 12-parameter `z2` only: FSRS6 HV `99,496`, LSTM HV `39,437`.

The 18-parameter `drop_sqrt_z` was the conservative default candidate; the
15-parameter variant won primary HV but had weaker budget coverage, making it a
more aggressive option.

### Horizon Sensitivity

The horizon comparison used the compressed 15-parameter retention-head Cost-ADR
formula and compared 365-day and 1825-day training/evaluation.

Matched horizon won:

- train365/eval365 HV: `8,910.69`;
- train1825/eval365 HV: `6,101.64`;
- train1825/eval1825 HV: `105,565.73`;
- train365/eval1825 HV: `84,943.16`.

The 365-day policy was systematically higher-retention, while the 1825-day
policy was more aggressive and more shaped by state. This result reinforced a
single-card finding: time/horizon is a real state variable, not a nuisance
constant.

## Main Research Lines

### Main Line A: Formal Multi-Card Scheduler Evaluation

This line starts with RL infrastructure and culminates in a formal language for
promotion:

- configs define all stages;
- baseline DR manifests are selected and staged;
- sweeps write logs and GPU evidence;
- Pareto analysis writes `analysis_summary.json`;
- promotion decisions use scheduler-only HV and coverage-aware AUCs.

This line produced the strong ADR pop16/gen20 baseline, then allowed later
Cost-ADR comparisons to be interpreted consistently.

### Main Line B: Single-Card Oracle Geometry

This line studies the same objective without deck competition. Its main output
is not one final policy but a set of structural facts:

- interval actions are a strong upper-bound action class;
- continuous retention improves over discrete retention in several exact rows;
- stationary policies approximate long-horizon behavior but miss the finite
  deadline mode;
- stability is the main state axis;
- high-stability states carry most cost tradeoff;
- sparse scalarization can create false pairwise deficits;
- compact distills can preserve much of exact policy behavior.

These facts directly informed Cost-ADR initialization, retention bounds, and
feature ablations.

### Main Line C: Cost-Conditioned Compact Policies

Cost-ADR is the bridge between the two lines. It tries to turn a portfolio of
many ADR policies into one compact policy conditioned on cost weight.

The Cost-ADR story is iterative:

1. Single-Card 24p cost-conditioned ADR beats native ADR on mean AUC but lacks
   coverage.
2. RL coverage-aware Cost-ADR fixes coverage but loses Pareto quality.
3. Hybrid quality selectors improve but remain below ADR.
4. Distill-initialized Cost-ADR produces a headline win.
5. Mean-init and stdpre runs show sensitivity.
6. Retention-head fitting and wider bounds fix a concrete action-head failure.
7. Structure ablation compresses to 18 or 15 parameters with better transfer.
8. Horizon comparison shows policies are horizon-sensitive.

This is the clearest example of research convergence: Single-Card analysis
generates hypotheses; RL Scheduler validates or rejects them under multi-card
constraints.

## Side Branches And Their Outcomes

| branch | purpose | outcome |
| --- | --- | --- |
| Simulated annealing FSRS6 | Early policy search route | Removed after CMA-ES/portfolio path became stronger. |
| FSRS6 AP | Search FSRS6 parameter deltas instead of ADR retention functions | Positive vs FSRS6 baseline but behind ADR on HV and AUC. |
| Default-weight ADR | Test whether user-fitted scheduler weights matter | Rejected; default weights turned positive ADR into negative external HV. |
| ADR time feature | Add remaining-time feature to ADR | Inconclusive; slight FSRS6 HV gain, LSTM HV loss. |
| Anki SM2 AP | Adaptive-parameter non-FSRS baseline | Rejected; far behind ADR in FSRS6 and LSTM HV. |
| LSTM-trained ADR | Train directly in target LSTM env | Strong LSTM improvement but FSRS6 regression; environment-specific. |
| Oracle stationary finite distill in RL Scheduler | Transfer Single-Card distill into multi-card scheduler | Inconclusive or rejected depending variant; strong Single-Card policy did not directly transfer. |
| UVFA PPO | Goal-conditioned learned single-card policy | Strong but large; useful baseline, not compact direction. |
| Target-conditioned direct policies | Meet fixed memory targets directly | Fast on feasible rows but insufficient confirmed coverage. |

## Methodological Lessons

### Training HV Is Not Promotion Evidence

Several experiments improved training HV but failed external Pareto evaluation.
Examples include default ADR, coverage-aware Cost-ADR, and some Cost-ADR
initialization variants. The project correctly shifted toward external
frontier metrics.

### Coverage Is Necessary But Not Sufficient

Coverage-aware Cost-ADR had very high coverage but weak or negative HV. Direct
target-memory policies had attractive time on feasible rows but only 36/56
confirmed targets. Coverage must be paired with Pareto quality.

### Common-Span Metrics Matter

Same-target time saved AUC and same-budget memory lift AUC use common covered
frontier spans. This avoids comparing schedulers on different memory/time
ranges, but coverage still determines how much of the frontier the diagnostic
describes.

### Sparse Scalarization Can Mislead

The Single-Card exact teacher initially appeared weak for user 2, but dense
low-weight scalarization reversed the direct comparison. This lesson applies to
multi-card portfolios as well: frontier point placement can change AUC and HV
interpretation.

### Action Representation Matters

Interval heads, retention heads, and desired-retention bounds are not
interchangeable. The retention-head improvement only worked after fitting
interval-implied retention and widening bounds to cover the actual implied
retention distribution.

### User-Specific State Scale Matters

Default-weight ADR failed even though the policy form was the same. Scheduler
state from user-fitted FSRS6 weights is part of the policy input geometry.

### Horizon Matters

Single-Card finite policies have deadline effects; Cost-ADR horizon comparison
showed matched-horizon training wins. A stationary or horizon-agnostic policy is
a structural approximation, not a universal solution.

## Current Research State

As of 2026-05-31, the practical state is:

- FSRS6 ADR pop16/gen20 remains the central multi-card baseline for first-eight
  users.
- FSRS6 baseline DR selection is formalized through the 16-DR CMA-ES manifest.
- Single-Card exact and distilled oracles provide policy-geometry guidance but
  are not direct multi-card replacements.
- Cost-ADR is the most promising bridge from Single-Card structure to RL
  Scheduler, especially the retention-head variants initialized from
  interval-implied retention.
- The best Cost-ADR direction is compressed, monotone, cost-conditioned, and
  retention-head based, with careful bounds and horizon awareness.

The most defensible next research steps are:

1. Confirm the 15-parameter and 18-parameter compressed Cost-ADR candidates
   with additional seeds or users.
2. Treat horizon as an explicit design dimension rather than assuming 1825-day
   policies transfer to shorter settings.
3. Keep dense scalarization or certified target-memory checks around any
   exact-oracle comparison.
4. Continue using formal RL Scheduler Pareto summaries for promotion decisions,
   not Single-Card metrics alone.
5. Separate policy geometry studies from final multi-card promotion reports,
   while using the former to motivate ablations.

## Evidence Map

Key RL Scheduler reports:

- `docs/rl_scheduler/experiments/2026-05-12-fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1.md`
- `docs/rl_scheduler/experiments/2026-05-12-fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.md`
- `docs/rl_scheduler/experiments/2026-05-12-fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.md`
- `docs/rl_scheduler/experiments/2026-05-12-fsrs6_ap_portfolio_users_1_8_pop16_v1.md`
- `docs/rl_scheduler/experiments/2026-05-13-fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-13-fsrs6_adr_time_portfolio_users_1_8_pop16_v1.md`
- `docs/rl_scheduler/experiments/2026-05-13-anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-18-fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.md`
- `docs/rl_scheduler/experiments/2026-05-19-fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.md`
- `docs/rl_scheduler/experiments/2026-05-21-fsrs6_adr_portfolio_users_1_8_pop16_v1_baseline_drs.md`
- `docs/rl_scheduler/experiments/2026-05-26-fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-26-fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-26-fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_retention_head_improvement.md`
- `docs/rl_scheduler/experiments/2026-05-28-fsrs6_cost_adr_structure_ablation_users_1_8_pop16_gen20_v1.md`
- `docs/rl_scheduler/experiments/2026-05-29-fsrs6_cost_adr_horizon_days_365_vs_1825_users_1_8.md`

Key Single-Card reports:

- `docs/single_card_tradeoff/experiments/2026-05-17-index.md`
- `docs/single_card_tradeoff/experiments/2026-05-17-first8_stationary_finite_distill.md`
- `docs/single_card_tradeoff/experiments/2026-05-17-first8_exact_vs_distill.md`
- `docs/single_card_tradeoff/experiments/2026-05-17-oracle_distill.md`
- `docs/single_card_tradeoff/experiments/2026-05-17-interval_oracle_distill.md`
- `docs/single_card_tradeoff/experiments/2026-05-17-stationary_finite_compression.md`
- `docs/single_card_tradeoff/experiments/2026-05-18-first8_stationary_finite_r4d1_e512.md`
- `docs/single_card_tradeoff/experiments/2026-05-19-adr_vs_476_tradeoff_first8_users.md`
- `docs/single_card_tradeoff/experiments/2026-05-20-adr_vs_stationary_finite_exact_first8_users.md`
- `docs/single_card_tradeoff/experiments/2026-05-20-distill_train_weights_1_4_control.md`
- `docs/single_card_tradeoff/experiments/2026-05-21-native_adr_vs_stationary_finite_distill_first8.md`
- `docs/single_card_tradeoff/experiments/2026-05-22-first8_oracle_distill_adr_frontier.md`
- `docs/single_card_tradeoff/experiments/2026-05-24-continuous_retention_remaining_time.md`
- `docs/single_card_tradeoff/experiments/2026-05-24-continuous_stationary_finite_sd_analysis.md`
- `docs/single_card_tradeoff/experiments/2026-05-25-target_memory_scheduler_comparison_first8.md`
- `docs/single_card_tradeoff/experiments/2026-05-26-cost_weight_conditioned_adr_policy.md`
