# Teacher-Occupancy Sampling: Discrete Vs Continuous Distill

## Question

Why does `teacher_occupancy` hurt
`fsrs6_oracle_stationary_finite_distill`, while it works well for
`fsrs6_oracle_continuous_stationary_finite_distill`?

## Short Answer

`teacher_occupancy` is not a universally good sampling distribution. It is a
deployment-weighted sampler, so it reduces pressure on low-density table cells.
That is useful only if the model and loss can tolerate poorer off-density table
fit.

The discrete distill is an argmax classifier over 11 retention actions with
plain cross-entropy. Occupancy sampling improves accuracy on sampled states, but
it removes the broad table coverage that the classifier needs to learn action
boundaries and cost-weight generalization. The result is a large full-table
agreement collapse and worse frontier AUC.

The continuous distill is a scalar retention regressor trained through implied
log-interval behavior. Its errors are smooth, near misses are meaningful, and
the underprediction-weighted interval loss targets the execution semantics. In
that setting, occupancy sampling can focus capacity on states that actually
matter in rollout without turning every off-density mistake into a hard action
flip.

## Evidence

First-eight discrete stationary finite distill ablation:

| supervision | mean relative AUC | mean coverage | train agreement | full-table agreement |
| --- | ---: | ---: | ---: | ---: |
| `uniform_table` | 12.02% | 98.31% | 83.29% | 83.21% |
| `teacher_occupancy` | 9.45% | 98.38% | 90.93% | 62.85% |

The discrete treatment has better sampled-state agreement but much worse
full-table agreement. The rollout frontier follows the full-table/generalization
failure, not the sampled-state training metric.

Continuous stationary finite distill rows from the current artifacts:

| recipe | mean relative AUC | mean coverage | full-table retention MAE | full-table log-interval MAE |
| --- | ---: | ---: | ---: | ---: |
| old h8 run | 11.91% | 93.15% | 0.0194 | 0.1761 |
| `teacher_occupancy`, underpred, h16 | 12.63% | 97.38% | 0.0506 | 0.2881 |
| `mixed05`, underpred, h16 | 12.81% | 96.20% | 0.0172 | 0.1627 |
| `teacher_occupancy`, underpred, h8 | 12.56% | 97.05% | 0.0553 | 0.3474 |

The continuous teacher-occupancy model is not better because it fits the whole
table better; it does not. It is better because the rollout-relevant interval
behavior improves enough to raise coverage and AUC. The `mixed05` row is a
useful caution: adding uniform samples back improves full-table fit and slightly
improves mean relative AUC in this artifact set, while pure occupancy keeps a
little more coverage.

## Mechanism

Both implementations sample occupancy in the same basic way: compute exact
teacher state occupancy, normalize it inside each `(user, cost_weight)` row, and
draw `(S,D)` cells from that distribution.

The key difference is the supervised target.

In the discrete distill, the training loop uses ordinary cross-entropy on an
action label:

- output: logits over 11 retention actions
- loss: `cross_entropy(logits, label)`
- execution: `argmax(logits)` followed by simulator action

This gives no credit for near misses. Predicting a neighboring action and
predicting a far-away action are both just wrong to cross-entropy until the
logit margin changes. Worse, a small logit perturbation can flip the deployed
action. When teacher-occupancy sampling drops low-density cells, the classifier
can lose decision boundaries that are rare under the exact teacher but still
matter under the student, under interpolated/evaluation cost weights, or after
early student mistakes move the trajectory off the teacher occupancy manifold.

In the continuous distill, the output and loss are aligned to the simulator's
execution semantics:

- output: scalar desired retention
- primary behavior loss: SmoothL1 on implied log interval
- auxiliary loss: SmoothL1 on retention logit
- extra weighting: high-cost and terminal underprediction penalties

This is smoother and more local. A small retention error usually means a small
log-interval error, not a categorical action flip. The underprediction weights
also protect the high-cost failure mode that would otherwise be most damaging.
So the model can spend more capacity on high-occupancy states without the same
catastrophic boundary loss seen in the discrete classifier.

## Interpretation

For the discrete distill, uniform table supervision is acting as a regularizer:
it forces the small 476-parameter classifier to learn the global policy surface,
including low-density boundaries and off-policy states. Removing that regularizer
raises on-distribution agreement but damages the deployed policy.

For the continuous distill, the interval-aware regression loss supplies a better
inductive bias, so occupancy sampling is closer to the true deployment objective.
It still sacrifices table-wide fit, but the sacrifice is less harmful because
the action surface is continuous and the loss measures interval behavior.

## Recommendation

Keep `uniform_table` as the default for
`fsrs6_oracle_stationary_finite_distill`.

Keep treating `teacher_occupancy` as a reasonable default ingredient for the
continuous distill only together with the interval-aware underprediction loss and
adequate capacity. If optimizing the continuous recipe further, test `mixed`
sampling explicitly; the current `mixed05` artifact suggests that a uniform
component can recover full-table fit and may slightly improve mean relative AUC.

Potential discrete follow-ups:

- `mixed` table sampling for discrete distill, not pure teacher occupancy.
- Cross-entropy with action-distance or implied-interval-aware label smoothing.
- Occupancy-weighted loss plus a uniform regularization batch.

## Evidence Paths

- Discrete occupancy ablation:
  `artifacts/single_card_tradeoff/stationary_finite_distill_teacher_occupancy_ablation/`
- Continuous teacher-occupancy h16:
  `artifacts/single_card_tradeoff/continuous_stationary_finite_distill_first8_teacherocc_underpred8_term32_h16/`
- Continuous mixed h16:
  `artifacts/single_card_tradeoff/continuous_stationary_finite_distill_first8_mixed05_underpred8_term32_h16/`
- Continuous h8 teacher-occupancy:
  `artifacts/single_card_tradeoff/continuous_stationary_finite_distill_first8_teacherocc_underpred8_term32_h8/`
