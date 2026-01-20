# Extensible Spaced-Repetition Simulator

This project is a small, dependency-light simulator inspired by [*What will a general simulator of spaced repetition consist of?*](https://l-m-sherlock.notion.site/What-will-a-general-simulator-of-spaced-repetition-consist-of-2c7c250163a1809684f3fe8cf8011a00) and mirrors the [Rust FSRS simulator](https://github.com/open-spaced-repetition/fsrs-rs/blob/main/src/simulation.rs) ideas in Python. It now separates the simulator into four modules so you can stress-test schedulers against richer “real world” assumptions.

## Usage
Install dependencies with uv, then run a quick simulation (no logs, just plots):

```bash
uv sync
uv run simulate.py --priority new-first --days 90 --no-log
```

Simulation logs now store metadata + totals by default; add `--log-reviews` to include per-event logs (can be large).

Simulating SSP-MMC-FSRS policies requires precomputed policy files. Generate them in the sibling repo, then point `SSPMMCScheduler` at the outputs (see [`../SSP-MMC-FSRS/README.md`](https://github.com/open-spaced-repetition/SSP-MMC-FSRS)).

All model/scheduler weights come from [`srs-benchmark`](https://github.com/open-spaced-repetition/srs-benchmark). The LSTM environment loads `weights/LSTM/<user_id>.pth`. FSRS/HLR/DASH weights are read from `result/*.jsonl`. Pass `--user-id` (defaults to 1) and optionally `--benchmark-result`/`--benchmark-partition` to select entries; override the repo path with `--srs-benchmark-root` if needed.

More parameter combinations:

```bash
uv run simulate.py --days 30 --deck 500 --learn-limit 20 --review-limit 200 --cost-limit-minutes 60 --seed 7 --no-progress --no-log
uv run simulate.py --environment fsrs3 --scheduler fsrs3 --desired-retention 0.85 --no-log
uv run simulate.py --environment fsrs6 --scheduler hlr --desired-retention 0.8 --no-log
uv run simulate.py --scheduler fsrs6 --scheduler-priority high_difficulty --no-log
uv run simulate.py --scheduler fixed --priority review-first --no-log
uv run simulate.py --log-dir logs/runs --days 180 --seed 123
uv run simulate.py --scheduler sspmmc --sspmmc-policy ../SSP-MMC-FSRS/outputs/policies/<policy>.json --no-log
```

Retention sweep + Pareto (compare environments, optional SSP-MMC policies):

```bash
uv run experiments/retention_sweep/run_sweep.py --environments fsrs6,lstm --schedulers fsrs6
uv run experiments/retention_sweep/run_sweep.py --environments fsrs6,lstm --schedulers sspmmc
uv run experiments/retention_sweep/run_sweep.py --environments fsrs6,lstm --schedulers fsrs6,sspmmc
uv run experiments/retention_sweep/build_pareto.py --environments fsrs6,lstm --schedulers fsrs6,sspmmc --sspmmc-root ../SSP-MMC-FSRS
```

By default, SSP-MMC policies are loaded from `../SSP-MMC-FSRS/outputs/policies/user_<id>`. Override with `--sspmmc-policy-dir` or `--sspmmc-policies`. Use `--schedulers` to compare DR sweeps across schedulers; include `sspmmc` to add policy curves. Retention sweep logs default to `logs/retention_sweep/user_<id>`, and Pareto plots are saved under `experiments/retention_sweep/plots`.

FSRS6 priority modes: `low_retrievability`, `high_retrievability`, `low_difficulty`, `high_difficulty`.

## Key Concepts
- **MemoryModel / Environment**: (`simulator.core.MemoryModel`) governs how recall probability and memory state evolve. Implementations live under `simulator/models`.
- **BehaviorModel / User**: (`simulator.core.BehaviorModel`) turns hidden retrievability into observed ratings, can skip days, and sets the first rating.
- **CostModel / Workload**: (`simulator.core.CostModel`) converts each review into a dynamic time cost (e.g. longer latency when R is low).
- **Scheduler** (`simulator.core.Scheduler`): the agent under test. It only receives a `CardView` projection (history, due date, prior intervals) and returns the next interval plus its internal state.
- **simulate** (`simulator.core.simulate`): a day-stepped loop that wires all four components together.

## Architecture & Control Flow
The simulator follows an environment-agent loop where each module owns a distinct responsibility and communicates through lightweight data structures.

### Data model
| Type | Purpose |
| --- | --- |
| `Card` | Internal state tracked by the simulator (id, due, lapses, memory/scheduler state, metadata). |
| `CardView` | Scheduler/behavior-visible projection of a card. Includes history but hides the environment’s ground-truth state. |
| `ReviewLog` | `(rating, elapsed, day)` tuples appended to `Card.history` and used for logging/analysis. |
| `SimulationStats` | Time series counters plus a chronological list of `Event` records (`day`, `action`, `card_id`, `rating`, `retrievability`, `cost`, `interval`, `due`). |

### Event loop
1. **Initialize deck** – create `Card` objects, seed future queue with due dates, and set up per-day counters.
2. **Daily setup** – each simulated day:
   - Call `behavior.start_day(day, rng)` to reset attendance/limit tracking.
   - Move cards whose `due <= day` from the future queue into the ready heap. Each ready entry stores `(scheduler.review_priority(view, day), tie_breaker, card_id)` so the scheduler can hint which review should run first (e.g., lowest retrievability).
3. **Behavior-driven actions** – repeatedly ask `behavior.choose_action(day, next_review_view, next_new_view, rng)`:
   - `next_review_view` is the highest-priority ready card, `next_new_view` is a placeholder for the next unseen card.
   - Behavior may return `Action.REVIEW`, `Action.LEARN`, or `None` (stop for the day). It enforces daily limits (new/review counts, cost ceiling) and implements heuristics such as new-first vs review-first.
4. **Learning path** – when choosing `Action.LEARN`:
   - Behavior picks an initial rating via `initial_rating`.
   - `MemoryModel.init_card` sets the ground-truth stability/difficulty.
   - `Scheduler.init_card` computes the first interval and scheduler state.
   - `CostModel.learning_cost` returns task time; the simulator updates stats, records `("new")` event, and schedules the next review by pushing `(due, priority, id)` back to the future queue.
5. **Review path** – when choosing `Action.REVIEW`:
   - Compute elapsed days and call `MemoryModel.predict_retention` for true retrievability.
   - Behavior samples a rating via `review_rating`; if it returns `None` the user skipped the rest of the day and the card is deferred.
   - Otherwise update ground-truth (`MemoryModel.update_card`), ask the scheduler for the next interval (`schedule`), compute review cost, update stats, and log a `("review")` event.
6. **Deferral** – once behavior stops or limits are reached, any remaining ready reviews are deferred by setting `card.due = day + 1` and re-queuing. This ensures they appear first on the next day but retain scheduler-provided priority hints.
7. **Post-processing** – after all days, compute daily retention (`1 - lapses/reviews`) and return `SimulationStats`.

### Priority plumbing
- **Scheduler hint** – `Scheduler.review_priority(view, day)` returns a tuple (default `(due, id)`). FSRS schedulers override it to sort by predicted retrievability or difficulty. The simulator stores the hint in `Card.metadata["scheduler_priority"]`.
- **Behavior ordering** – `BehaviorModel.priority_key(view)` prepends its own policy (e.g., review-first) and consumes the scheduler hint so user strategies can favor reviews or new cards without losing the scheduler’s ordering inside each bucket.

This separation lets you benchmark schedulers against arbitrary memory models and user behaviors while keeping transparency about where each decision is made.

## Provided Models
- `FSRS6Model`: FSRS v6-style environment (21 params loaded from `srs-benchmark` for the selected user).
- `FSRS3Model`: FSRS v3-style environment (13 params loaded from `srs-benchmark`).
- `HLRModel`: half-life regression with three weights loaded from `srs-benchmark`.
- `DASHModel`: stateless logistic model with placeholder features and nine weights loaded from `srs-benchmark`.
- `LSTMModel`: neural forgetting-curve predictor inspired by the srs-benchmark LSTM (requires PyTorch and `--user-id` weights; runs on CPU by default; use `--lstm-device` to force `cuda`/`cpu`; expects day-based intervals like the original `delta_t` feature).

## Provided Schedulers
- `FSRS6Scheduler` / `FSRSScheduler`: FSRS v6-style state; loads weights from `srs-benchmark` for the selected user.
- `FSRS3Scheduler`: FSRS v3-style scheduler with weights from `srs-benchmark`.
- `HLRScheduler`: schedules using half-life regression weights from `srs-benchmark`.
- `DASHScheduler`: logistic retention solver that mirrors the DASH model and uses weights from `srs-benchmark`.
- `SSPMMCScheduler`: loads precomputed SSP-MMC-FSRS policies (JSON + `.npz`) and maintains its own FSRS6 state so it can target optimal retention under any environment.
- `FixedIntervalScheduler`: baseline that doubles intervals on success and resets on failure.

## Provided Behavior & Cost Models
- `StochasticBehavior`: configurable attendance probability, lazy-good bias, and daily limits (max new/reviews/cost).
- `StatefulCostModel`: combines FSRS state rating costs (learning/review/relearning) with a latency penalty that grows as retrievability drops.

## Extend
- Add a new memory model: subclass `MemoryModel`, implement `init_card`, `predict_retention`, and `update_card`.
- Add a new behavior model: subclass `BehaviorModel`, implement `initial_rating` and `review_rating`.
- Add a new cost model: subclass `CostModel`, implement `review_cost`.
- Add a new scheduler: subclass `Scheduler`, implement `init_card` and `schedule` that operate on `CardView`.
- Swap components in `simulate` to study how scheduler policies perform under different ground-truth models, user behaviors, and workload assumptions.
