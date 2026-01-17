# Extensible Spaced-Repetition Simulator

This project is a small, dependency-light simulator that mirrors the Rust FSRS simulator’s ideas in Python. It now separates the simulator into four modules so you can stress-test schedulers against richer “real world” assumptions.

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
- `FSRS6Model`: FSRS v6-style environment (21 params; defaults loaded from `config/fsrs6.json`).
- `FSRS3Model`: FSRS v3-style environment (13 params; defaults loaded from `config/fsrs3.json`).
- `HLRModel`: half-life regression with three weights (defaults in `config/hlr.json`).
- `DASHModel`: stateless logistic model with placeholder features (drop in your own feature builder to align with the full DASH paper); weights live in `config/dash.json`.
- `LSTMModel`: neural forgetting-curve predictor inspired by the srs-benchmark LSTM (requires PyTorch and optional pretrained weights; defaults read from `config/lstm.json` and expects day-based intervals like the original `delta_t_secs` feature).

## Provided Schedulers
- `FSRS6Scheduler` / `FSRSScheduler`: FSRS v6-style state; pass different weights to study model mismatch.
- `FSRS3Scheduler`: FSRS v3-style scheduler.
- `HLRScheduler`: schedules using half-life regression.
- `DASHScheduler`: logistic retention solver that mirrors the DASH model and searches for intervals hitting a target retention.
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
