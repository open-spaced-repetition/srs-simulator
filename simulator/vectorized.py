from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from tqdm import tqdm

from simulator.behavior import StochasticBehavior
from simulator.core import SimulationStats, new_first_priority, review_first_priority
from simulator.cost import StatefulCostModel
from simulator.models.fsrs import FSRS6Model
from simulator.models.lstm import EPS as LSTM_EPS, LSTMModel
from simulator.schedulers.fsrs import FSRS6Scheduler


def _resolve_priority_mode(behavior: StochasticBehavior) -> str:
    priority_fn = getattr(behavior, "_priority_fn", None)
    if priority_fn is review_first_priority:
        return "review-first"
    if priority_fn is new_first_priority:
        return "new-first"
    raise ValueError(
        "Vectorized engine only supports review_first_priority or new_first_priority."
    )


def _prefix_count(costs: torch.Tensor, limit: Optional[float]) -> int:
    if costs.numel() == 0:
        return 0
    if limit is None or math.isinf(limit):
        return int(costs.numel())
    if limit <= 0.0:
        return 0
    cumulative = torch.cumsum(costs, dim=0)
    allowed = (cumulative - costs) < limit
    return int(allowed.sum().item())


def _clamp(values: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return torch.clamp(values, min=min_value, max=max_value)


def _forgetting_curve(
    decay: torch.Tensor,
    factor: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    s_min: float,
) -> torch.Tensor:
    return torch.pow(1.0 + factor * t / torch.clamp(s, min=s_min), decay)


def _init_state(
    weights: torch.Tensor,
    rating: torch.Tensor,
    d_min: float,
    d_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rating_f = rating.to(dtype=weights.dtype)
    s = weights[rating - 1]
    d = weights[4] - torch.exp(weights[5] * (rating_f - 1.0)) + 1.0
    d = _clamp(d, d_min, d_max)
    return s, d


def _next_d(
    weights: torch.Tensor,
    d: torch.Tensor,
    rating: torch.Tensor,
    init_d: torch.Tensor,
    d_min: float,
    d_max: float,
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    delta_d = -weights[6] * (rating_f - 3.0)
    new_d = d + delta_d * (10.0 - d) / 9.0
    new_d = weights[7] * init_d + (1.0 - weights[7]) * new_d
    return _clamp(new_d, d_min, d_max)


def _stability_short_term(
    weights: torch.Tensor, s: torch.Tensor, rating: torch.Tensor
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    sinc = torch.exp(weights[17] * (rating_f - 3.0 + weights[18])) * torch.pow(
        s, -weights[19]
    )
    safe = torch.maximum(sinc, torch.tensor(1.0, device=s.device, dtype=s.dtype))
    scale = torch.where(rating >= 3, safe, sinc)
    return s * scale


def _stability_after_success(
    weights: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    rating: torch.Tensor,
) -> torch.Tensor:
    hard_penalty = torch.where(rating == 2, weights[15], 1.0)
    easy_bonus = torch.where(rating == 4, weights[16], 1.0)
    inc = (
        torch.exp(weights[8])
        * (11.0 - d)
        * torch.pow(s, -weights[9])
        * (torch.exp((1.0 - r) * weights[10]) - 1.0)
    )
    return s * (1.0 + inc * hard_penalty * easy_bonus)


def _stability_after_failure(
    weights: torch.Tensor, s: torch.Tensor, r: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    new_s = (
        weights[11]
        * torch.pow(d, -weights[12])
        * (torch.pow(s + 1.0, weights[13]) - 1.0)
        * torch.exp((1.0 - r) * weights[14])
    )
    new_min = s / torch.exp(weights[17] * weights[18])
    return torch.minimum(new_s, new_min)


def _lstm_retention(
    elapsed_scaled: torch.Tensor,
    weights: torch.Tensor,
    stabilities: torch.Tensor,
    decays: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    denom = stabilities + eps
    total = torch.sum(
        weights * torch.pow(1.0 + elapsed_scaled.unsqueeze(-1) / denom, -decays),
        dim=-1,
    )
    total = torch.clamp(total, min=0.0, max=1.0)
    return (1.0 - eps) * total


@torch.inference_mode()
def simulate_fsrs6_vectorized(
    *,
    days: int,
    deck_size: int,
    environment: FSRS6Model,
    scheduler: FSRS6Scheduler,
    behavior: StochasticBehavior,
    cost_model: StatefulCostModel,
    seed: int = 0,
    device: Optional[str] = None,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SimulationStats:
    if not isinstance(environment, FSRS6Model):
        raise ValueError("Vectorized engine requires FSRS6Model as the environment.")
    if not isinstance(scheduler, FSRS6Scheduler):
        raise ValueError("Vectorized engine requires FSRS6Scheduler as the scheduler.")
    if not isinstance(behavior, StochasticBehavior):
        raise ValueError("Vectorized engine requires StochasticBehavior.")
    if not isinstance(cost_model, StatefulCostModel):
        raise ValueError("Vectorized engine requires StatefulCostModel.")

    priority_mode = _resolve_priority_mode(behavior)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

    env_w = torch.tensor(environment.params.weights, device=torch_device, dtype=dtype)
    sched_w = torch.tensor(scheduler.params.weights, device=torch_device, dtype=dtype)
    env_bounds = environment.params.bounds
    sched_bounds = scheduler.params.bounds

    env_decay = -env_w[20]
    env_factor = (
        torch.pow(torch.tensor(0.9, device=torch_device, dtype=dtype), 1.0 / env_decay)
        - 1.0
    )
    sched_decay = -sched_w[20]
    sched_factor = (
        torch.pow(
            torch.tensor(0.9, device=torch_device, dtype=dtype), 1.0 / sched_decay
        )
        - 1.0
    )
    sched_retention_factor = (
        torch.pow(
            torch.tensor(scheduler.desired_retention, device=torch_device, dtype=dtype),
            1.0 / sched_decay,
        )
        - 1.0
    )

    env_init_d = _clamp(
        env_w[4] - torch.exp(env_w[5] * 3.0) + 1.0, env_bounds.d_min, env_bounds.d_max
    )
    sched_init_d = _clamp(
        sched_w[4] - torch.exp(sched_w[5] * 3.0) + 1.0,
        sched_bounds.d_min,
        sched_bounds.d_max,
    )

    review_costs = torch.tensor(
        cost_model.state_costs.review, device=torch_device, dtype=dtype
    )
    learn_costs = torch.tensor(
        cost_model.state_costs.learning, device=torch_device, dtype=dtype
    )

    success_weights = torch.tensor(
        behavior.success_dist.success_weights, device=torch_device, dtype=dtype
    )
    first_rating_prob = torch.tensor(
        behavior.first_rating_prob, device=torch_device, dtype=dtype
    )

    due = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    last_review = torch.full((deck_size,), -1, dtype=torch.int64, device=torch_device)
    intervals = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    reps = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    lapses = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    mem_s = torch.full((deck_size,), env_bounds.s_min, dtype=dtype, device=torch_device)
    mem_d = torch.full((deck_size,), env_bounds.d_min, dtype=dtype, device=torch_device)
    sched_s = torch.full(
        (deck_size,), sched_bounds.s_min, dtype=dtype, device=torch_device
    )
    sched_d = torch.full(
        (deck_size,), sched_bounds.d_min, dtype=dtype, device=torch_device
    )

    daily_reviews = [0 for _ in range(days)]
    daily_new = [0 for _ in range(days)]
    daily_retention = [0.0 for _ in range(days)]
    daily_cost = [0.0 for _ in range(days)]
    daily_memorized = [0.0 for _ in range(days)]
    daily_lapses = [0 for _ in range(days)]
    total_reviews = 0
    total_lapses = 0
    total_cost = 0.0
    events = []

    new_ptr = 0
    progress_bar = None
    if progress:
        progress_bar = tqdm(total=days, desc="Simulating", unit="day", leave=False)

    max_new = behavior.max_new_per_day
    max_reviews = behavior.max_reviews_per_day
    max_cost = behavior.max_cost_per_day
    attendance_prob = behavior.attendance_prob
    lazy_good_bias = behavior.lazy_good_bias

    if max_new is None:
        max_new = deck_size
    if max_reviews is None:
        max_reviews = deck_size

    eps_due = torch.tensor(1e-4, device=torch_device, dtype=dtype)
    eps_id = torch.tensor(1e-8, device=torch_device, dtype=dtype)
    all_ids = torch.arange(deck_size, device=torch_device)

    if progress_callback is not None:
        progress_callback(0, days)
    try:
        for day in range(days):
            if progress_callback is not None:
                progress_callback(day + 1, days)
            if progress_bar is not None:
                progress_bar.update(1)
            day_float = torch.tensor(float(day), device=torch_device, dtype=dtype)

            learned_mask = reps > 0
            if learned_mask.any():
                learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
                elapsed_all = day_float - last_review[learned_idx].to(dtype)
                memorized = _forgetting_curve(
                    env_decay,
                    env_factor,
                    elapsed_all,
                    mem_s[learned_idx],
                    env_bounds.s_min,
                )
                daily_memorized[day] = float(memorized.sum().item())

            attending = True
            if attendance_prob < 1.0:
                attending = (
                    torch.rand((), device=torch_device, generator=gen).item()
                    <= attendance_prob
                )
            if not attending:
                continue

            cost_today = 0.0
            reviews_today = 0
            lapses_today = 0
            learned_today = 0

            def run_reviews(remaining_limit: Optional[float]) -> tuple[int, int, float]:
                if max_reviews <= 0:
                    return 0, 0, 0.0
                need_review = (reps > 0) & (due <= day)
                if not need_review.any():
                    return 0, 0, 0.0
                review_idx = torch.nonzero(need_review, as_tuple=False).squeeze(1)
                elapsed = day_float - last_review[review_idx].to(dtype)

                r_env = _forgetting_curve(
                    env_decay,
                    env_factor,
                    elapsed,
                    mem_s[review_idx],
                    env_bounds.s_min,
                )

                rand = torch.rand(r_env.shape, device=torch_device, generator=gen)
                fail = rand > r_env
                rating = torch.ones(r_env.shape, device=torch_device, dtype=torch.int64)
                if lazy_good_bias > 0.0:
                    rand2 = torch.rand(r_env.shape, device=torch_device, generator=gen)
                    lazy = rand2 < lazy_good_bias
                else:
                    lazy = torch.zeros_like(fail)
                samples = (
                    torch.multinomial(
                        success_weights,
                        num_samples=r_env.numel(),
                        replacement=True,
                        generator=gen,
                    )
                    .to(torch.int64)
                    .view_as(r_env)
                    + 2
                )
                rating = torch.where(fail, rating, torch.where(lazy, 3, samples))

                base_latency = cost_model.base * (
                    1.0 + cost_model.penalty * torch.clamp(1.0 - r_env, min=0.0)
                )
                review_cost = base_latency + review_costs[rating - 1]

                sched_elapsed = elapsed
                r_sched = _forgetting_curve(
                    sched_decay,
                    sched_factor,
                    sched_elapsed,
                    sched_s[review_idx],
                    sched_bounds.s_min,
                )
                if scheduler.priority_mode == "low_retrievability":
                    primary = r_sched
                elif scheduler.priority_mode == "high_retrievability":
                    primary = -r_sched
                elif scheduler.priority_mode == "low_difficulty":
                    primary = sched_d[review_idx]
                else:
                    primary = -sched_d[review_idx]
                key = (
                    primary
                    + due[review_idx].to(dtype) * eps_due
                    + all_ids[review_idx].to(dtype) * eps_id
                )
                order = torch.argsort(key)
                review_idx = review_idx[order]
                elapsed = elapsed[order]
                rating = rating[order]
                r_env = r_env[order]
                review_cost = review_cost[order]

                count = _prefix_count(review_cost, remaining_limit)
                count = min(count, max_reviews)
                if count <= 0:
                    return 0, 0, 0.0

                exec_idx = review_idx[:count]
                exec_elapsed = elapsed[:count]
                exec_rating = rating[:count]
                exec_r_env = r_env[:count]
                exec_cost = review_cost[:count]

                exec_s = mem_s[exec_idx]
                exec_d = mem_d[exec_idx]
                short_term = exec_elapsed < 1.0
                success = exec_rating > 1

                new_s = exec_s
                new_s = torch.where(
                    short_term,
                    _stability_short_term(env_w, exec_s, exec_rating),
                    new_s,
                )
                new_s = torch.where(
                    ~short_term & success,
                    _stability_after_success(
                        env_w, exec_s, exec_r_env, exec_d, exec_rating
                    ),
                    new_s,
                )
                new_s = torch.where(
                    ~short_term & ~success,
                    _stability_after_failure(env_w, exec_s, exec_r_env, exec_d),
                    new_s,
                )
                new_d = _next_d(
                    env_w,
                    exec_d,
                    exec_rating,
                    env_init_d,
                    env_bounds.d_min,
                    env_bounds.d_max,
                )
                mem_s[exec_idx] = _clamp(new_s, env_bounds.s_min, env_bounds.s_max)
                mem_d[exec_idx] = _clamp(new_d, env_bounds.d_min, env_bounds.d_max)

                sched_exec_s = sched_s[exec_idx]
                sched_exec_d = sched_d[exec_idx]
                sched_r = _forgetting_curve(
                    sched_decay,
                    sched_factor,
                    exec_elapsed,
                    sched_exec_s,
                    sched_bounds.s_min,
                )
                sched_short = exec_elapsed < 1.0
                sched_success = exec_rating > 1

                sched_new_s = sched_exec_s
                sched_new_s = torch.where(
                    sched_short,
                    _stability_short_term(sched_w, sched_exec_s, exec_rating),
                    sched_new_s,
                )
                sched_new_s = torch.where(
                    ~sched_short & sched_success,
                    _stability_after_success(
                        sched_w, sched_exec_s, sched_r, sched_exec_d, exec_rating
                    ),
                    sched_new_s,
                )
                sched_new_s = torch.where(
                    ~sched_short & ~sched_success,
                    _stability_after_failure(
                        sched_w, sched_exec_s, sched_r, sched_exec_d
                    ),
                    sched_new_s,
                )
                sched_new_d = _next_d(
                    sched_w,
                    sched_exec_d,
                    exec_rating,
                    sched_init_d,
                    sched_bounds.d_min,
                    sched_bounds.d_max,
                )
                sched_s[exec_idx] = _clamp(
                    sched_new_s, sched_bounds.s_min, sched_bounds.s_max
                )
                sched_d[exec_idx] = _clamp(
                    sched_new_d, sched_bounds.d_min, sched_bounds.d_max
                )

                intervals_next = torch.clamp(
                    sched_s[exec_idx] / sched_factor * sched_retention_factor,
                    min=1.0,
                )
                interval_days = torch.clamp(torch.round(intervals_next), min=1.0).to(
                    torch.int64
                )
                intervals[exec_idx] = interval_days
                last_review[exec_idx] = day
                due[exec_idx] = day + interval_days

                reps[exec_idx] += 1
                lapses[exec_idx] += (exec_rating == 1).to(torch.int64)

                lapse_count = int((exec_rating == 1).sum().item())
                cost_sum = float(exec_cost.sum().item())
                return count, lapse_count, cost_sum

            def run_learning(remaining_limit: Optional[float]) -> tuple[int, float]:
                nonlocal new_ptr
                remaining = deck_size - new_ptr
                if remaining <= 0 or max_new <= 0:
                    return 0, 0.0
                candidate = min(max_new, remaining)
                if candidate <= 0:
                    return 0, 0.0
                ratings = (
                    torch.multinomial(
                        first_rating_prob,
                        num_samples=candidate,
                        replacement=True,
                        generator=gen,
                    ).to(torch.int64)
                    + 1
                )
                learn_cost = learn_costs[ratings - 1]
                count = _prefix_count(learn_cost, remaining_limit)
                count = min(count, candidate)
                if count <= 0:
                    return 0, 0.0

                exec_idx = new_ptr + torch.arange(count, device=torch_device)
                exec_rating = ratings[:count]
                exec_cost = learn_cost[:count]

                s_init, d_init = _init_state(
                    env_w, exec_rating, env_bounds.d_min, env_bounds.d_max
                )
                mem_s[exec_idx] = _clamp(s_init, env_bounds.s_min, env_bounds.s_max)
                mem_d[exec_idx] = _clamp(d_init, env_bounds.d_min, env_bounds.d_max)

                sched_s_init, sched_d_init = _init_state(
                    sched_w, exec_rating, sched_bounds.d_min, sched_bounds.d_max
                )
                sched_s[exec_idx] = _clamp(
                    sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                )
                sched_d[exec_idx] = _clamp(
                    sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                )

                intervals_next = torch.clamp(
                    sched_s[exec_idx] / sched_factor * sched_retention_factor,
                    min=1.0,
                )
                interval_days = torch.clamp(torch.round(intervals_next), min=1.0).to(
                    torch.int64
                )
                intervals[exec_idx] = interval_days
                last_review[exec_idx] = day
                due[exec_idx] = day + interval_days

                reps[exec_idx] = 1

                new_ptr += count
                return count, float(exec_cost.sum().item())

            if priority_mode == "review-first":
                review_count, review_lapses, review_cost = run_reviews(max_cost)
                cost_today += review_cost
                reviews_today += review_count
                lapses_today += review_lapses

                remaining_cost = None
                if max_cost is not None:
                    remaining_cost = max_cost - cost_today
                learn_count, learn_cost = run_learning(remaining_cost)
                cost_today += learn_cost
                learned_today += learn_count
            else:
                learn_count, learn_cost = run_learning(max_cost)
                cost_today += learn_cost
                learned_today += learn_count

                remaining_cost = None
                if max_cost is not None:
                    remaining_cost = max_cost - cost_today
                review_count, review_lapses, review_cost = run_reviews(remaining_cost)
                cost_today += review_cost
                reviews_today += review_count
                lapses_today += review_lapses

            daily_reviews[day] = reviews_today
            daily_new[day] = learned_today
            daily_lapses[day] = lapses_today
            daily_cost[day] = cost_today
            total_reviews += reviews_today
            total_lapses += lapses_today
            total_cost += cost_today
    finally:
        if progress_bar is not None:
            progress_bar.close()

    for i, r in enumerate(daily_reviews):
        daily_retention[i] = 0.0 if r == 0 else 1.0 - daily_lapses[i] / r

    total_projected_retrievability = 0.0
    learned_mask = reps > 0
    if learned_mask.any():
        learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
        elapsed_final = torch.tensor(float(days), device=torch_device, dtype=dtype) - (
            last_review[learned_idx].to(dtype)
        )
        projected = _forgetting_curve(
            env_decay,
            env_factor,
            elapsed_final,
            mem_s[learned_idx],
            env_bounds.s_min,
        )
        total_projected_retrievability = float(projected.sum().item())

    return SimulationStats(
        daily_reviews=daily_reviews,
        daily_new=daily_new,
        daily_retention=daily_retention,
        daily_cost=daily_cost,
        daily_memorized=daily_memorized,
        total_reviews=total_reviews,
        total_lapses=total_lapses,
        total_cost=total_cost,
        events=events,
        total_projected_retrievability=total_projected_retrievability,
    )


@torch.inference_mode()
def simulate_lstm_vectorized(
    *,
    days: int,
    deck_size: int,
    environment: LSTMModel,
    scheduler: FSRS6Scheduler,
    behavior: StochasticBehavior,
    cost_model: StatefulCostModel,
    seed: int = 0,
    device: Optional[str] = None,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SimulationStats:
    if not isinstance(environment, LSTMModel):
        raise ValueError("Vectorized engine requires LSTMModel as the environment.")
    if not isinstance(scheduler, FSRS6Scheduler):
        raise ValueError("Vectorized engine requires FSRS6Scheduler as the scheduler.")
    if not isinstance(behavior, StochasticBehavior):
        raise ValueError("Vectorized engine requires StochasticBehavior.")
    if not isinstance(cost_model, StatefulCostModel):
        raise ValueError("Vectorized engine requires StatefulCostModel.")

    priority_mode = _resolve_priority_mode(behavior)
    torch_device = torch.device(device) if device is not None else environment.device
    if torch_device != environment.device:
        environment.network.to(torch_device)
        environment.device = torch_device
    env_dtype = next(environment.network.parameters()).dtype
    environment.dtype = env_dtype

    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

    sched_w = torch.tensor(
        scheduler.params.weights, device=torch_device, dtype=env_dtype
    )
    sched_bounds = scheduler.params.bounds
    sched_decay = -sched_w[20]
    sched_factor = (
        torch.pow(
            torch.tensor(0.9, device=torch_device, dtype=env_dtype), 1.0 / sched_decay
        )
        - 1.0
    )
    sched_retention_factor = (
        torch.pow(
            torch.tensor(
                scheduler.desired_retention, device=torch_device, dtype=env_dtype
            ),
            1.0 / sched_decay,
        )
        - 1.0
    )
    sched_init_d = _clamp(
        sched_w[4] - torch.exp(sched_w[5] * 3.0) + 1.0,
        sched_bounds.d_min,
        sched_bounds.d_max,
    )

    max_events = int(environment.max_events)
    n_curves = int(environment.network.n_curves)
    interval_scale = float(environment.interval_scale)
    use_duration_feature = environment.use_duration_feature
    duration_value = None
    if use_duration_feature:
        duration_value = torch.tensor(
            environment.default_duration_ms, device=torch_device, dtype=env_dtype
        )
    default_retention = float(environment.default_retention)

    review_costs = torch.tensor(
        cost_model.state_costs.review, device=torch_device, dtype=env_dtype
    )
    learn_costs = torch.tensor(
        cost_model.state_costs.learning, device=torch_device, dtype=env_dtype
    )

    success_weights = torch.tensor(
        behavior.success_dist.success_weights, device=torch_device, dtype=env_dtype
    )
    first_rating_prob = torch.tensor(
        behavior.first_rating_prob, device=torch_device, dtype=env_dtype
    )

    due = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    last_review = torch.full((deck_size,), -1, dtype=torch.int64, device=torch_device)
    intervals = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    reps = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    lapses = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)

    sched_s = torch.full(
        (deck_size,), sched_bounds.s_min, dtype=env_dtype, device=torch_device
    )
    sched_d = torch.full(
        (deck_size,), sched_bounds.d_min, dtype=env_dtype, device=torch_device
    )

    event_delays = torch.zeros(
        (deck_size, max_events), dtype=env_dtype, device=torch_device
    )
    event_ratings = torch.zeros(
        (deck_size, max_events), dtype=torch.int64, device=torch_device
    )
    event_counts = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    mem_w = torch.zeros((deck_size, n_curves), dtype=env_dtype, device=torch_device)
    mem_s = torch.zeros((deck_size, n_curves), dtype=env_dtype, device=torch_device)
    mem_d = torch.zeros((deck_size, n_curves), dtype=env_dtype, device=torch_device)
    has_curves = torch.zeros(deck_size, dtype=torch.bool, device=torch_device)

    daily_reviews = [0 for _ in range(days)]
    daily_new = [0 for _ in range(days)]
    daily_retention = [0.0 for _ in range(days)]
    daily_cost = [0.0 for _ in range(days)]
    daily_memorized = [0.0 for _ in range(days)]
    daily_lapses = [0 for _ in range(days)]
    total_reviews = 0
    total_lapses = 0
    total_cost = 0.0
    events = []

    new_ptr = 0
    progress_bar = None
    if progress:
        progress_bar = tqdm(total=days, desc="Simulating", unit="day", leave=False)

    max_new = behavior.max_new_per_day
    max_reviews = behavior.max_reviews_per_day
    max_cost = behavior.max_cost_per_day
    attendance_prob = behavior.attendance_prob
    lazy_good_bias = behavior.lazy_good_bias

    if max_new is None:
        max_new = deck_size
    if max_reviews is None:
        max_reviews = deck_size

    eps_due = torch.tensor(1e-4, device=torch_device, dtype=env_dtype)
    eps_id = torch.tensor(1e-8, device=torch_device, dtype=env_dtype)
    all_ids = torch.arange(deck_size, device=torch_device)

    def update_curves(
        idx: torch.Tensor, delays: torch.Tensor, ratings: torch.Tensor
    ) -> None:
        if idx.numel() == 0:
            return
        counts = event_counts[idx]
        full_mask = counts >= max_events
        if full_mask.any():
            full_idx = idx[full_mask]
            event_delays[full_idx, :-1] = event_delays[full_idx, 1:]
            event_ratings[full_idx, :-1] = event_ratings[full_idx, 1:]
            event_delays[full_idx, -1] = delays[full_mask]
            event_ratings[full_idx, -1] = ratings[full_mask]
            event_counts[full_idx] = max_events
        partial_mask = ~full_mask
        if partial_mask.any():
            part_idx = idx[partial_mask]
            positions = counts[partial_mask]
            event_delays[part_idx, positions] = delays[partial_mask]
            event_ratings[part_idx, positions] = ratings[partial_mask]
            event_counts[part_idx] = positions + 1

        lengths = event_counts[idx]
        unique_lengths = torch.unique(lengths)
        for length in unique_lengths.tolist():
            if length <= 0:
                continue
            length_mask = lengths == length
            group_idx = idx[length_mask]
            delays_group = event_delays[group_idx, :length]
            ratings_group = event_ratings[group_idx, :length]
            delay_scaled = torch.clamp(delays_group, min=0.0) * interval_scale
            seq_delay = delay_scaled.transpose(0, 1).unsqueeze(-1)
            rating_seq = ratings_group.transpose(0, 1).unsqueeze(-1).to(env_dtype)
            if use_duration_feature:
                duration_seq = duration_value.expand(seq_delay.shape)
                sequence = torch.cat([seq_delay, duration_seq, rating_seq], dim=-1)
            else:
                sequence = torch.cat([seq_delay, rating_seq], dim=-1)
            w_lnh, s_lnh, d_lnh = environment.network(sequence)
            mem_w[group_idx] = w_lnh[-1]
            mem_s[group_idx] = s_lnh[-1]
            mem_d[group_idx] = d_lnh[-1]
            has_curves[group_idx] = True

    if progress_callback is not None:
        progress_callback(0, days)
    try:
        for day in range(days):
            if progress_callback is not None:
                progress_callback(day + 1, days)
            if progress_bar is not None:
                progress_bar.update(1)
            day_float = torch.tensor(float(day), device=torch_device, dtype=env_dtype)

            learned_mask = reps > 0
            if learned_mask.any():
                learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
                elapsed_all = day_float - last_review[learned_idx].to(env_dtype)
                elapsed_scaled = torch.clamp(elapsed_all, min=0.0) * interval_scale
                memorized = torch.full(
                    elapsed_scaled.shape,
                    default_retention,
                    dtype=env_dtype,
                    device=torch_device,
                )
                curves_mask = has_curves[learned_idx]
                if curves_mask.any():
                    curves_idx = learned_idx[curves_mask]
                    memorized[curves_mask] = _lstm_retention(
                        elapsed_scaled[curves_mask],
                        mem_w[curves_idx],
                        mem_s[curves_idx],
                        mem_d[curves_idx],
                        LSTM_EPS,
                    )
                daily_memorized[day] = float(memorized.sum().item())

            attending = True
            if attendance_prob < 1.0:
                attending = (
                    torch.rand((), device=torch_device, generator=gen).item()
                    <= attendance_prob
                )
            if not attending:
                continue

            cost_today = 0.0
            reviews_today = 0
            lapses_today = 0
            learned_today = 0

            def run_reviews(remaining_limit: Optional[float]) -> tuple[int, int, float]:
                if max_reviews <= 0:
                    return 0, 0, 0.0
                need_review = (reps > 0) & (due <= day)
                if not need_review.any():
                    return 0, 0, 0.0
                review_idx = torch.nonzero(need_review, as_tuple=False).squeeze(1)
                elapsed = day_float - last_review[review_idx].to(env_dtype)
                elapsed_scaled = torch.clamp(elapsed, min=0.0) * interval_scale

                r_env = torch.full(
                    elapsed_scaled.shape,
                    default_retention,
                    dtype=env_dtype,
                    device=torch_device,
                )
                curves_mask = has_curves[review_idx]
                if curves_mask.any():
                    curves_idx = review_idx[curves_mask]
                    r_env[curves_mask] = _lstm_retention(
                        elapsed_scaled[curves_mask],
                        mem_w[curves_idx],
                        mem_s[curves_idx],
                        mem_d[curves_idx],
                        LSTM_EPS,
                    )

                rand = torch.rand(r_env.shape, device=torch_device, generator=gen)
                fail = rand > r_env
                rating = torch.ones(r_env.shape, device=torch_device, dtype=torch.int64)
                if lazy_good_bias > 0.0:
                    rand2 = torch.rand(r_env.shape, device=torch_device, generator=gen)
                    lazy = rand2 < lazy_good_bias
                else:
                    lazy = torch.zeros_like(fail)
                samples = (
                    torch.multinomial(
                        success_weights,
                        num_samples=r_env.numel(),
                        replacement=True,
                        generator=gen,
                    )
                    .to(torch.int64)
                    .view_as(r_env)
                    + 2
                )
                rating = torch.where(fail, rating, torch.where(lazy, 3, samples))

                base_latency = cost_model.base * (
                    1.0 + cost_model.penalty * torch.clamp(1.0 - r_env, min=0.0)
                )
                review_cost = base_latency + review_costs[rating - 1]

                sched_elapsed = elapsed
                r_sched = _forgetting_curve(
                    sched_decay,
                    sched_factor,
                    sched_elapsed,
                    sched_s[review_idx],
                    sched_bounds.s_min,
                )
                if scheduler.priority_mode == "low_retrievability":
                    primary = r_sched
                elif scheduler.priority_mode == "high_retrievability":
                    primary = -r_sched
                elif scheduler.priority_mode == "low_difficulty":
                    primary = sched_d[review_idx]
                else:
                    primary = -sched_d[review_idx]
                key = (
                    primary
                    + due[review_idx].to(env_dtype) * eps_due
                    + all_ids[review_idx].to(env_dtype) * eps_id
                )
                order = torch.argsort(key)
                review_idx = review_idx[order]
                elapsed = elapsed[order]
                rating = rating[order]
                r_env = r_env[order]
                review_cost = review_cost[order]

                count = _prefix_count(review_cost, remaining_limit)
                count = min(count, max_reviews)
                if count <= 0:
                    return 0, 0, 0.0

                exec_idx = review_idx[:count]
                exec_elapsed = elapsed[:count]
                exec_rating = rating[:count]
                exec_cost = review_cost[:count]

                sched_exec_s = sched_s[exec_idx]
                sched_exec_d = sched_d[exec_idx]
                sched_r = _forgetting_curve(
                    sched_decay,
                    sched_factor,
                    exec_elapsed,
                    sched_exec_s,
                    sched_bounds.s_min,
                )
                sched_short = exec_elapsed < 1.0
                sched_success = exec_rating > 1

                sched_new_s = sched_exec_s
                sched_new_s = torch.where(
                    sched_short,
                    _stability_short_term(sched_w, sched_exec_s, exec_rating),
                    sched_new_s,
                )
                sched_new_s = torch.where(
                    ~sched_short & sched_success,
                    _stability_after_success(
                        sched_w, sched_exec_s, sched_r, sched_exec_d, exec_rating
                    ),
                    sched_new_s,
                )
                sched_new_s = torch.where(
                    ~sched_short & ~sched_success,
                    _stability_after_failure(
                        sched_w, sched_exec_s, sched_r, sched_exec_d
                    ),
                    sched_new_s,
                )
                sched_new_d = _next_d(
                    sched_w,
                    sched_exec_d,
                    exec_rating,
                    sched_init_d,
                    sched_bounds.d_min,
                    sched_bounds.d_max,
                )
                sched_s[exec_idx] = _clamp(
                    sched_new_s, sched_bounds.s_min, sched_bounds.s_max
                )
                sched_d[exec_idx] = _clamp(
                    sched_new_d, sched_bounds.d_min, sched_bounds.d_max
                )

                intervals_next = torch.clamp(
                    sched_s[exec_idx] / sched_factor * sched_retention_factor,
                    min=1.0,
                )
                interval_days = torch.clamp(torch.round(intervals_next), min=1.0).to(
                    torch.int64
                )
                intervals[exec_idx] = interval_days
                last_review[exec_idx] = day
                due[exec_idx] = day + interval_days

                update_curves(exec_idx, exec_elapsed, exec_rating)

                reps[exec_idx] += 1
                lapses[exec_idx] += (exec_rating == 1).to(torch.int64)

                lapse_count = int((exec_rating == 1).sum().item())
                cost_sum = float(exec_cost.sum().item())
                return count, lapse_count, cost_sum

            def run_learning(remaining_limit: Optional[float]) -> tuple[int, float]:
                nonlocal new_ptr
                remaining = deck_size - new_ptr
                if remaining <= 0 or max_new <= 0:
                    return 0, 0.0
                candidate = min(max_new, remaining)
                if candidate <= 0:
                    return 0, 0.0
                ratings = (
                    torch.multinomial(
                        first_rating_prob,
                        num_samples=candidate,
                        replacement=True,
                        generator=gen,
                    ).to(torch.int64)
                    + 1
                )
                learn_cost = learn_costs[ratings - 1]
                count = _prefix_count(learn_cost, remaining_limit)
                count = min(count, candidate)
                if count <= 0:
                    return 0, 0.0

                exec_idx = new_ptr + torch.arange(count, device=torch_device)
                exec_rating = ratings[:count]
                exec_cost = learn_cost[:count]

                sched_s_init, sched_d_init = _init_state(
                    sched_w, exec_rating, sched_bounds.d_min, sched_bounds.d_max
                )
                sched_s[exec_idx] = _clamp(
                    sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                )
                sched_d[exec_idx] = _clamp(
                    sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                )

                intervals_next = torch.clamp(
                    sched_s[exec_idx] / sched_factor * sched_retention_factor,
                    min=1.0,
                )
                interval_days = torch.clamp(torch.round(intervals_next), min=1.0).to(
                    torch.int64
                )
                intervals[exec_idx] = interval_days
                last_review[exec_idx] = day
                due[exec_idx] = day + interval_days

                reps[exec_idx] = 1

                learn_elapsed = torch.zeros(count, device=torch_device, dtype=env_dtype)
                update_curves(exec_idx, learn_elapsed, exec_rating)

                new_ptr += count
                return count, float(exec_cost.sum().item())

            if priority_mode == "review-first":
                review_count, review_lapses, review_cost = run_reviews(max_cost)
                cost_today += review_cost
                reviews_today += review_count
                lapses_today += review_lapses

                remaining_cost = None
                if max_cost is not None:
                    remaining_cost = max_cost - cost_today
                learn_count, learn_cost = run_learning(remaining_cost)
                cost_today += learn_cost
                learned_today += learn_count
            else:
                learn_count, learn_cost = run_learning(max_cost)
                cost_today += learn_cost
                learned_today += learn_count

                remaining_cost = None
                if max_cost is not None:
                    remaining_cost = max_cost - cost_today
                review_count, review_lapses, review_cost = run_reviews(remaining_cost)
                cost_today += review_cost
                reviews_today += review_count
                lapses_today += review_lapses

            daily_reviews[day] = reviews_today
            daily_new[day] = learned_today
            daily_lapses[day] = lapses_today
            daily_cost[day] = cost_today
            total_reviews += reviews_today
            total_lapses += lapses_today
            total_cost += cost_today
    finally:
        if progress_bar is not None:
            progress_bar.close()

    for i, r in enumerate(daily_reviews):
        daily_retention[i] = 0.0 if r == 0 else 1.0 - daily_lapses[i] / r

    total_projected_retrievability = 0.0
    learned_mask = reps > 0
    if learned_mask.any():
        learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
        elapsed_final = torch.tensor(
            float(days), device=torch_device, dtype=env_dtype
        ) - (last_review[learned_idx].to(env_dtype))
        elapsed_scaled = torch.clamp(elapsed_final, min=0.0) * interval_scale
        projected = torch.full(
            elapsed_scaled.shape,
            default_retention,
            dtype=env_dtype,
            device=torch_device,
        )
        curves_mask = has_curves[learned_idx]
        if curves_mask.any():
            curves_idx = learned_idx[curves_mask]
            projected[curves_mask] = _lstm_retention(
                elapsed_scaled[curves_mask],
                mem_w[curves_idx],
                mem_s[curves_idx],
                mem_d[curves_idx],
                LSTM_EPS,
            )
        total_projected_retrievability = float(projected.sum().item())

    return SimulationStats(
        daily_reviews=daily_reviews,
        daily_new=daily_new,
        daily_retention=daily_retention,
        daily_cost=daily_cost,
        daily_memorized=daily_memorized,
        total_reviews=total_reviews,
        total_lapses=total_lapses,
        total_cost=total_cost,
        events=events,
        total_projected_retrievability=total_projected_retrievability,
    )
