from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from tqdm import tqdm

from simulator.behavior import StochasticBehavior
from simulator.core import SimulationStats, new_first_priority, review_first_priority
from simulator.cost import StatefulCostModel
from simulator.vectorized.registry import resolve_env_ops, resolve_scheduler_ops
from simulator.vectorized.types import VectorizedConfig


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


@torch.inference_mode()
def simulate(
    *,
    days: int,
    deck_size: int,
    environment,
    scheduler,
    behavior: StochasticBehavior,
    cost_model: StatefulCostModel,
    seed: int = 0,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    lstm_batch_size: int = 2048,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SimulationStats:
    if not isinstance(behavior, StochasticBehavior):
        raise ValueError("Vectorized engine requires StochasticBehavior.")
    if not isinstance(cost_model, StatefulCostModel):
        raise ValueError("Vectorized engine requires StatefulCostModel.")

    config = VectorizedConfig(
        device=torch.device(device) if device is not None else None,
        dtype=dtype,
        lstm_batch_size=lstm_batch_size,
    )
    env_ops = resolve_env_ops(environment, config)
    sched_ops = resolve_scheduler_ops(
        scheduler, config, device=env_ops.device, dtype=env_ops.dtype
    )

    priority_mode = _resolve_priority_mode(behavior)
    torch_device = env_ops.device
    env_dtype = env_ops.dtype
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

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
    env_state = env_ops.init_state(deck_size)
    sched_state = sched_ops.init_state(deck_size)

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
                memorized = env_ops.retrievability(env_state, learned_idx, elapsed_all)
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

                r_env = env_ops.retrievability(env_state, review_idx, elapsed)

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
                lazy_rating = torch.full_like(samples, 3)
                rating = torch.where(
                    fail, rating, torch.where(lazy, lazy_rating, samples)
                )

                base_latency = cost_model.base * (
                    1.0 + cost_model.penalty * torch.clamp(1.0 - r_env, min=0.0)
                )
                review_cost = base_latency + review_costs[rating - 1]

                primary = sched_ops.review_priority(sched_state, review_idx, elapsed)
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
                prev_interval = intervals[exec_idx].to(env_dtype)

                env_ops.update_review(
                    env_state, exec_idx, exec_elapsed, exec_rating, r_env[:count]
                )
                intervals_next = sched_ops.update_review(
                    sched_state, exec_idx, exec_elapsed, exec_rating, prev_interval
                )

                interval_days = torch.clamp(
                    torch.floor(intervals_next + 0.5), min=1.0
                ).to(torch.int64)
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

                env_ops.update_learn(env_state, exec_idx, exec_rating)
                intervals_next = sched_ops.update_learn(
                    sched_state, exec_idx, exec_rating
                )
                interval_days = torch.clamp(
                    torch.floor(intervals_next + 0.5), min=1.0
                ).to(torch.int64)
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
        elapsed_final = torch.tensor(float(days), device=torch_device, dtype=env_dtype)
        elapsed_final = elapsed_final - last_review[learned_idx].to(env_dtype)
        projected = env_ops.retrievability(env_state, learned_idx, elapsed_final)
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
