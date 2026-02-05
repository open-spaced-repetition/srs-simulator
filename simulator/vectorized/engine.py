from __future__ import annotations

import math
import time
from typing import Callable, Optional, Sequence

import torch
from tqdm import tqdm

from simulator.behavior import StochasticBehavior
from simulator.core import SimulationStats, new_first_priority, review_first_priority
from simulator.cost import StatefulCostModel
from simulator.fuzz import resolve_max_interval
from simulator.schedulers.lstm import LSTMScheduler
from simulator.vectorized.fuzz import (
    round_intervals,
    with_learning_fuzz,
    with_review_fuzz,
)
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
    fuzz: bool = False,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    short_term_source: Optional[str] = None,
    learning_steps: Optional[Sequence[float]] = None,
    relearning_steps: Optional[Sequence[float]] = None,
    short_term_threshold: float = 0.5,
    short_term_loops_limit: Optional[int] = None,
) -> SimulationStats:
    if not isinstance(behavior, StochasticBehavior):
        raise ValueError("Vectorized engine requires StochasticBehavior.")
    if not isinstance(cost_model, StatefulCostModel):
        raise ValueError("Vectorized engine requires StatefulCostModel.")
    if short_term_source not in {None, "steps", "sched"}:
        raise ValueError("short_term_source must be None, 'steps', or 'sched'.")
    if short_term_source == "sched" and not isinstance(scheduler, LSTMScheduler):
        raise ValueError("--short-term-source=sched requires --sched lstm.")
    if short_term_loops_limit is not None and short_term_loops_limit < 0:
        raise ValueError("short_term_loops_limit must be >= 0.")
    steps_mode = short_term_source == "steps"
    sched_mode = short_term_source == "sched"
    learning_steps = list(learning_steps or [])
    relearning_steps = list(relearning_steps or [])

    config = VectorizedConfig(
        device=torch.device(device) if device is not None else None,
        dtype=dtype,
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
    fuzz_max = resolve_max_interval(scheduler)

    review_costs = torch.tensor(
        cost_model.state_costs.review, device=torch_device, dtype=env_dtype
    )
    learning_review_costs = torch.tensor(
        cost_model.state_costs.learning, device=torch_device, dtype=env_dtype
    )
    relearning_review_costs = torch.tensor(
        cost_model.state_costs.relearning, device=torch_device, dtype=env_dtype
    )
    learn_costs = learning_review_costs
    success_weights = torch.tensor(
        behavior.success_dist.success_weights, device=torch_device, dtype=env_dtype
    )
    learning_success_weights = torch.tensor(
        behavior.learning_success_dist.success_weights,
        device=torch_device,
        dtype=env_dtype,
    )
    relearning_success_weights = torch.tensor(
        behavior.relearning_success_dist.success_weights,
        device=torch_device,
        dtype=env_dtype,
    )
    first_rating_prob = torch.tensor(
        behavior.first_rating_prob, device=torch_device, dtype=env_dtype
    )

    due = torch.zeros(deck_size, dtype=env_dtype, device=torch_device)
    last_review = torch.full((deck_size,), -1.0, dtype=env_dtype, device=torch_device)
    intervals = torch.zeros(deck_size, dtype=env_dtype, device=torch_device)
    reps = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    lapses = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)
    env_state = env_ops.init_state(deck_size)
    sched_state = sched_ops.init_state(deck_size)

    phase_none = 0
    phase_learning = 1
    phase_relearning = 2
    short_phase = torch.zeros(deck_size, dtype=torch.int8, device=torch_device)
    short_remaining = torch.zeros(deck_size, dtype=torch.int64, device=torch_device)

    daily_reviews = [0 for _ in range(days)]
    daily_new = [0 for _ in range(days)]
    daily_retention = [0.0 for _ in range(days)]
    daily_cost = [0.0 for _ in range(days)]
    daily_memorized = [0.0 for _ in range(days)]
    daily_lapses = [0 for _ in range(days)]
    daily_phase_reviews = [0 for _ in range(days)]
    daily_phase_lapses = [0 for _ in range(days)]
    daily_short_loops = [0 for _ in range(days)] if steps_mode else None
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

    day_secs = torch.tensor(86_400.0, device=torch_device, dtype=env_dtype)
    min_short_days = torch.tensor(1.0 / 86_400.0, device=torch_device, dtype=env_dtype)
    learning_steps_secs = torch.tensor(
        [step * 60.0 for step in learning_steps], device=torch_device, dtype=env_dtype
    )
    relearning_steps_secs = torch.tensor(
        [step * 60.0 for step in relearning_steps],
        device=torch_device,
        dtype=env_dtype,
    )
    learning_steps_len = int(learning_steps_secs.numel())
    relearning_steps_len = int(relearning_steps_secs.numel())

    if progress_callback is not None:
        progress_callback(0, days)

    def _maybe_round_in_days(secs: torch.Tensor) -> torch.Tensor:
        return torch.where(
            secs > day_secs, torch.round(secs / day_secs) * day_secs, secs
        )

    def _schedule_steps(
        remaining: torch.Tensor,
        rating: torch.Tensor,
        steps_secs: torch.Tensor,
        steps_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delay_secs = torch.zeros_like(rating, dtype=env_dtype)
        next_remaining = remaining.clone()
        use_steps = torch.zeros_like(rating, dtype=torch.bool)
        if steps_len <= 0:
            return delay_secs, next_remaining, use_steps

        idx = torch.clamp(steps_len - remaining, min=0, max=steps_len - 1).to(
            torch.int64
        )

        mask_again = rating == 1
        if mask_again.any():
            delay_secs = torch.where(mask_again, steps_secs[0], delay_secs)
            next_remaining = torch.where(
                mask_again, torch.full_like(remaining, steps_len), next_remaining
            )
            use_steps |= mask_again

        mask_hard = rating == 2
        if mask_hard.any():
            hard_delay = steps_secs[idx]
            first_mask = idx == 0
            if steps_len > 1:
                # Match Anki's learning-step "hard" interval: midpoint between first two steps.
                # See anki/rslib/src/scheduler/states/fuzz.rs (hard interval for learning steps).
                hard_first = _maybe_round_in_days(
                    torch.floor((steps_secs[0] + steps_secs[1]) / 2.0)
                )
            else:
                # Single-step fallback: 1.5x the first step, capped to +1 day (Anki-style).
                hard_first = _maybe_round_in_days(
                    torch.minimum(steps_secs[0] * 1.5, steps_secs[0] + day_secs)
                )
            hard_delay = torch.where(first_mask, hard_first, hard_delay)
            delay_secs = torch.where(mask_hard, hard_delay, delay_secs)
            use_steps |= mask_hard

        mask_good = rating == 3
        if mask_good.any():
            next_idx = idx + 1
            has_next = next_idx < steps_len
            good_delay = steps_secs[torch.clamp(next_idx, max=steps_len - 1)]
            good_mask = mask_good & has_next
            delay_secs = torch.where(good_mask, good_delay, delay_secs)
            next_remaining = torch.where(
                good_mask,
                torch.clamp(steps_len - next_idx, min=0),
                next_remaining,
            )
            use_steps |= good_mask

        return delay_secs, next_remaining, use_steps

    try:
        time_long_reviews = 0.0
        time_short_reviews = 0.0
        short_review_loops = 0
        short_review_days = 0
        for day in range(days):
            if progress_callback is not None:
                progress_callback(day + 1, days)
            if progress_bar is not None:
                progress_bar.update(1)
            day_start = float(day)
            day_end = day_start + 1.0
            day_float = torch.tensor(day_start, device=torch_device, dtype=env_dtype)

            learned_mask = reps > 0
            if learned_mask.any():
                learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
                elapsed_all = day_float - last_review[learned_idx]
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
            cost_for_limits = 0.0
            long_reviews_today = 0
            short_reviews_today = 0
            lapses_today = 0
            learned_today = 0
            phase_reviews_today = 0
            phase_lapses_today = 0

            def _sample_ratings(
                r_env: torch.Tensor, phase: torch.Tensor
            ) -> torch.Tensor:
                rand = torch.rand(r_env.shape, device=torch_device, generator=gen)
                fail = rand > r_env
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
                if steps_mode or sched_mode:
                    learning_mask = phase == phase_learning
                    if learning_mask.any():
                        samples[learning_mask] = (
                            torch.multinomial(
                                learning_success_weights,
                                num_samples=int(learning_mask.sum().item()),
                                replacement=True,
                                generator=gen,
                            ).to(torch.int64)
                            + 2
                        )
                    relearning_mask = phase == phase_relearning
                    if relearning_mask.any():
                        samples[relearning_mask] = (
                            torch.multinomial(
                                relearning_success_weights,
                                num_samples=int(relearning_mask.sum().item()),
                                replacement=True,
                                generator=gen,
                            ).to(torch.int64)
                            + 2
                        )
                lazy_rating = torch.full_like(samples, 3)
                return torch.where(
                    fail,
                    torch.ones_like(samples),
                    torch.where(lazy, lazy_rating, samples),
                )

            def _review_cost_for(
                r_env: torch.Tensor, rating: torch.Tensor, phase: torch.Tensor
            ) -> torch.Tensor:
                base_latency = cost_model.base * (
                    1.0 + cost_model.penalty * torch.clamp(1.0 - r_env, min=0.0)
                )
                review_cost_component = review_costs[rating - 1]
                if steps_mode or sched_mode:
                    learning_mask = phase == phase_learning
                    if learning_mask.any():
                        review_cost_component[learning_mask] = learning_review_costs[
                            rating[learning_mask] - 1
                        ]
                    relearning_mask = phase == phase_relearning
                    if relearning_mask.any():
                        review_cost_component[relearning_mask] = (
                            relearning_review_costs[rating[relearning_mask] - 1]
                        )
                return base_latency + review_cost_component

            def run_long_reviews(
                now_tensor: torch.Tensor,
                remaining_limit: Optional[float],
                remaining_reviews: int,
            ) -> tuple[int, int, int, int, float]:
                if remaining_reviews <= 0 or max_reviews <= 0:
                    return 0, 0, 0, 0, 0.0
                need_review = (reps > 0) & (due <= now_tensor)
                if steps_mode or sched_mode:
                    need_review &= short_phase == phase_none
                if not need_review.any():
                    return 0, 0, 0, 0, 0.0

                review_idx = torch.nonzero(need_review, as_tuple=False).squeeze(1)
                elapsed = now_tensor - last_review[review_idx]
                r_env = env_ops.retrievability(env_state, review_idx, elapsed)
                phase = torch.zeros_like(review_idx, dtype=torch.int8)
                rating = _sample_ratings(r_env, phase)
                review_cost = _review_cost_for(r_env, rating, phase)

                primary = sched_ops.review_priority(sched_state, review_idx, elapsed)
                if steps_mode and (learning_steps_len > 0 or relearning_steps_len > 0):
                    short_mask = phase != phase_none
                    if short_mask.any():
                        primary = torch.where(
                            short_mask, due[review_idx].to(env_dtype), primary
                        )
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
                phase = phase[order]

                count = _prefix_count(review_cost, remaining_limit)
                count = min(count, remaining_reviews)
                if count <= 0:
                    return 0, 0, 0, 0, 0.0

                exec_idx = review_idx[:count]
                exec_elapsed = elapsed[:count]
                exec_rating = rating[:count]
                exec_cost = review_cost[:count]
                exec_phase = phase[:count]
                prev_interval = intervals[exec_idx]

                env_ops.update_review(
                    env_state, exec_idx, exec_elapsed, exec_rating, r_env[:count]
                )
                intervals_next = sched_ops.update_review(
                    sched_state, exec_idx, exec_elapsed, exec_rating, prev_interval
                )

                interval_days = intervals_next.to(env_dtype)
                short_mask = torch.zeros_like(exec_rating, dtype=torch.bool)
                if steps_mode and relearning_steps_len > 0:
                    mask_enter = exec_rating == 1
                    if mask_enter.any():
                        start_remaining = torch.full_like(
                            exec_rating[mask_enter], relearning_steps_len
                        )
                        delay_secs, next_remaining, use_steps = _schedule_steps(
                            start_remaining,
                            exec_rating[mask_enter],
                            relearning_steps_secs,
                            relearning_steps_len,
                        )
                        interval_days[mask_enter] = torch.where(
                            use_steps,
                            delay_secs / day_secs,
                            interval_days[mask_enter],
                        )
                        short_mask[mask_enter] = use_steps
                        short_phase[exec_idx[mask_enter]] = torch.where(
                            use_steps,
                            torch.full_like(exec_rating[mask_enter], phase_relearning),
                            torch.full_like(exec_rating[mask_enter], phase_none),
                        ).to(torch.int8)
                        short_remaining[exec_idx[mask_enter]] = torch.where(
                            use_steps,
                            next_remaining,
                            torch.zeros_like(next_remaining),
                        )

                elif sched_mode:
                    short_mask = interval_days < short_term_threshold
                    interval_days = torch.where(
                        short_mask,
                        torch.clamp(interval_days, min=min_short_days),
                        interval_days,
                    )
                    new_phase = torch.where(
                        short_mask,
                        torch.where(
                            exec_phase != phase_none,
                            exec_phase,
                            torch.where(
                                exec_rating == 1,
                                torch.full_like(exec_phase, phase_relearning),
                                torch.full_like(exec_phase, phase_learning),
                            ),
                        ),
                        torch.full_like(exec_phase, phase_none),
                    )
                    short_phase[exec_idx] = new_phase
                    short_remaining[exec_idx] = 0

                if short_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            interval_days[short_mask].shape,
                            device=torch_device,
                            generator=gen,
                        )
                        secs = interval_days[short_mask] * day_secs
                        secs = with_learning_fuzz(secs, fuzz_factors)
                        interval_days[short_mask] = torch.clamp(
                            secs / day_secs, min=min_short_days
                        )
                    else:
                        interval_days[short_mask] = torch.clamp(
                            interval_days[short_mask], min=min_short_days
                        )

                long_mask = ~short_mask
                if long_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            interval_days[long_mask].shape,
                            device=torch_device,
                            generator=gen,
                        )
                        interval_long = with_review_fuzz(
                            interval_days[long_mask],
                            fuzz_factors,
                            minimum=1,
                            maximum=fuzz_max,
                        )
                    else:
                        interval_long = round_intervals(
                            interval_days[long_mask], minimum=1
                        )
                    interval_days[long_mask] = interval_long.to(env_dtype)

                intervals[exec_idx] = interval_days
                last_review[exec_idx] = now_tensor
                floor_now = torch.floor(now_tensor)
                due[exec_idx] = torch.where(
                    short_mask, now_tensor + interval_days, floor_now + interval_days
                )

                reps[exec_idx] += 1
                lapses[exec_idx] += (exec_rating == 1).to(torch.int64)

                phase_review_mask = exec_phase == phase_none
                phase_review_count = int(phase_review_mask.sum().item())
                phase_lapse_count = int(
                    ((exec_rating == 1) & phase_review_mask).sum().item()
                )
                lapse_count = int((exec_rating == 1).sum().item())
                cost_sum = float(exec_cost.sum().item())
                return (
                    count,
                    lapse_count,
                    phase_review_count,
                    phase_lapse_count,
                    cost_sum,
                )

            def run_learning(
                now_tensor: torch.Tensor,
                remaining_limit: Optional[float],
                remaining_new: int,
            ) -> tuple[int, float]:
                nonlocal new_ptr
                if remaining_new <= 0 or max_new <= 0:
                    return 0, 0.0
                remaining = deck_size - new_ptr
                if remaining <= 0:
                    return 0, 0.0
                candidate = min(remaining_new, remaining)
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

                interval_days = intervals_next.to(env_dtype)
                short_mask = torch.zeros_like(exec_rating, dtype=torch.bool)
                if steps_mode and learning_steps_len > 0:
                    start_remaining = torch.full_like(exec_rating, learning_steps_len)
                    delay_secs, next_remaining, use_steps = _schedule_steps(
                        start_remaining,
                        exec_rating,
                        learning_steps_secs,
                        learning_steps_len,
                    )
                    interval_days = torch.where(
                        use_steps, delay_secs / day_secs, interval_days
                    )
                    short_mask = use_steps
                    short_phase[exec_idx] = torch.where(
                        use_steps,
                        torch.full_like(exec_rating, phase_learning),
                        torch.full_like(exec_rating, phase_none),
                    ).to(torch.int8)
                    short_remaining[exec_idx] = torch.where(
                        use_steps, next_remaining, torch.zeros_like(next_remaining)
                    )
                elif sched_mode:
                    short_mask = interval_days < short_term_threshold
                    interval_days = torch.where(
                        short_mask,
                        torch.clamp(interval_days, min=min_short_days),
                        interval_days,
                    )
                    short_phase[exec_idx] = torch.where(
                        short_mask,
                        torch.full_like(exec_rating, phase_learning),
                        torch.full_like(exec_rating, phase_none),
                    ).to(torch.int8)
                    short_remaining[exec_idx] = 0

                if short_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            interval_days[short_mask].shape,
                            device=torch_device,
                            generator=gen,
                        )
                        secs = interval_days[short_mask] * day_secs
                        secs = with_learning_fuzz(secs, fuzz_factors)
                        interval_days[short_mask] = torch.clamp(
                            secs / day_secs, min=min_short_days
                        )
                    else:
                        interval_days[short_mask] = torch.clamp(
                            interval_days[short_mask], min=min_short_days
                        )

                long_mask = ~short_mask
                if long_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            interval_days[long_mask].shape,
                            device=torch_device,
                            generator=gen,
                        )
                        interval_long = with_review_fuzz(
                            interval_days[long_mask],
                            fuzz_factors,
                            minimum=1,
                            maximum=fuzz_max,
                        )
                    else:
                        interval_long = round_intervals(
                            interval_days[long_mask], minimum=1
                        )
                    interval_days[long_mask] = interval_long.to(env_dtype)

                intervals[exec_idx] = interval_days
                last_review[exec_idx] = now_tensor
                floor_now = torch.floor(now_tensor)
                due[exec_idx] = torch.where(
                    short_mask, now_tensor + interval_days, floor_now + interval_days
                )

                reps[exec_idx] = 1

                new_ptr += count
                return count, float(exec_cost.sum().item())

            def run_short_reviews(
                day_end_tensor: torch.Tensor,
            ) -> tuple[int, int, float, int]:
                if not (steps_mode or sched_mode):
                    return 0, 0, 0.0, 0
                short_reviews = 0
                short_lapses = 0
                short_cost = 0.0
                short_loops = 0
                while True:
                    if (
                        short_term_loops_limit is not None
                        and short_loops >= short_term_loops_limit
                    ):
                        break
                    short_mask = (short_phase != phase_none) & (due < day_end_tensor)
                    if not short_mask.any():
                        break
                    short_loops += 1
                    exec_idx = torch.nonzero(short_mask, as_tuple=False).squeeze(1)
                    now_tensor = due[exec_idx]
                    exec_elapsed = now_tensor - last_review[exec_idx]
                    r_env = env_ops.retrievability(env_state, exec_idx, exec_elapsed)
                    exec_phase = short_phase[exec_idx]
                    exec_rating = _sample_ratings(r_env, exec_phase)
                    exec_cost = _review_cost_for(r_env, exec_rating, exec_phase)
                    prev_interval = intervals[exec_idx]

                    env_ops.update_review(
                        env_state, exec_idx, exec_elapsed, exec_rating, r_env
                    )
                    intervals_next = sched_ops.update_review(
                        sched_state, exec_idx, exec_elapsed, exec_rating, prev_interval
                    )

                    interval_days = intervals_next.to(env_dtype)
                    next_short_mask = torch.zeros_like(exec_rating, dtype=torch.bool)
                    if steps_mode and (
                        learning_steps_len > 0 or relearning_steps_len > 0
                    ):
                        new_phase = exec_phase.clone()
                        new_remaining = short_remaining[exec_idx].clone()

                        if learning_steps_len > 0:
                            mask_learning = exec_phase == phase_learning
                            if mask_learning.any():
                                delay_secs, next_remaining, use_steps = _schedule_steps(
                                    new_remaining[mask_learning],
                                    exec_rating[mask_learning],
                                    learning_steps_secs,
                                    learning_steps_len,
                                )
                                interval_days[mask_learning] = torch.where(
                                    use_steps,
                                    delay_secs / day_secs,
                                    interval_days[mask_learning],
                                )
                                next_short_mask[mask_learning] = use_steps
                                new_phase[mask_learning] = torch.where(
                                    use_steps,
                                    torch.full_like(
                                        new_phase[mask_learning], phase_learning
                                    ),
                                    torch.full_like(
                                        new_phase[mask_learning], phase_none
                                    ),
                                )
                                new_remaining[mask_learning] = torch.where(
                                    use_steps,
                                    next_remaining,
                                    torch.zeros_like(next_remaining),
                                )

                        if relearning_steps_len > 0:
                            mask_relearning = exec_phase == phase_relearning
                            if mask_relearning.any():
                                delay_secs, next_remaining, use_steps = _schedule_steps(
                                    new_remaining[mask_relearning],
                                    exec_rating[mask_relearning],
                                    relearning_steps_secs,
                                    relearning_steps_len,
                                )
                                interval_days[mask_relearning] = torch.where(
                                    use_steps,
                                    delay_secs / day_secs,
                                    interval_days[mask_relearning],
                                )
                                next_short_mask[mask_relearning] = use_steps
                                new_phase[mask_relearning] = torch.where(
                                    use_steps,
                                    torch.full_like(
                                        new_phase[mask_relearning], phase_relearning
                                    ),
                                    torch.full_like(
                                        new_phase[mask_relearning], phase_none
                                    ),
                                )
                                new_remaining[mask_relearning] = torch.where(
                                    use_steps,
                                    next_remaining,
                                    torch.zeros_like(next_remaining),
                                )

                        short_phase[exec_idx] = new_phase
                        short_remaining[exec_idx] = new_remaining

                    elif sched_mode:
                        next_short_mask = interval_days < short_term_threshold
                        interval_days = torch.where(
                            next_short_mask,
                            torch.clamp(interval_days, min=min_short_days),
                            interval_days,
                        )
                        new_phase = torch.where(
                            next_short_mask,
                            torch.where(
                                exec_phase != phase_none,
                                exec_phase,
                                torch.where(
                                    exec_rating == 1,
                                    torch.full_like(exec_phase, phase_relearning),
                                    torch.full_like(exec_phase, phase_learning),
                                ),
                            ),
                            torch.full_like(exec_phase, phase_none),
                        )
                        short_phase[exec_idx] = new_phase
                        short_remaining[exec_idx] = 0

                    if next_short_mask.any():
                        if fuzz:
                            fuzz_factors = torch.rand(
                                interval_days[next_short_mask].shape,
                                device=torch_device,
                                generator=gen,
                            )
                            secs = interval_days[next_short_mask] * day_secs
                            secs = with_learning_fuzz(secs, fuzz_factors)
                            interval_days[next_short_mask] = torch.clamp(
                                secs / day_secs, min=min_short_days
                            )
                        else:
                            interval_days[next_short_mask] = torch.clamp(
                                interval_days[next_short_mask], min=min_short_days
                            )

                    long_mask = ~next_short_mask
                    if long_mask.any():
                        if fuzz:
                            fuzz_factors = torch.rand(
                                interval_days[long_mask].shape,
                                device=torch_device,
                                generator=gen,
                            )
                            interval_long = with_review_fuzz(
                                interval_days[long_mask],
                                fuzz_factors,
                                minimum=1,
                                maximum=fuzz_max,
                            )
                        else:
                            interval_long = round_intervals(
                                interval_days[long_mask], minimum=1
                            )
                        interval_days[long_mask] = interval_long.to(env_dtype)

                    intervals[exec_idx] = interval_days
                    last_review[exec_idx] = now_tensor
                    floor_now = torch.floor(now_tensor)
                    due[exec_idx] = torch.where(
                        next_short_mask,
                        now_tensor + interval_days,
                        floor_now + interval_days,
                    )

                    reps[exec_idx] += 1
                    lapses[exec_idx] += (exec_rating == 1).to(torch.int64)

                    short_reviews += exec_idx.numel()
                    short_lapses += int((exec_rating == 1).sum().item())
                    short_cost += float(exec_cost.sum().item())

                return short_reviews, short_lapses, short_cost, short_loops

            day_start_tensor = torch.tensor(
                day_start, device=torch_device, dtype=env_dtype
            )
            day_end_tensor = torch.tensor(day_end, device=torch_device, dtype=env_dtype)

            if priority_mode == "review-first":
                remaining_cost = (
                    max_cost - cost_for_limits if max_cost is not None else None
                )
                reviews_left = max_reviews - long_reviews_today
                t0 = time.perf_counter()
                (
                    review_count,
                    review_lapses,
                    review_phase_count,
                    review_phase_lapses,
                    review_cost,
                ) = run_long_reviews(day_start_tensor, remaining_cost, reviews_left)
                time_long_reviews += time.perf_counter() - t0
                cost_today += review_cost
                cost_for_limits += review_cost
                long_reviews_today += review_count
                lapses_today += review_lapses
                phase_reviews_today += review_phase_count
                phase_lapses_today += review_phase_lapses

                remaining_cost = (
                    max_cost - cost_for_limits if max_cost is not None else None
                )
                new_left = max_new - learned_today
                learn_count, learn_cost = run_learning(
                    day_start_tensor, remaining_cost, new_left
                )
                cost_today += learn_cost
                cost_for_limits += learn_cost
                learned_today += learn_count

                t0 = time.perf_counter()
                short_count, short_lapses, short_cost, short_loops = run_short_reviews(
                    day_end_tensor
                )
                time_short_reviews += time.perf_counter() - t0
                cost_today += short_cost
                short_reviews_today += short_count
                lapses_today += short_lapses
                if short_loops:
                    short_review_loops += short_loops
                    short_review_days += 1
                if daily_short_loops is not None:
                    daily_short_loops[day] = short_loops
            else:
                remaining_cost = (
                    max_cost - cost_for_limits if max_cost is not None else None
                )
                new_left = max_new - learned_today
                learn_count, learn_cost = run_learning(
                    day_start_tensor, remaining_cost, new_left
                )
                cost_today += learn_cost
                cost_for_limits += learn_cost
                learned_today += learn_count

                remaining_cost = (
                    max_cost - cost_for_limits if max_cost is not None else None
                )
                reviews_left = max_reviews - long_reviews_today
                t0 = time.perf_counter()
                (
                    review_count,
                    review_lapses,
                    review_phase_count,
                    review_phase_lapses,
                    review_cost,
                ) = run_long_reviews(day_start_tensor, remaining_cost, reviews_left)
                time_long_reviews += time.perf_counter() - t0
                cost_today += review_cost
                cost_for_limits += review_cost
                long_reviews_today += review_count
                lapses_today += review_lapses
                phase_reviews_today += review_phase_count
                phase_lapses_today += review_phase_lapses

                t0 = time.perf_counter()
                short_count, short_lapses, short_cost, short_loops = run_short_reviews(
                    day_end_tensor
                )
                time_short_reviews += time.perf_counter() - t0
                cost_today += short_cost
                short_reviews_today += short_count
                lapses_today += short_lapses
                if short_loops:
                    short_review_loops += short_loops
                    short_review_days += 1
                if daily_short_loops is not None:
                    daily_short_loops[day] = short_loops

            reviews_today = long_reviews_today + short_reviews_today

            daily_reviews[day] = reviews_today
            daily_new[day] = learned_today
            daily_lapses[day] = lapses_today
            daily_cost[day] = cost_today
            daily_phase_reviews[day] = phase_reviews_today
            daily_phase_lapses[day] = phase_lapses_today
            total_reviews += reviews_today
            total_lapses += lapses_today
            total_cost += cost_today
    finally:
        if progress_bar is not None:
            progress_bar.close()

    for i, r in enumerate(daily_phase_reviews):
        daily_retention[i] = math.nan if r == 0 else 1.0 - daily_phase_lapses[i] / r

    total_projected_retrievability = 0.0
    learned_mask = reps > 0
    if learned_mask.any():
        learned_idx = torch.nonzero(learned_mask, as_tuple=False).squeeze(1)
        elapsed_final = torch.tensor(float(days), device=torch_device, dtype=env_dtype)
        elapsed_final = elapsed_final - last_review[learned_idx]
        projected = env_ops.retrievability(env_state, learned_idx, elapsed_final)
        total_projected_retrievability = float(projected.sum().item())

    stats = SimulationStats(
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
        timing={
            "long_reviews_s": time_long_reviews,
            "short_reviews_s": time_short_reviews,
            "short_review_loops": float(short_review_loops),
            "short_review_loop_days": float(short_review_days),
        },
        daily_phase_reviews=daily_phase_reviews,
        daily_phase_lapses=daily_phase_lapses,
        daily_short_loops=daily_short_loops,
    )
    return stats
