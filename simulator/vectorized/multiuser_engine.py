from __future__ import annotations

from typing import Callable, Optional
import time

import torch

from simulator.core import SimulationStats
from simulator.fuzz import resolve_max_interval
from simulator.vectorized.fuzz import (
    round_intervals,
    with_learning_fuzz,
    with_review_fuzz,
)
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost
from simulator.vectorized.types import VectorizedConfig


def _prefix_count(costs: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(costs, dim=1)
    allowed = (cumulative - costs) < limit[:, None]
    return allowed.sum(dim=1)


@torch.inference_mode()
def simulate_multiuser(
    *,
    days: int,
    deck_size: int,
    env_ops,
    sched_ops,
    behavior: MultiUserBehavior,
    cost_model: MultiUserCost,
    priority_mode: str = "review-first",
    seed: int = 0,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    fuzz: bool = False,
    progress: bool = False,
    progress_label: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    short_term_source: Optional[str] = None,
    learning_steps: Optional[list[float]] = None,
    relearning_steps: Optional[list[float]] = None,
    short_term_threshold: float = 0.5,
) -> list[SimulationStats]:
    config = VectorizedConfig(
        device=torch.device(device) if device is not None else None,
        dtype=dtype,
    )
    torch_device = config.device or env_ops.device
    env_dtype = config.dtype or env_ops.dtype
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)
    fuzz_max = resolve_max_interval(sched_ops)

    if short_term_source not in {None, "steps"}:
        raise ValueError("short_term_source must be None or 'steps'.")
    steps_mode = short_term_source == "steps"
    learning_steps = list(learning_steps or [])
    relearning_steps = list(relearning_steps or [])

    user_count = int(behavior.success_weights.shape[0])
    due = torch.zeros((user_count, deck_size), dtype=env_dtype, device=torch_device)
    last_review = torch.full(
        (user_count, deck_size), -1.0, dtype=env_dtype, device=torch_device
    )
    intervals = torch.zeros(
        (user_count, deck_size), dtype=env_dtype, device=torch_device
    )
    reps = torch.zeros((user_count, deck_size), dtype=torch.int64, device=torch_device)
    lapses = torch.zeros(
        (user_count, deck_size), dtype=torch.int64, device=torch_device
    )
    env_state = env_ops.init_state(user_count, deck_size)
    sched_state = sched_ops.init_state(user_count, deck_size)

    phase_none = 0
    phase_learning = 1
    phase_relearning = 2
    short_phase = torch.zeros(
        (user_count, deck_size), dtype=torch.int8, device=torch_device
    )
    short_remaining = torch.zeros(
        (user_count, deck_size), dtype=torch.int64, device=torch_device
    )

    daily_reviews = torch.zeros(
        (user_count, days), dtype=torch.int64, device=torch_device
    )
    daily_new = torch.zeros_like(daily_reviews)
    daily_lapses = torch.zeros_like(daily_reviews)
    daily_phase_reviews = torch.zeros_like(daily_reviews)
    daily_phase_lapses = torch.zeros_like(daily_reviews)
    daily_cost = torch.zeros((user_count, days), dtype=env_dtype, device=torch_device)
    daily_memorized = torch.zeros(
        (user_count, days), dtype=env_dtype, device=torch_device
    )
    daily_gpu_peak_bytes: list[int] | None = None
    if torch_device.type == "cuda":
        daily_gpu_peak_bytes = [0 for _ in range(days)]

    total_reviews = torch.zeros(user_count, dtype=torch.int64, device=torch_device)
    total_lapses = torch.zeros_like(total_reviews)
    total_cost = torch.zeros(user_count, dtype=env_dtype, device=torch_device)
    time_long_reviews = 0.0
    time_short_reviews = 0.0
    time_learning = 0.0
    short_review_loops = 0
    short_review_loop_days = 0

    new_ptr = torch.zeros(user_count, dtype=torch.int64, device=torch_device)
    if priority_mode not in {"review-first", "new-first"}:
        raise ValueError("priority_mode must be 'review-first' or 'new-first'.")

    max_new = behavior.max_new_per_day.to(device=torch_device)
    max_reviews = behavior.max_reviews_per_day.to(device=torch_device)
    max_cost = behavior.max_cost_per_day.to(device=torch_device)
    attendance_prob = behavior.attendance_prob.to(device=torch_device)
    lazy_good_bias = behavior.lazy_good_bias.to(device=torch_device)
    success_weights = behavior.success_weights.to(device=torch_device)
    learning_success_weights = behavior.learning_success_weights.to(device=torch_device)
    relearning_success_weights = behavior.relearning_success_weights.to(
        device=torch_device
    )
    first_rating_prob = behavior.first_rating_prob.to(device=torch_device)
    review_costs = cost_model.review_costs.to(device=torch_device)
    learn_costs = cost_model.learn_costs.to(device=torch_device)
    learning_review_costs = cost_model.learning_review_costs.to(device=torch_device)
    relearning_review_costs = cost_model.relearning_review_costs.to(device=torch_device)
    base_latency = cost_model.base.to(device=torch_device)
    penalty = cost_model.penalty.to(device=torch_device)

    eps_due = torch.tensor(1e-4, device=torch_device, dtype=env_dtype)
    eps_id = torch.tensor(1e-8, device=torch_device, dtype=env_dtype)
    all_ids = torch.arange(deck_size, device=torch_device).to(env_dtype)
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

    progress_bar = None
    if progress:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=days,
            desc=progress_label or "Simulating",
            unit="day",
            leave=False,
        )

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
                hard_first = _maybe_round_in_days(
                    torch.floor((steps_secs[0] + steps_secs[1]) / 2.0)
                )
            else:
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
        if progress_callback is not None:
            progress_callback(0, days)
        for day in range(days):
            if daily_gpu_peak_bytes is not None:
                torch.cuda.reset_peak_memory_stats(torch_device)
            if progress_bar is not None:
                progress_bar.update(1)
            if progress_callback is not None:
                progress_callback(day + 1, days)
            day_float = torch.tensor(float(day), device=torch_device, dtype=env_dtype)

            learned_mask = reps > 0
            if learned_mask.any():
                elapsed_all = day_float - last_review.to(env_dtype)
                memorized = env_ops.retrievability(env_state, elapsed_all)
                daily_memorized[:, day] = (memorized * learned_mask).sum(dim=1)

            attending = (
                torch.rand((user_count,), device=torch_device, generator=gen)
                <= attendance_prob
            )
            if not attending.any():
                continue

            cost_today = torch.zeros(user_count, dtype=env_dtype, device=torch_device)
            reviews_today = torch.zeros_like(total_reviews)
            lapses_today = torch.zeros_like(total_reviews)
            learned_today = torch.zeros_like(total_reviews)
            phase_reviews_today = torch.zeros_like(total_reviews)
            phase_lapses_today = torch.zeros_like(total_reviews)

            def run_long_reviews(
                limit: torch.Tensor,
            ) -> tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
                if torch.all(max_reviews <= 0):
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                    )
                need_review = (reps > 0) & (due <= day_float) & attending[:, None]
                if steps_mode:
                    need_review &= short_phase == phase_none
                if not need_review.any():
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                    )
                elapsed_all = day_float - last_review.to(env_dtype)
                primary = sched_ops.review_priority(sched_state, elapsed_all)
                key = primary + due.to(env_dtype) * eps_due + all_ids * eps_id
                large = torch.tensor(1e9, device=torch_device, dtype=env_dtype)
                key = torch.where(need_review, key, large)
                order = torch.argsort(key, dim=1)
                sorted_valid = need_review.gather(1, order)

                due_user, due_card = need_review.nonzero(as_tuple=True)
                elapsed_due = elapsed_all[due_user, due_card]
                r_due = env_ops.retrievability_entries(
                    env_state, due_user, due_card, elapsed_due
                )
                rand = torch.rand(r_due.shape, device=torch_device, generator=gen)
                fail = rand > r_due
                lazy = torch.zeros_like(fail)
                if lazy_good_bias.max().item() > 0:
                    lazy_rand = torch.rand(
                        r_due.shape, device=torch_device, generator=gen
                    )
                    lazy = lazy_rand < lazy_good_bias[due_user]
                success_weights_sel = success_weights.index_select(0, due_user)
                success_sample = (
                    torch.multinomial(
                        success_weights_sel,
                        num_samples=1,
                        replacement=True,
                        generator=gen,
                    )
                    .squeeze(1)
                    .to(torch.int64)
                    + 2
                )
                rating_due = torch.where(
                    fail,
                    torch.ones_like(success_sample),
                    torch.where(
                        lazy, torch.full_like(success_sample, 3), success_sample
                    ),
                )
                rating_full = torch.zeros_like(reps)
                rating_full[due_user, due_card] = rating_due
                r_full = torch.zeros_like(elapsed_all)
                r_full[due_user, due_card] = r_due
                base = base_latency[due_user] * (
                    1.0 + penalty[due_user] * torch.clamp(1.0 - r_due, min=0.0)
                )
                review_cost_due = base + review_costs[due_user, rating_due - 1]
                review_cost_full = torch.zeros_like(elapsed_all)
                review_cost_full[due_user, due_card] = review_cost_due

                sorted_cost = review_cost_full.gather(1, order)
                sorted_cost = torch.where(
                    sorted_valid, sorted_cost, torch.zeros_like(sorted_cost)
                )
                cumulative = torch.cumsum(sorted_cost, dim=1)
                allowed = (cumulative - sorted_cost) < limit[:, None]
                allowed &= sorted_valid
                count = allowed.sum(dim=1)
                count = torch.minimum(count, max_reviews)

                max_pos = torch.arange(deck_size, device=torch_device)[None, :]
                selected_sorted = max_pos < count[:, None]
                selected_mask = torch.zeros_like(need_review)
                selected_mask.scatter_(1, order, selected_sorted)
                selected_mask &= need_review

                sel_user, sel_card = selected_mask.nonzero(as_tuple=True)
                if sel_user.numel() == 0:
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                    )
                exec_elapsed = elapsed_all[sel_user, sel_card]
                exec_rating = rating_full[sel_user, sel_card]
                exec_prev = intervals[sel_user, sel_card].to(env_dtype)
                exec_r = r_full[sel_user, sel_card]

                env_ops.update_review(
                    env_state, sel_user, sel_card, exec_elapsed, exec_rating
                )
                intervals_next = sched_ops.update_review(
                    sched_state,
                    sel_user,
                    sel_card,
                    exec_elapsed,
                    exec_rating,
                    exec_prev,
                )
                interval_days = intervals_next.to(env_dtype)
                short_mask = torch.zeros_like(exec_rating, dtype=torch.bool)
                if steps_mode and relearning_steps_len > 0:
                    enter_mask = exec_rating == 1
                    if enter_mask.any():
                        start_remaining = torch.full_like(
                            exec_rating[enter_mask], relearning_steps_len
                        )
                        delay_secs, next_remaining, use_steps = _schedule_steps(
                            start_remaining,
                            exec_rating[enter_mask],
                            relearning_steps_secs,
                            relearning_steps_len,
                        )
                        interval_days[enter_mask] = torch.where(
                            use_steps,
                            delay_secs / day_secs,
                            interval_days[enter_mask],
                        )
                        short_mask[enter_mask] = use_steps
                        short_phase[sel_user[enter_mask], sel_card[enter_mask]] = (
                            torch.where(
                                use_steps,
                                torch.full_like(
                                    exec_rating[enter_mask], phase_relearning
                                ),
                                torch.full_like(exec_rating[enter_mask], phase_none),
                            ).to(torch.int8)
                        )
                        short_remaining[sel_user[enter_mask], sel_card[enter_mask]] = (
                            torch.where(
                                use_steps,
                                next_remaining,
                                torch.zeros_like(next_remaining),
                            )
                        )

                if short_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            short_mask.sum(), device=torch_device, generator=gen
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
                            long_mask.sum(), device=torch_device, generator=gen
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

                intervals[sel_user, sel_card] = interval_days
                last_review[sel_user, sel_card] = day_float
                due[sel_user, sel_card] = day_float + interval_days
                reps[sel_user, sel_card] += 1
                lapses[sel_user, sel_card] += (exec_rating == 1).to(torch.int64)

                review_counts = torch.bincount(sel_user, minlength=user_count)
                lapse_counts = torch.bincount(
                    sel_user[exec_rating == 1], minlength=user_count
                )
                cost_sums = torch.zeros(
                    user_count, device=torch_device, dtype=env_dtype
                )
                cost_sums.scatter_add_(
                    0, sel_user, review_cost_full[sel_user, sel_card]
                )
                # Long reviews only include phase_none cards, so phase counts == totals.
                return (
                    review_counts,
                    lapse_counts,
                    cost_sums,
                    review_counts,
                    lapse_counts,
                )

            def run_learning(limit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                remaining = deck_size - new_ptr
                candidate = torch.minimum(max_new, remaining)
                candidate = torch.where(
                    attending, candidate, torch.zeros_like(candidate)
                )
                max_candidate = int(candidate.max().item())
                if max_candidate <= 0:
                    return torch.zeros_like(total_reviews), torch.zeros_like(total_cost)
                positions = torch.arange(max_candidate, device=torch_device)[None, :]
                candidate_mask = positions < candidate[:, None]

                rating = (
                    torch.multinomial(
                        first_rating_prob,
                        num_samples=max_candidate,
                        replacement=True,
                        generator=gen,
                    ).to(torch.int64)
                    + 1
                )
                user_idx = torch.arange(user_count, device=torch_device)[:, None]
                learn_cost = learn_costs[user_idx, rating - 1]
                learn_cost = torch.where(
                    candidate_mask, learn_cost, torch.zeros_like(learn_cost)
                )
                count = _prefix_count(learn_cost, limit)
                count = torch.minimum(count, candidate)

                selected_sorted = positions < count[:, None]
                selected_sorted &= candidate_mask
                sel_user, sel_pos = selected_sorted.nonzero(as_tuple=True)
                if sel_user.numel() == 0:
                    return torch.zeros_like(total_reviews), torch.zeros_like(total_cost)
                sel_card = new_ptr[sel_user] + sel_pos
                exec_rating = rating[sel_user, sel_pos]
                exec_cost = learn_cost[sel_user, sel_pos]

                env_ops.update_learn(env_state, sel_user, sel_card, exec_rating)
                intervals_next = sched_ops.update_learn(
                    sched_state, sel_user, sel_card, exec_rating
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
                    short_phase[sel_user, sel_card] = torch.where(
                        use_steps,
                        torch.full_like(exec_rating, phase_learning),
                        torch.full_like(exec_rating, phase_none),
                    ).to(torch.int8)
                    short_remaining[sel_user, sel_card] = torch.where(
                        use_steps, next_remaining, torch.zeros_like(next_remaining)
                    )

                if short_mask.any():
                    if fuzz:
                        fuzz_factors = torch.rand(
                            short_mask.sum(), device=torch_device, generator=gen
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
                            long_mask.sum(), device=torch_device, generator=gen
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

                intervals[sel_user, sel_card] = interval_days
                last_review[sel_user, sel_card] = day_float
                due[sel_user, sel_card] = day_float + interval_days
                reps[sel_user, sel_card] = 1

                learn_counts = torch.bincount(sel_user, minlength=user_count)
                cost_sums = torch.zeros(
                    user_count, device=torch_device, dtype=env_dtype
                )
                cost_sums.scatter_add_(0, sel_user, exec_cost)
                new_ptr.add_(learn_counts)
                return learn_counts, cost_sums

            def run_short_reviews(
                day_end: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                if not steps_mode:
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
                        0,
                    )
                short_counts = torch.zeros_like(total_reviews)
                short_lapses = torch.zeros_like(total_reviews)
                short_cost = torch.zeros_like(total_cost)
                loops = 0
                while True:
                    short_mask = (
                        (short_phase != phase_none)
                        & (due < day_end)
                        & attending[:, None]
                    )
                    if not short_mask.any():
                        break
                    loops += 1
                    exec_user, exec_card = short_mask.nonzero(as_tuple=True)
                    now_tensor = due[exec_user, exec_card]
                    exec_elapsed = now_tensor - last_review[exec_user, exec_card]
                    r_due = env_ops.retrievability_entries(
                        env_state, exec_user, exec_card, exec_elapsed
                    )
                    exec_phase = short_phase[exec_user, exec_card]

                    rand = torch.rand(r_due.shape, device=torch_device, generator=gen)
                    fail = rand > r_due
                    lazy = torch.zeros_like(fail)
                    if lazy_good_bias.max().item() > 0:
                        lazy_rand = torch.rand(
                            r_due.shape, device=torch_device, generator=gen
                        )
                        lazy = lazy_rand < lazy_good_bias[exec_user]

                    weights_sel = success_weights.index_select(0, exec_user)
                    learning_mask = exec_phase == phase_learning
                    if learning_mask.any():
                        weights_sel[learning_mask] = (
                            learning_success_weights.index_select(
                                0, exec_user[learning_mask]
                            )
                        )
                    relearning_mask = exec_phase == phase_relearning
                    if relearning_mask.any():
                        weights_sel[relearning_mask] = (
                            relearning_success_weights.index_select(
                                0, exec_user[relearning_mask]
                            )
                        )

                    success_sample = (
                        torch.multinomial(
                            weights_sel,
                            num_samples=1,
                            replacement=True,
                            generator=gen,
                        )
                        .squeeze(1)
                        .to(torch.int64)
                        + 2
                    )
                    exec_rating = torch.where(
                        fail,
                        torch.ones_like(success_sample),
                        torch.where(
                            lazy, torch.full_like(success_sample, 3), success_sample
                        ),
                    )

                    base = base_latency[exec_user] * (
                        1.0 + penalty[exec_user] * torch.clamp(1.0 - r_due, min=0.0)
                    )
                    review_cost_due = review_costs[exec_user, exec_rating - 1]
                    if learning_mask.any():
                        review_cost_due[learning_mask] = learning_review_costs[
                            exec_user[learning_mask], exec_rating[learning_mask] - 1
                        ]
                    if relearning_mask.any():
                        review_cost_due[relearning_mask] = relearning_review_costs[
                            exec_user[relearning_mask], exec_rating[relearning_mask] - 1
                        ]
                    review_cost_due = base + review_cost_due

                    env_ops.update_review(
                        env_state, exec_user, exec_card, exec_elapsed, exec_rating
                    )
                    intervals_next = sched_ops.update_review(
                        sched_state,
                        exec_user,
                        exec_card,
                        exec_elapsed,
                        exec_rating,
                        intervals[exec_user, exec_card],
                    )

                    interval_days = intervals_next.to(env_dtype)
                    next_short_mask = torch.zeros_like(exec_rating, dtype=torch.bool)
                    if learning_steps_len > 0:
                        if learning_mask.any():
                            remaining = short_remaining[exec_user, exec_card]
                            delay_secs, next_remaining, use_steps = _schedule_steps(
                                remaining[learning_mask],
                                exec_rating[learning_mask],
                                learning_steps_secs,
                                learning_steps_len,
                            )
                            interval_days[learning_mask] = torch.where(
                                use_steps,
                                delay_secs / day_secs,
                                interval_days[learning_mask],
                            )
                            next_short_mask[learning_mask] = use_steps
                            short_phase[
                                exec_user[learning_mask], exec_card[learning_mask]
                            ] = torch.where(
                                use_steps,
                                torch.full_like(
                                    exec_rating[learning_mask], phase_learning
                                ),
                                torch.full_like(exec_rating[learning_mask], phase_none),
                            ).to(torch.int8)
                            short_remaining[
                                exec_user[learning_mask], exec_card[learning_mask]
                            ] = torch.where(
                                use_steps,
                                next_remaining,
                                torch.zeros_like(next_remaining),
                            )
                    if relearning_steps_len > 0:
                        if relearning_mask.any():
                            remaining = short_remaining[exec_user, exec_card]
                            delay_secs, next_remaining, use_steps = _schedule_steps(
                                remaining[relearning_mask],
                                exec_rating[relearning_mask],
                                relearning_steps_secs,
                                relearning_steps_len,
                            )
                            interval_days[relearning_mask] = torch.where(
                                use_steps,
                                delay_secs / day_secs,
                                interval_days[relearning_mask],
                            )
                            next_short_mask[relearning_mask] = use_steps
                            short_phase[
                                exec_user[relearning_mask], exec_card[relearning_mask]
                            ] = torch.where(
                                use_steps,
                                torch.full_like(
                                    exec_rating[relearning_mask], phase_relearning
                                ),
                                torch.full_like(
                                    exec_rating[relearning_mask], phase_none
                                ),
                            ).to(torch.int8)
                            short_remaining[
                                exec_user[relearning_mask], exec_card[relearning_mask]
                            ] = torch.where(
                                use_steps,
                                next_remaining,
                                torch.zeros_like(next_remaining),
                            )

                    if next_short_mask.any():
                        if fuzz:
                            fuzz_factors = torch.rand(
                                next_short_mask.sum(),
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
                                long_mask.sum(), device=torch_device, generator=gen
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

                    intervals[exec_user, exec_card] = interval_days
                    last_review[exec_user, exec_card] = now_tensor
                    floor_now = torch.floor(now_tensor)
                    due[exec_user, exec_card] = torch.where(
                        next_short_mask,
                        now_tensor + interval_days,
                        floor_now + interval_days,
                    )
                    reps[exec_user, exec_card] += 1
                    lapses[exec_user, exec_card] += (exec_rating == 1).to(torch.int64)

                    short_counts += torch.bincount(exec_user, minlength=user_count)
                    short_lapses += torch.bincount(
                        exec_user[exec_rating == 1], minlength=user_count
                    )
                    cost_sums = torch.zeros(
                        user_count, device=torch_device, dtype=env_dtype
                    )
                    cost_sums.scatter_add_(0, exec_user, review_cost_due)
                    short_cost += cost_sums

                return short_counts, short_lapses, short_cost, loops

            if priority_mode == "review-first":
                t0 = time.perf_counter()
                (
                    review_counts,
                    review_lapses,
                    review_cost,
                    phase_counts,
                    phase_lapses,
                ) = run_long_reviews(max_cost)
                time_long_reviews += time.perf_counter() - t0
                cost_today += review_cost
                reviews_today += review_counts
                lapses_today += review_lapses
                phase_reviews_today += phase_counts
                phase_lapses_today += phase_lapses
                remaining = torch.clamp(max_cost - cost_today, min=0.0)
                t0 = time.perf_counter()
                learn_counts, learn_cost = run_learning(remaining)
                time_learning += time.perf_counter() - t0
                cost_today += learn_cost
                learned_today += learn_counts
                t0 = time.perf_counter()
                short_counts, short_lapses, short_cost, short_loops = run_short_reviews(
                    day_float + 1
                )
                time_short_reviews += time.perf_counter() - t0
                cost_today += short_cost
                reviews_today += short_counts
                lapses_today += short_lapses
                if short_loops:
                    short_review_loops += short_loops
                    short_review_loop_days += 1
            else:
                t0 = time.perf_counter()
                learn_counts, learn_cost = run_learning(max_cost)
                time_learning += time.perf_counter() - t0
                cost_today += learn_cost
                learned_today += learn_counts
                remaining = torch.clamp(max_cost - cost_today, min=0.0)
                t0 = time.perf_counter()
                (
                    review_counts,
                    review_lapses,
                    review_cost,
                    phase_counts,
                    phase_lapses,
                ) = run_long_reviews(remaining)
                time_long_reviews += time.perf_counter() - t0
                cost_today += review_cost
                reviews_today += review_counts
                lapses_today += review_lapses
                phase_reviews_today += phase_counts
                phase_lapses_today += phase_lapses
                t0 = time.perf_counter()
                short_counts, short_lapses, short_cost, short_loops = run_short_reviews(
                    day_float + 1
                )
                time_short_reviews += time.perf_counter() - t0
                cost_today += short_cost
                reviews_today += short_counts
                lapses_today += short_lapses
                if short_loops:
                    short_review_loops += short_loops
                    short_review_loop_days += 1

            daily_reviews[:, day] = reviews_today
            daily_new[:, day] = learned_today
            daily_lapses[:, day] = lapses_today
            daily_cost[:, day] = cost_today
            daily_phase_reviews[:, day] = phase_reviews_today
            daily_phase_lapses[:, day] = phase_lapses_today
            total_reviews += reviews_today
            total_lapses += lapses_today
            total_cost += cost_today
            if daily_gpu_peak_bytes is not None:
                daily_gpu_peak_bytes[day] = int(
                    torch.cuda.max_memory_allocated(torch_device)
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    daily_retention = torch.full_like(daily_cost, float("nan"))
    if steps_mode:
        review_mask = daily_phase_reviews > 0
        daily_retention[review_mask] = 1.0 - (
            daily_phase_lapses[review_mask].to(env_dtype)
            / daily_phase_reviews[review_mask].to(env_dtype)
        )
    else:
        review_mask = daily_reviews > 0
        daily_retention[review_mask] = 1.0 - (
            daily_lapses[review_mask].to(env_dtype)
            / daily_reviews[review_mask].to(env_dtype)
        )

    elapsed_final = torch.tensor(float(days), device=torch_device, dtype=env_dtype)
    elapsed_final = elapsed_final - last_review.to(env_dtype)
    projected = env_ops.retrievability(env_state, elapsed_final)
    learned_mask = reps > 0
    total_projected = (projected * learned_mask).sum(dim=1)

    stats: list[SimulationStats] = []
    for user in range(user_count):
        stats.append(
            SimulationStats(
                daily_reviews=daily_reviews[user].tolist(),
                daily_new=daily_new[user].tolist(),
                daily_retention=daily_retention[user].tolist(),
                daily_cost=daily_cost[user].tolist(),
                daily_memorized=daily_memorized[user].tolist(),
                total_reviews=int(total_reviews[user].item()),
                total_lapses=int(total_lapses[user].item()),
                total_cost=float(total_cost[user].item()),
                events=[],
                total_projected_retrievability=float(total_projected[user].item()),
                daily_gpu_peak_bytes=daily_gpu_peak_bytes,
                daily_phase_reviews=daily_phase_reviews[user].tolist(),
                daily_phase_lapses=daily_phase_lapses[user].tolist(),
                timing={
                    "long_reviews_s": time_long_reviews,
                    "short_reviews_s": time_short_reviews,
                    "learning_s": time_learning,
                    "short_review_loops": float(short_review_loops),
                    "short_review_loop_days": float(short_review_loop_days),
                },
            )
        )
    return stats
