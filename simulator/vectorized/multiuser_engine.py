from __future__ import annotations

from typing import Callable, Optional

import torch

from simulator.core import SimulationStats
from simulator.fuzz import resolve_max_interval
from simulator.vectorized.fuzz import round_intervals, with_review_fuzz
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

    user_count = int(behavior.success_weights.shape[0])
    due = torch.zeros((user_count, deck_size), dtype=torch.int64, device=torch_device)
    last_review = torch.full(
        (user_count, deck_size), -1, dtype=torch.int64, device=torch_device
    )
    intervals = torch.zeros(
        (user_count, deck_size), dtype=torch.int64, device=torch_device
    )
    reps = torch.zeros((user_count, deck_size), dtype=torch.int64, device=torch_device)
    lapses = torch.zeros(
        (user_count, deck_size), dtype=torch.int64, device=torch_device
    )
    env_state = env_ops.init_state(user_count, deck_size)
    sched_state = sched_ops.init_state(user_count, deck_size)

    daily_reviews = torch.zeros(
        (user_count, days), dtype=torch.int64, device=torch_device
    )
    daily_new = torch.zeros_like(daily_reviews)
    daily_lapses = torch.zeros_like(daily_reviews)
    daily_cost = torch.zeros((user_count, days), dtype=env_dtype, device=torch_device)
    daily_memorized = torch.zeros(
        (user_count, days), dtype=env_dtype, device=torch_device
    )

    total_reviews = torch.zeros(user_count, dtype=torch.int64, device=torch_device)
    total_lapses = torch.zeros_like(total_reviews)
    total_cost = torch.zeros(user_count, dtype=env_dtype, device=torch_device)

    new_ptr = torch.zeros(user_count, dtype=torch.int64, device=torch_device)
    if priority_mode not in {"review-first", "new-first"}:
        raise ValueError("priority_mode must be 'review-first' or 'new-first'.")

    max_new = behavior.max_new_per_day.to(device=torch_device)
    max_reviews = behavior.max_reviews_per_day.to(device=torch_device)
    max_cost = behavior.max_cost_per_day.to(device=torch_device)
    attendance_prob = behavior.attendance_prob.to(device=torch_device)
    lazy_good_bias = behavior.lazy_good_bias.to(device=torch_device)
    success_weights = behavior.success_weights.to(device=torch_device)
    first_rating_prob = behavior.first_rating_prob.to(device=torch_device)
    review_costs = cost_model.review_costs.to(device=torch_device)
    learn_costs = cost_model.learn_costs.to(device=torch_device)
    base_latency = cost_model.base.to(device=torch_device)
    penalty = cost_model.penalty.to(device=torch_device)

    eps_due = torch.tensor(1e-4, device=torch_device, dtype=env_dtype)
    eps_id = torch.tensor(1e-8, device=torch_device, dtype=env_dtype)
    all_ids = torch.arange(deck_size, device=torch_device).to(env_dtype)

    progress_bar = None
    if progress:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=days,
            desc=progress_label or "Simulating",
            unit="day",
            leave=False,
        )

    try:
        if progress_callback is not None:
            progress_callback(0, days)
        for day in range(days):
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

            def run_reviews(
                limit: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if torch.all(max_reviews <= 0):
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
                    )
                need_review = (reps > 0) & (due <= day) & attending[:, None]
                if not need_review.any():
                    return (
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_reviews),
                        torch.zeros_like(total_cost),
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
                if fuzz:
                    fuzz_factors = torch.rand(
                        intervals_next.shape, device=torch_device, generator=gen
                    )
                    interval_days = with_review_fuzz(
                        intervals_next,
                        fuzz_factors,
                        minimum=1,
                        maximum=fuzz_max,
                    )
                else:
                    interval_days = round_intervals(intervals_next, minimum=1)
                intervals[sel_user, sel_card] = interval_days
                last_review[sel_user, sel_card] = day
                due[sel_user, sel_card] = day + interval_days
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
                return review_counts, lapse_counts, cost_sums

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
                if fuzz:
                    fuzz_factors = torch.rand(
                        intervals_next.shape, device=torch_device, generator=gen
                    )
                    interval_days = with_review_fuzz(
                        intervals_next,
                        fuzz_factors,
                        minimum=1,
                        maximum=fuzz_max,
                    )
                else:
                    interval_days = round_intervals(intervals_next, minimum=1)
                intervals[sel_user, sel_card] = interval_days
                last_review[sel_user, sel_card] = day
                due[sel_user, sel_card] = day + interval_days
                reps[sel_user, sel_card] = 1

                learn_counts = torch.bincount(sel_user, minlength=user_count)
                cost_sums = torch.zeros(
                    user_count, device=torch_device, dtype=env_dtype
                )
                cost_sums.scatter_add_(0, sel_user, exec_cost)
                new_ptr.add_(learn_counts)
                return learn_counts, cost_sums

            if priority_mode == "review-first":
                review_counts, review_lapses, review_cost = run_reviews(max_cost)
                cost_today += review_cost
                reviews_today += review_counts
                lapses_today += review_lapses
                remaining = torch.clamp(max_cost - cost_today, min=0.0)
                learn_counts, learn_cost = run_learning(remaining)
                cost_today += learn_cost
                learned_today += learn_counts
            else:
                learn_counts, learn_cost = run_learning(max_cost)
                cost_today += learn_cost
                learned_today += learn_counts
                remaining = torch.clamp(max_cost - cost_today, min=0.0)
                review_counts, review_lapses, review_cost = run_reviews(remaining)
                cost_today += review_cost
                reviews_today += review_counts
                lapses_today += review_lapses

            daily_reviews[:, day] = reviews_today
            daily_new[:, day] = learned_today
            daily_lapses[:, day] = lapses_today
            daily_cost[:, day] = cost_today
            total_reviews += reviews_today
            total_lapses += lapses_today
            total_cost += cost_today
    finally:
        if progress_bar is not None:
            progress_bar.close()

    daily_retention = torch.full_like(daily_cost, float("nan"))
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
            )
        )
    return stats
