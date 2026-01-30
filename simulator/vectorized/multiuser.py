from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from simulator.core import SimulationStats
from simulator.math.fsrs import Bounds
from simulator.schedulers.anki_sm2 import _anki_next_interval
from simulator.schedulers.memrise import MemriseScheduler
from simulator.vectorized.types import VectorizedConfig


@dataclass
class MultiUserBehavior:
    attendance_prob: torch.Tensor
    lazy_good_bias: torch.Tensor
    max_new_per_day: torch.Tensor
    max_reviews_per_day: torch.Tensor
    max_cost_per_day: torch.Tensor
    success_weights: torch.Tensor
    first_rating_prob: torch.Tensor


@dataclass
class MultiUserCost:
    base: torch.Tensor
    penalty: torch.Tensor
    learn_costs: torch.Tensor
    review_costs: torch.Tensor


@dataclass
class FSRS6BatchState:
    s: torch.Tensor
    d: torch.Tensor


class FSRS6BatchSchedulerOps:
    PRIORITY_MODES = {
        "low_retrievability",
        "high_retrievability",
        "low_difficulty",
        "high_difficulty",
    }

    def __init__(
        self,
        *,
        weights: torch.Tensor,
        desired_retention: float,
        bounds: Bounds,
        priority_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        self.device = device
        self.dtype = dtype
        self._weights = weights.to(device=device, dtype=dtype)
        self._bounds = bounds
        self._decay = -self._weights[:, 20]
        base = torch.tensor(0.9, device=device, dtype=dtype)
        self._factor = torch.pow(base, 1.0 / self._decay) - 1.0
        self._retention_factor = (
            torch.pow(
                torch.tensor(desired_retention, device=device, dtype=dtype),
                1.0 / self._decay,
            )
            - 1.0
        )
        self._mean_reversion_d = torch.clamp(
            self._weights[:, 4] - torch.exp(self._weights[:, 5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        self._priority_mode = priority_mode

    def init_state(self, user_count: int, deck_size: int) -> FSRS6BatchState:
        s = torch.full(
            (user_count, deck_size),
            self._bounds.s_min,
            dtype=self.dtype,
            device=self.device,
        )
        d = torch.full(
            (user_count, deck_size),
            self._bounds.d_min,
            dtype=self.dtype,
            device=self.device,
        )
        return FSRS6BatchState(s=s, d=d)

    def review_priority(
        self, state: FSRS6BatchState, elapsed: torch.Tensor
    ) -> torch.Tensor:
        r_sched = _fsrs6_forgetting_curve(
            self._decay[:, None],
            self._factor[:, None],
            elapsed,
            state.s,
            self._bounds.s_min,
        )
        if self._priority_mode == "low_retrievability":
            return r_sched
        if self._priority_mode == "high_retrievability":
            return -r_sched
        if self._priority_mode == "low_difficulty":
            return state.d
        return -state.d

    def update_review(
        self,
        state: FSRS6BatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        prev_interval: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        weights = self._weights.index_select(0, user_idx)
        decay = self._decay.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention_factor = self._retention_factor.index_select(0, user_idx)
        mean_reversion_d = self._mean_reversion_d.index_select(0, user_idx)

        sched_s = state.s[user_idx, card_idx]
        sched_d = state.d[user_idx, card_idx]
        sched_r = _fsrs6_forgetting_curve(
            decay,
            factor,
            elapsed,
            sched_s,
            self._bounds.s_min,
        )
        sched_short = elapsed < 1.0
        sched_success = rating > 1

        sched_new_s = sched_s
        sched_new_s = torch.where(
            sched_short,
            _fsrs6_stability_short_term(weights, sched_s, rating),
            sched_new_s,
        )
        sched_new_s = torch.where(
            ~sched_short & sched_success,
            _fsrs6_stability_after_success(weights, sched_s, sched_r, sched_d, rating),
            sched_new_s,
        )
        sched_new_s = torch.where(
            ~sched_short & ~sched_success,
            _fsrs6_stability_after_failure(weights, sched_s, sched_r, sched_d),
            sched_new_s,
        )
        sched_new_d = _fsrs6_next_d(
            weights,
            sched_d,
            rating,
            mean_reversion_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[user_idx, card_idx] = torch.clamp(
            sched_new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = torch.clamp(
            sched_new_d, self._bounds.d_min, self._bounds.d_max
        )
        return torch.clamp(
            state.s[user_idx, card_idx] / factor * retention_factor, min=1.0
        )

    def update_learn(
        self,
        state: FSRS6BatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        weights = self._weights.index_select(0, user_idx)
        decay = self._decay.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention_factor = self._retention_factor.index_select(0, user_idx)
        s_init, d_init = _fsrs6_init_state(
            weights, rating, self._bounds.d_min, self._bounds.d_max
        )
        state.s[user_idx, card_idx] = torch.clamp(
            s_init, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = torch.clamp(
            d_init, self._bounds.d_min, self._bounds.d_max
        )
        return torch.clamp(
            state.s[user_idx, card_idx] / factor * retention_factor, min=1.0
        )


@dataclass
class AnkiBatchState:
    ease: torch.Tensor


class AnkiSM2BatchSchedulerOps:
    def __init__(
        self,
        *,
        graduating_interval: float,
        easy_interval: float,
        easy_bonus: float,
        hard_interval_factor: float,
        ease_start: float,
        ease_min: float,
        ease_max: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self._graduating_interval = float(graduating_interval)
        self._easy_interval = float(easy_interval)
        self._easy_bonus = float(easy_bonus)
        self._hard_interval_factor = float(hard_interval_factor)
        self._ease_start = float(ease_start)
        self._ease_min = float(ease_min)
        self._ease_max = float(ease_max)

    def init_state(self, user_count: int, deck_size: int) -> AnkiBatchState:
        ease = torch.full(
            (user_count, deck_size),
            self._ease_start,
            device=self.device,
            dtype=self.dtype,
        )
        return AnkiBatchState(ease=ease)

    def review_priority(
        self, state: AnkiBatchState, elapsed: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(elapsed, device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: AnkiBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        prev_interval: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[user_idx, card_idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
        )
        state.ease[user_idx, card_idx] = new_ease
        return intervals

    def update_learn(
        self,
        state: AnkiBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        prev_interval = torch.zeros_like(rating, dtype=self.dtype)
        elapsed = torch.zeros_like(prev_interval)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[user_idx, card_idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
        )
        state.ease[user_idx, card_idx] = new_ease
        return intervals


@dataclass
class MemriseBatchState:
    sequence: torch.Tensor


class MemriseBatchSchedulerOps:
    def __init__(
        self, scheduler: MemriseScheduler, *, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.device = device
        self.dtype = dtype
        self._sequence = torch.tensor(scheduler.sequence, device=device, dtype=dtype)
        self._seq_len = int(self._sequence.numel())

    def init_state(self, user_count: int, deck_size: int) -> MemriseBatchState:
        return MemriseBatchState(sequence=self._sequence)

    def review_priority(
        self, state: MemriseBatchState, elapsed: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(elapsed, device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: MemriseBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        prev_interval: torch.Tensor,
    ) -> torch.Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        if self._seq_len == 0:
            intervals = torch.ones_like(prev_interval, dtype=self.dtype)
        else:
            dist = torch.abs(prev_interval.unsqueeze(1) - self._sequence.unsqueeze(0))
            closest = torch.argmin(dist, dim=1)
            next_idx = torch.clamp(closest + 1, max=self._seq_len - 1)
            intervals = self._sequence[next_idx]
        fail = rating == 1
        is_new = prev_interval == 0.0
        return torch.where(
            is_new | fail,
            torch.ones_like(intervals, dtype=self.dtype),
            intervals,
        )

    def update_learn(
        self,
        state: MemriseBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(rating.shape[0], device=self.device, dtype=self.dtype)


def _fsrs6_forgetting_curve(
    decay: torch.Tensor,
    factor: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    s_min: float,
) -> torch.Tensor:
    return torch.pow(1.0 + factor * t / torch.clamp(s, min=s_min), decay)


def _fsrs6_init_state(
    weights: torch.Tensor,
    rating: torch.Tensor,
    d_min: float,
    d_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rating_f = rating.to(dtype=weights.dtype)
    idx = torch.clamp(rating - 1, min=0, max=3).unsqueeze(-1)
    s = torch.gather(weights[:, :4], 1, idx).squeeze(1)
    d = weights[:, 4] - torch.exp(weights[:, 5] * (rating_f - 1.0)) + 1.0
    d = torch.clamp(d, d_min, d_max)
    return s, d


def _fsrs6_next_d(
    weights: torch.Tensor,
    d: torch.Tensor,
    rating: torch.Tensor,
    init_d: torch.Tensor,
    d_min: float,
    d_max: float,
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    delta_d = -weights[:, 6] * (rating_f - 3.0)
    new_d = d + delta_d * (10.0 - d) / 9.0
    new_d = weights[:, 7] * init_d + (1.0 - weights[:, 7]) * new_d
    return torch.clamp(new_d, d_min, d_max)


def _fsrs6_stability_short_term(
    weights: torch.Tensor, s: torch.Tensor, rating: torch.Tensor
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    sinc = torch.exp(weights[:, 17] * (rating_f - 3.0 + weights[:, 18])) * torch.pow(
        s, -weights[:, 19]
    )
    safe = torch.maximum(sinc, torch.tensor(1.0, device=s.device, dtype=s.dtype))
    scale = torch.where(rating >= 3, safe, sinc)
    return s * scale


def _fsrs6_stability_after_success(
    weights: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    rating: torch.Tensor,
) -> torch.Tensor:
    hard_penalty = torch.where(rating == 2, weights[:, 15], 1.0)
    easy_bonus = torch.where(rating == 4, weights[:, 16], 1.0)
    inc = (
        torch.exp(weights[:, 8])
        * (11.0 - d)
        * torch.pow(s, -weights[:, 9])
        * (torch.exp((1.0 - r) * weights[:, 10]) - 1.0)
    )
    return s * (1.0 + inc * hard_penalty * easy_bonus)


def _fsrs6_stability_after_failure(
    weights: torch.Tensor, s: torch.Tensor, r: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    new_s = (
        weights[:, 11]
        * torch.pow(d, -weights[:, 12])
        * (torch.pow(s + 1.0, weights[:, 13]) - 1.0)
        * torch.exp((1.0 - r) * weights[:, 14])
    )
    new_min = s / torch.exp(weights[:, 17] * weights[:, 18])
    return torch.minimum(new_s, new_min)


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
    progress: bool = False,
    progress_label: Optional[str] = None,
) -> list[SimulationStats]:
    config = VectorizedConfig(
        device=torch.device(device) if device is not None else None,
        dtype=dtype,
    )
    torch_device = config.device or env_ops.device
    env_dtype = config.dtype or env_ops.dtype
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(seed)

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
        for day in range(days):
            if progress_bar is not None:
                progress_bar.update(1)
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
                cum = success_weights_sel.cumsum(dim=1)
                success_rand = torch.rand(
                    r_due.shape, device=torch_device, generator=gen
                )
                success_sample = (success_rand[:, None] > cum).sum(dim=1) + 2
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
                interval_days = torch.clamp(
                    torch.floor(intervals_next + 0.5), min=1.0
                ).to(torch.int64)
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

                cum = first_rating_prob.cumsum(dim=1)
                rand = torch.rand(
                    (user_count, max_candidate), device=torch_device, generator=gen
                )
                rating = (rand.unsqueeze(-1) > cum[:, None, :]).sum(dim=-1) + 1
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
                interval_days = torch.clamp(
                    torch.floor(intervals_next + 0.5), min=1.0
                ).to(torch.int64)
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

    daily_retention = torch.zeros_like(daily_cost)
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
