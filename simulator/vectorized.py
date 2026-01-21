from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from simulator.behavior import StochasticBehavior
from simulator.core import (
    Scheduler,
    SimulationStats,
    new_first_priority,
    review_first_priority,
)
from simulator.cost import StatefulCostModel
from simulator.models.fsrs import FSRS6Model
from simulator.models.lstm import EPS as LSTM_EPS, LSTMModel
from simulator.schedulers import (
    AnkiSM2Scheduler,
    DASHScheduler,
    FixedIntervalScheduler,
    FSRS3Scheduler,
    FSRS6Scheduler,
    HLRScheduler,
    MemriseScheduler,
    SSPMMCScheduler,
)


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


def _fsrs3_forgetting_curve(
    t: torch.Tensor, s: torch.Tensor, s_min: float
) -> torch.Tensor:
    base = torch.tensor(0.9, device=s.device, dtype=s.dtype)
    return torch.pow(base, t / torch.clamp(s, min=s_min))


def _fsrs3_init_state(
    weights: torch.Tensor,
    rating: torch.Tensor,
    s_min: float,
    s_max: float,
    d_min: float,
    d_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rating_f = rating.to(dtype=weights.dtype)
    s = weights[0] + weights[1] * (rating_f - 1.0)
    d = weights[2] + weights[3] * (rating_f - 3.0)
    s = _clamp(s, s_min, s_max)
    d = _clamp(d, d_min, d_max)
    return s, d


def _fsrs3_stability_after_success(
    weights: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    inc = (
        torch.exp(weights[6])
        * (11.0 - d)
        * torch.pow(s, weights[7])
        * (torch.exp((1.0 - r) * weights[8]) - 1.0)
    )
    return s * (1.0 + inc)


def _fsrs3_stability_after_failure(
    weights: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    return (
        weights[9]
        * torch.pow(d, weights[10])
        * torch.pow(s, weights[11])
        * torch.exp((1.0 - r) * weights[12])
    )


def _anki_next_interval(
    prev_interval: torch.Tensor,
    rating: torch.Tensor,
    ease: torch.Tensor,
    *,
    graduating_interval: float,
    easy_interval: float,
    easy_bonus: float,
    hard_interval_factor: float,
    ease_min: float,
    ease_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_ease = ease.clone()
    new_ease = torch.where(rating == 1, new_ease - 0.2, new_ease)
    new_ease = torch.where(rating == 2, new_ease - 0.15, new_ease)
    new_ease = torch.where(rating == 4, new_ease + 0.15, new_ease)
    new_ease = torch.clamp(new_ease, min=ease_min, max=ease_max)

    is_new_card = prev_interval == 0.0
    new_card_interval = torch.where(
        rating < 4,
        torch.tensor(
            graduating_interval,
            device=prev_interval.device,
            dtype=prev_interval.dtype,
        ),
        torch.tensor(
            easy_interval, device=prev_interval.device, dtype=prev_interval.dtype
        ),
    )

    elapsed = prev_interval
    existing_interval = torch.where(rating == 1, prev_interval * 0.0, prev_interval)
    hard_interval = torch.maximum(
        elapsed * hard_interval_factor, prev_interval * hard_interval_factor / 2.0
    )
    existing_interval = torch.where(rating == 2, hard_interval, existing_interval)
    easy_interval_val = torch.maximum(elapsed * new_ease, prev_interval) * easy_bonus
    existing_interval = torch.where(rating == 4, easy_interval_val, existing_interval)
    normal_interval = torch.maximum(elapsed * new_ease, prev_interval)
    existing_interval = torch.where(
        (rating != 1) & (rating != 2) & (rating != 4),
        normal_interval,
        existing_interval,
    )

    interval = torch.where(is_new_card, new_card_interval, existing_interval)
    interval = torch.clamp(interval, min=1.0)
    return interval, new_ease


def _sspmmc_s2i(
    stability: torch.Tensor,
    s_min: float,
    s_mid: float,
    s_state_small_len: int,
    log_s_min: float,
    short_step: float,
    long_step: float,
    s_last: torch.Tensor,
    s_grid_size: int,
) -> torch.Tensor:
    stability = torch.clamp(stability, min=s_min)
    if s_state_small_len <= 0:
        return torch.zeros_like(stability, dtype=torch.int64)
    small_mask = stability <= s_mid
    idx_small = torch.ceil((torch.log(stability) - log_s_min) / short_step)
    idx_small = torch.clamp(idx_small, 0.0, float(s_state_small_len - 1))
    large_len = s_grid_size - s_state_small_len
    if large_len <= 0:
        idx_large = torch.full_like(idx_small, float(s_state_small_len - 1))
    else:
        offset = torch.ceil((stability - s_last - long_step) / long_step)
        offset = torch.clamp(offset, 0.0, float(large_len - 1))
        idx_large = float(s_state_small_len) + offset
    idx = torch.where(small_mask, idx_small, idx_large)
    return torch.clamp(idx, 0.0, float(s_grid_size - 1)).to(torch.int64)


def _sspmmc_d2i(
    difficulty: torch.Tensor, d_min: float, d_max: float, d_size: int
) -> torch.Tensor:
    ratio = (difficulty - d_min) / (d_max - d_min)
    idx = torch.floor(ratio * float(d_size))
    return torch.clamp(idx, 0.0, float(d_size - 1)).to(torch.int64)


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

                intervals_next = None
                if scheduler_kind in {"fsrs6", "sspmmc"}:
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
                    if scheduler_kind == "fsrs6":
                        intervals_next = torch.clamp(
                            sched_s[exec_idx] / sched_factor * sched_retention_factor,
                            min=1.0,
                        )
                    else:
                        s_idx = _sspmmc_s2i(
                            sched_s[exec_idx],
                            policy_s_min,
                            policy_s_mid,
                            policy_s_state_small_len,
                            policy_log_s_min,
                            policy_short_step,
                            policy_long_step,
                            policy_s_last,
                            policy_s_grid_size,
                        )
                        d_idx = _sspmmc_d2i(
                            sched_d[exec_idx],
                            policy_d_min,
                            policy_d_max,
                            policy_d_size,
                        )
                        desired = policy_retention[d_idx, s_idx]
                        graduated = (s_idx >= policy_s_grid_size - 1) | (
                            sched_s[exec_idx] >= policy_s_max
                        )
                        intervals_next = torch.where(
                            graduated,
                            torch.full_like(desired, policy_retire_interval),
                            sched_s[exec_idx]
                            / sched_factor
                            * (torch.pow(desired, 1.0 / sched_decay) - 1.0),
                        )
                        intervals_next = torch.clamp(intervals_next, min=1.0)
                elif scheduler_kind == "fsrs3":
                    sched_exec_s = sched_s[exec_idx]
                    sched_exec_d = sched_d[exec_idx]
                    sched_r = _fsrs3_forgetting_curve(
                        exec_elapsed, sched_exec_s, sched_bounds.s_min
                    )
                    sched_success = exec_rating > 1
                    current_d = sched_exec_d + sched_w[4] * (
                        exec_rating.to(env_dtype) - 3.0
                    )
                    sched_new_d = 0.5 * sched_w[2] + 0.5 * current_d
                    sched_new_d = _clamp(
                        sched_new_d, sched_bounds.d_min, sched_bounds.d_max
                    )
                    sched_new_s = torch.where(
                        sched_success,
                        _fsrs3_stability_after_success(
                            sched_w, sched_exec_s, sched_new_d, sched_r
                        ),
                        _fsrs3_stability_after_failure(
                            sched_w, sched_exec_s, sched_new_d, sched_r
                        ),
                    )
                    sched_s[exec_idx] = _clamp(
                        sched_new_s, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = sched_new_d
                    intervals_next = torch.clamp(
                        sched_s[exec_idx] * fsrs3_log_desired / fsrs3_ln, min=1.0
                    )
                elif scheduler_kind == "hlr":
                    success = (exec_rating > 1).to(env_dtype)
                    hlr_right[exec_idx] = hlr_right[exec_idx] + success
                    hlr_wrong[exec_idx] = hlr_wrong[exec_idx] + (1.0 - success)
                    half = torch.pow(
                        torch.tensor(2.0, device=torch_device, dtype=env_dtype),
                        hlr_w[0] * hlr_right[exec_idx]
                        + hlr_w[1] * hlr_wrong[exec_idx]
                        + hlr_w[2],
                    )
                    intervals_next = torch.clamp(half * hlr_log_factor, min=1.0)
                elif scheduler_kind == "fixed":
                    intervals_next = torch.full_like(exec_elapsed, fixed_interval)
                elif scheduler_kind == "memrise":
                    prev_interval = intervals[exec_idx].to(env_dtype)
                    intervals_next = torch.full_like(prev_interval, 1.0)
                    needs_lookup = (prev_interval > 0) & (exec_rating > 1)
                    if needs_lookup.any():
                        prev_vals = prev_interval[needs_lookup]
                        diffs = torch.abs(
                            prev_vals.unsqueeze(-1) - memrise_seq.unsqueeze(0)
                        )
                        closest = torch.argmin(diffs, dim=-1)
                        next_idx = torch.clamp(closest + 1, max=memrise_len - 1)
                        intervals_next[needs_lookup] = memrise_seq[next_idx]
                elif scheduler_kind == "anki_sm2":
                    prev_interval = intervals[exec_idx].to(env_dtype)
                    ease = anki_ease[exec_idx]
                    intervals_next, new_ease = _anki_next_interval(
                        prev_interval,
                        exec_rating,
                        ease,
                        graduating_interval=anki_params["graduating_interval"],
                        easy_interval=anki_params["easy_interval"],
                        easy_bonus=anki_params["easy_bonus"],
                        hard_interval_factor=anki_params["hard_interval_factor"],
                        ease_min=anki_params["ease_min"],
                        ease_max=anki_params["ease_max"],
                    )
                    anki_ease[exec_idx] = new_ease
                else:
                    raise ValueError(
                        f"Unsupported scheduler '{scheduler_kind}' in vectorized LSTM engine."
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

                interval_days = None
                if scheduler_kind in {"fsrs6", "sspmmc"}:
                    sched_s_init, sched_d_init = _init_state(
                        sched_w, exec_rating, sched_bounds.d_min, sched_bounds.d_max
                    )
                    sched_s[exec_idx] = _clamp(
                        sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = _clamp(
                        sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                    )
                    if scheduler_kind == "fsrs6":
                        intervals_next = torch.clamp(
                            sched_s[exec_idx] / sched_factor * sched_retention_factor,
                            min=1.0,
                        )
                    else:
                        s_idx = _sspmmc_s2i(
                            sched_s[exec_idx],
                            policy_s_min,
                            policy_s_mid,
                            policy_s_state_small_len,
                            policy_log_s_min,
                            policy_short_step,
                            policy_long_step,
                            policy_s_last,
                            policy_s_grid_size,
                        )
                        d_idx = _sspmmc_d2i(
                            sched_d[exec_idx], policy_d_min, policy_d_max, policy_d_size
                        )
                        retention = policy_retention[d_idx, s_idx]
                        graduated = (s_idx >= policy_s_grid_size - 1) | (
                            sched_s[exec_idx] >= policy_s_max
                        )
                        retire_val = torch.tensor(
                            policy_retire_interval,
                            device=torch_device,
                            dtype=env_dtype,
                        )
                        intervals_next = torch.where(
                            graduated,
                            retire_val,
                            sched_s[exec_idx]
                            / sched_factor
                            * (torch.pow(retention, 1.0 / sched_decay) - 1.0),
                        )
                elif scheduler_kind == "fsrs3":
                    sched_s_init, sched_d_init = _fsrs3_init_state(
                        sched_w,
                        exec_rating,
                        sched_bounds.s_min,
                        sched_bounds.s_max,
                        sched_bounds.d_min,
                        sched_bounds.d_max,
                    )
                    sched_s[exec_idx] = _clamp(
                        sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = _clamp(
                        sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                    )
                    intervals_next = sched_s[exec_idx] * (fsrs3_log_desired / fsrs3_ln)
                elif scheduler_kind == "hlr":
                    success = exec_rating > 1
                    right = success.to(env_dtype)
                    wrong = (~success).to(env_dtype)
                    hlr_right[exec_idx] = right
                    hlr_wrong[exec_idx] = wrong
                    half = torch.pow(
                        torch.tensor(2.0, device=torch_device, dtype=env_dtype),
                        hlr_w[0] * right + hlr_w[1] * wrong + hlr_w[2],
                    )
                    intervals_next = half * hlr_log_factor
                elif scheduler_kind == "fixed":
                    interval_days = torch.full(
                        (count,),
                        fixed_interval_days,
                        dtype=torch.int64,
                        device=torch_device,
                    )
                elif scheduler_kind == "memrise":
                    intervals_next = torch.ones(
                        count, device=torch_device, dtype=env_dtype
                    )
                elif scheduler_kind == "anki_sm2":
                    prev_interval = torch.zeros(
                        count, device=torch_device, dtype=env_dtype
                    )
                    intervals_next, new_ease = _anki_next_interval(
                        prev_interval, exec_rating, anki_ease[exec_idx], **anki_params
                    )
                    anki_ease[exec_idx] = new_ease
                else:
                    raise RuntimeError(
                        f"Vectorized LSTM engine unsupported scheduler: {scheduler_kind}"
                    )

                if interval_days is None:
                    interval_days = torch.clamp(
                        torch.round(intervals_next), min=1.0
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
    scheduler: Scheduler,
    behavior: StochasticBehavior,
    cost_model: StatefulCostModel,
    seed: int = 0,
    device: Optional[str] = None,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SimulationStats:
    if not isinstance(environment, LSTMModel):
        raise ValueError("Vectorized engine requires LSTMModel as the environment.")
    scheduler_kind = None
    if isinstance(scheduler, FSRS6Scheduler):
        scheduler_kind = "fsrs6"
    elif isinstance(scheduler, FSRS3Scheduler):
        scheduler_kind = "fsrs3"
    elif isinstance(scheduler, HLRScheduler):
        scheduler_kind = "hlr"
    elif isinstance(scheduler, FixedIntervalScheduler):
        scheduler_kind = "fixed"
    elif isinstance(scheduler, MemriseScheduler):
        scheduler_kind = "memrise"
    elif isinstance(scheduler, AnkiSM2Scheduler):
        scheduler_kind = "anki_sm2"
    elif isinstance(scheduler, SSPMMCScheduler):
        scheduler_kind = "sspmmc"
    elif isinstance(scheduler, DASHScheduler):
        raise ValueError(
            "Vectorized LSTM engine does not support DASHScheduler; "
            "use the event-driven engine instead."
        )
    else:
        raise ValueError(
            "Vectorized LSTM engine requires a supported scheduler "
            "(FSRS6, FSRS3, HLR, fixed, Memrise, Anki SM-2, or SSPMMC)."
        )
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

    sched_w = None
    sched_bounds = None
    sched_decay = None
    sched_factor = None
    sched_retention_factor = None
    sched_init_d = None
    fsrs3_ln = None
    fsrs3_log_desired = None
    hlr_w = None
    hlr_log_factor = None
    fixed_interval = None
    fixed_interval_days = None
    memrise_seq = None
    memrise_len = None
    anki_params = None
    policy_retention = None
    policy_s_grid = None
    policy_s_grid_size = None
    policy_s_state_small_len = None
    policy_s_mid = None
    policy_s_min = None
    policy_s_max = None
    policy_log_s_min = None
    policy_short_step = None
    policy_long_step = None
    policy_s_last = None
    policy_d_min = None
    policy_d_max = None
    policy_d_size = None
    policy_retire_interval = None

    if scheduler_kind in {"fsrs6", "sspmmc"}:
        sched_w = torch.tensor(
            scheduler.params.weights, device=torch_device, dtype=env_dtype
        )
        sched_bounds = scheduler.params.bounds
        sched_decay = -sched_w[20]
        sched_factor = (
            torch.pow(
                torch.tensor(0.9, device=torch_device, dtype=env_dtype),
                1.0 / sched_decay,
            )
            - 1.0
        )
        if scheduler_kind == "fsrs6":
            sched_retention_factor = (
                torch.pow(
                    torch.tensor(
                        scheduler.desired_retention,
                        device=torch_device,
                        dtype=env_dtype,
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
        if scheduler_kind == "sspmmc":
            policy = scheduler.policy
            policy_retention = torch.tensor(
                policy.retention_matrix, device=torch_device, dtype=env_dtype
            )
            policy_s_grid = torch.tensor(
                policy.s_grid, device=torch_device, dtype=env_dtype
            )
            policy_s_grid_size = int(policy.s_grid.size)
            policy_s_state_small_len = int(policy.s_state_small_len)
            policy_s_mid = float(policy.s_mid)
            policy_s_min = float(policy.state_space["s_min"])
            policy_s_max = float(policy.state_space["s_max"])
            policy_log_s_min = math.log(policy_s_min)
            policy_short_step = float(policy.state_space["short_step"])
            policy_long_step = float(policy.state_space["long_step"])
            policy_d_min = float(policy.state_space["d_min"])
            policy_d_max = float(policy.state_space["d_max"])
            policy_d_size = int(
                min(
                    math.ceil(
                        (policy_d_max - policy_d_min)
                        / float(policy.state_space["d_eps"])
                        + 1
                    ),
                    int(policy.d_grid.size),
                )
            )
            policy_retire_interval = float(scheduler.retire_interval)
            if policy_s_state_small_len > 0:
                policy_s_last = policy_s_grid[policy_s_state_small_len - 1]
            else:
                policy_s_last = torch.tensor(
                    policy_s_min, device=torch_device, dtype=env_dtype
                )
    elif scheduler_kind == "fsrs3":
        sched_w = torch.tensor(
            scheduler.params.weights, device=torch_device, dtype=env_dtype
        )
        sched_bounds = scheduler.params.bounds
        fsrs3_ln = math.log(0.9)
        fsrs3_log_desired = math.log(scheduler.desired_retention)
    elif scheduler_kind == "hlr":
        hlr_w = torch.tensor(scheduler.w, device=torch_device, dtype=env_dtype)
        hlr_log_factor = math.log(scheduler.desired_retention) / math.log(0.5)
    elif scheduler_kind == "fixed":
        fixed_interval = float(scheduler.interval)
        fixed_interval_days = max(1, int(round(fixed_interval)))
    elif scheduler_kind == "memrise":
        memrise_seq = torch.tensor(
            scheduler.sequence, device=torch_device, dtype=env_dtype
        )
        memrise_len = int(memrise_seq.numel())
    elif scheduler_kind == "anki_sm2":
        anki_params = {
            "graduating_interval": float(scheduler.graduating_interval),
            "easy_interval": float(scheduler.easy_interval),
            "easy_bonus": float(scheduler.easy_bonus),
            "hard_interval_factor": float(scheduler.hard_interval_factor),
            "ease_start": float(scheduler.ease_start),
            "ease_min": float(scheduler.ease_min),
            "ease_max": float(scheduler.ease_max),
        }

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

    sched_s = None
    sched_d = None
    hlr_right = None
    hlr_wrong = None
    anki_ease = None
    if scheduler_kind in {"fsrs6", "sspmmc", "fsrs3"} and sched_bounds is not None:
        sched_s = torch.full(
            (deck_size,), sched_bounds.s_min, dtype=env_dtype, device=torch_device
        )
        sched_d = torch.full(
            (deck_size,), sched_bounds.d_min, dtype=env_dtype, device=torch_device
        )
    if scheduler_kind == "hlr":
        hlr_right = torch.zeros(deck_size, dtype=env_dtype, device=torch_device)
        hlr_wrong = torch.zeros(deck_size, dtype=env_dtype, device=torch_device)
    if scheduler_kind == "anki_sm2" and anki_params is not None:
        anki_ease = torch.full(
            (deck_size,),
            anki_params["ease_start"],
            dtype=env_dtype,
            device=torch_device,
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
        if lengths.numel() == 0:
            return
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return
        order = torch.argsort(lengths, descending=True)
        idx_sorted = idx[order]
        lengths_sorted = lengths[order]
        pack_batch = 2048
        for start in range(0, idx_sorted.numel(), pack_batch):
            end = min(start + pack_batch, idx_sorted.numel())
            batch_idx = idx_sorted[start:end]
            batch_lengths = lengths_sorted[start:end]
            if batch_lengths.numel() == 0:
                continue
            batch_max = int(batch_lengths.max().item())
            if batch_max <= 0:
                continue
            delays_group = event_delays[batch_idx, :batch_max]
            ratings_group = event_ratings[batch_idx, :batch_max]
            delay_scaled = torch.clamp(delays_group, min=0.0) * interval_scale
            seq_delay = delay_scaled.transpose(0, 1).unsqueeze(-1)
            rating_seq = ratings_group.transpose(0, 1).unsqueeze(-1).to(env_dtype)
            if use_duration_feature:
                duration_seq = duration_value.expand(seq_delay.shape)
                sequence = torch.cat([seq_delay, duration_seq, rating_seq], dim=-1)
            else:
                sequence = torch.cat([seq_delay, rating_seq], dim=-1)
            packed = pack_padded_sequence(
                sequence, batch_lengths.to("cpu"), enforce_sorted=True
            )
            w_last, s_last, d_last = environment.network.forward_packed_last(
                packed, batch_lengths
            )
            mem_w[batch_idx] = w_last
            mem_s[batch_idx] = s_last
            mem_d[batch_idx] = d_last
            has_curves[batch_idx] = True

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

                primary = torch.zeros_like(review_cost)
                if scheduler_kind in {"fsrs6", "sspmmc"}:
                    sched_elapsed = elapsed
                    r_sched = _forgetting_curve(
                        sched_decay,
                        sched_factor,
                        sched_elapsed,
                        sched_s[review_idx],
                        sched_bounds.s_min,
                    )
                    if scheduler_kind == "fsrs6":
                        if scheduler.priority_mode == "low_retrievability":
                            primary = r_sched
                        elif scheduler.priority_mode == "high_retrievability":
                            primary = -r_sched
                        elif scheduler.priority_mode == "low_difficulty":
                            primary = sched_d[review_idx]
                        else:
                            primary = -sched_d[review_idx]
                elif scheduler_kind == "fsrs3":
                    sched_elapsed = elapsed
                    r_sched = _fsrs3_forgetting_curve(
                        sched_elapsed, sched_s[review_idx], sched_bounds.s_min
                    )
                    primary = r_sched
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

                interval_days = None
                if scheduler_kind in {"fsrs6", "sspmmc"}:
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

                    if scheduler_kind == "fsrs6":
                        intervals_next = torch.clamp(
                            sched_s[exec_idx] / sched_factor * sched_retention_factor,
                            min=1.0,
                        )
                    else:
                        s_idx = _sspmmc_s2i(
                            sched_s[exec_idx],
                            policy_s_min,
                            policy_s_mid,
                            policy_s_state_small_len,
                            policy_log_s_min,
                            policy_short_step,
                            policy_long_step,
                            policy_s_last,
                            policy_s_grid_size,
                        )
                        d_idx = _sspmmc_d2i(
                            sched_d[exec_idx], policy_d_min, policy_d_max, policy_d_size
                        )
                        retention = policy_retention[d_idx, s_idx]
                        graduated = (s_idx >= policy_s_grid_size - 1) | (
                            sched_s[exec_idx] >= policy_s_max
                        )
                        retire_val = torch.tensor(
                            policy_retire_interval,
                            device=torch_device,
                            dtype=env_dtype,
                        )
                        intervals_next = torch.where(
                            graduated,
                            retire_val,
                            sched_s[exec_idx]
                            / sched_factor
                            * (torch.pow(retention, 1.0 / sched_decay) - 1.0),
                        )
                elif scheduler_kind == "fsrs3":
                    sched_exec_s = sched_s[exec_idx]
                    sched_exec_d = sched_d[exec_idx]
                    sched_r = _fsrs3_forgetting_curve(
                        exec_elapsed, sched_exec_s, sched_bounds.s_min
                    )
                    sched_success = exec_rating > 1
                    sched_new_s = torch.where(
                        sched_success,
                        _fsrs3_stability_after_success(
                            sched_w, sched_exec_s, sched_exec_d, sched_r
                        ),
                        _fsrs3_stability_after_failure(
                            sched_w, sched_exec_s, sched_exec_d, sched_r
                        ),
                    )
                    d_update = sched_exec_d + sched_w[4] * (
                        exec_rating.to(env_dtype) - 3.0
                    )
                    sched_new_d = 0.5 * sched_w[2] + 0.5 * d_update
                    sched_s[exec_idx] = _clamp(
                        sched_new_s, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = _clamp(
                        sched_new_d, sched_bounds.d_min, sched_bounds.d_max
                    )
                    intervals_next = sched_s[exec_idx] * (fsrs3_log_desired / fsrs3_ln)
                elif scheduler_kind == "hlr":
                    right = hlr_right[exec_idx]
                    wrong = hlr_wrong[exec_idx]
                    success = exec_rating > 1
                    right = right + success.to(env_dtype)
                    wrong = wrong + (~success).to(env_dtype)
                    hlr_right[exec_idx] = right
                    hlr_wrong[exec_idx] = wrong
                    half = torch.pow(
                        torch.tensor(2.0, device=torch_device, dtype=env_dtype),
                        hlr_w[0] * right + hlr_w[1] * wrong + hlr_w[2],
                    )
                    intervals_next = half * hlr_log_factor
                elif scheduler_kind == "fixed":
                    interval_days = torch.full(
                        (count,),
                        fixed_interval_days,
                        dtype=torch.int64,
                        device=torch_device,
                    )
                elif scheduler_kind == "memrise":
                    prev_interval = intervals[exec_idx].to(env_dtype)
                    fail = exec_rating == 1
                    is_new_card = prev_interval == 0.0
                    if memrise_len == 0:
                        intervals_next = torch.ones_like(prev_interval)
                    else:
                        dist = torch.abs(
                            prev_interval.unsqueeze(1) - memrise_seq.unsqueeze(0)
                        )
                        closest = torch.argmin(dist, dim=1)
                        next_idx = torch.clamp(closest + 1, max=memrise_len - 1)
                        intervals_next = memrise_seq[next_idx]
                    intervals_next = torch.where(
                        is_new_card | fail,
                        torch.ones_like(intervals_next),
                        intervals_next,
                    )
                elif scheduler_kind == "anki_sm2":
                    prev_interval = intervals[exec_idx].to(env_dtype)
                    intervals_next, new_ease = _anki_next_interval(
                        prev_interval, exec_rating, anki_ease[exec_idx], **anki_params
                    )
                    anki_ease[exec_idx] = new_ease
                else:
                    raise RuntimeError(
                        f"Vectorized LSTM engine unsupported scheduler: {scheduler_kind}"
                    )

                if interval_days is None:
                    interval_days = torch.clamp(
                        torch.round(intervals_next), min=1.0
                    ).to(torch.int64)
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

                interval_days = None
                if scheduler_kind in {"fsrs6", "sspmmc"}:
                    sched_s_init, sched_d_init = _init_state(
                        sched_w, exec_rating, sched_bounds.d_min, sched_bounds.d_max
                    )
                    sched_s[exec_idx] = _clamp(
                        sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = _clamp(
                        sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                    )
                    if scheduler_kind == "fsrs6":
                        intervals_next = torch.clamp(
                            sched_s[exec_idx] / sched_factor * sched_retention_factor,
                            min=1.0,
                        )
                    else:
                        s_idx = _sspmmc_s2i(
                            sched_s[exec_idx],
                            policy_s_min,
                            policy_s_mid,
                            policy_s_state_small_len,
                            policy_log_s_min,
                            policy_short_step,
                            policy_long_step,
                            policy_s_last,
                            policy_s_grid_size,
                        )
                        d_idx = _sspmmc_d2i(
                            sched_d[exec_idx], policy_d_min, policy_d_max, policy_d_size
                        )
                        retention = policy_retention[d_idx, s_idx]
                        graduated = (s_idx >= policy_s_grid_size - 1) | (
                            sched_s[exec_idx] >= policy_s_max
                        )
                        retire_val = torch.tensor(
                            policy_retire_interval,
                            device=torch_device,
                            dtype=env_dtype,
                        )
                        intervals_next = torch.where(
                            graduated,
                            retire_val,
                            sched_s[exec_idx]
                            / sched_factor
                            * (torch.pow(retention, 1.0 / sched_decay) - 1.0),
                        )
                elif scheduler_kind == "fsrs3":
                    sched_s_init, sched_d_init = _fsrs3_init_state(
                        sched_w,
                        exec_rating,
                        sched_bounds.s_min,
                        sched_bounds.s_max,
                        sched_bounds.d_min,
                        sched_bounds.d_max,
                    )
                    sched_s[exec_idx] = _clamp(
                        sched_s_init, sched_bounds.s_min, sched_bounds.s_max
                    )
                    sched_d[exec_idx] = _clamp(
                        sched_d_init, sched_bounds.d_min, sched_bounds.d_max
                    )
                    intervals_next = sched_s[exec_idx] * (fsrs3_log_desired / fsrs3_ln)
                elif scheduler_kind == "hlr":
                    success = exec_rating > 1
                    right = success.to(env_dtype)
                    wrong = (~success).to(env_dtype)
                    hlr_right[exec_idx] = right
                    hlr_wrong[exec_idx] = wrong
                    half = torch.pow(
                        torch.tensor(2.0, device=torch_device, dtype=env_dtype),
                        hlr_w[0] * right + hlr_w[1] * wrong + hlr_w[2],
                    )
                    intervals_next = half * hlr_log_factor
                elif scheduler_kind == "fixed":
                    interval_days = torch.full(
                        (count,),
                        fixed_interval_days,
                        dtype=torch.int64,
                        device=torch_device,
                    )
                elif scheduler_kind == "memrise":
                    intervals_next = torch.ones(
                        count, device=torch_device, dtype=env_dtype
                    )
                elif scheduler_kind == "anki_sm2":
                    prev_interval = torch.zeros(
                        count, device=torch_device, dtype=env_dtype
                    )
                    intervals_next, new_ease = _anki_next_interval(
                        prev_interval, exec_rating, anki_ease[exec_idx], **anki_params
                    )
                    anki_ease[exec_idx] = new_ease
                else:
                    raise RuntimeError(
                        f"Vectorized LSTM engine unsupported scheduler: {scheduler_kind}"
                    )

                if interval_days is None:
                    interval_days = torch.clamp(
                        torch.round(intervals_next), min=1.0
                    ).to(torch.int64)
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
