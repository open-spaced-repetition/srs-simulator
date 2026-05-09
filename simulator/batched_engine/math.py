from __future__ import annotations

import math

import torch


def clamp(values: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return torch.clamp(values, min=min_value, max=max_value)


def forgetting_curve(
    decay: torch.Tensor,
    factor: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    s_min: float,
) -> torch.Tensor:
    return torch.pow(1.0 + factor * t / torch.clamp(s, min=s_min), decay)


def init_state(
    weights: torch.Tensor,
    rating: torch.Tensor,
    d_min: float,
    d_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rating_f = rating.to(dtype=weights.dtype)
    s = weights[rating - 1]
    d = weights[4] - torch.exp(weights[5] * (rating_f - 1.0)) + 1.0
    d = clamp(d, d_min, d_max)
    return s, d


def next_d(
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
    return clamp(new_d, d_min, d_max)


def stability_short_term(
    weights: torch.Tensor, s: torch.Tensor, rating: torch.Tensor
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    sinc = torch.exp(weights[17] * (rating_f - 3.0 + weights[18])) * torch.pow(
        s, -weights[19]
    )
    safe = torch.maximum(sinc, torch.tensor(1.0, device=s.device, dtype=s.dtype))
    scale = torch.where(rating >= 3, safe, sinc)
    return s * scale


def stability_after_success(
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


def stability_after_failure(
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


def fsrs3_forgetting_curve(
    t: torch.Tensor, s: torch.Tensor, s_min: float
) -> torch.Tensor:
    base = torch.tensor(0.9, device=s.device, dtype=s.dtype)
    return torch.pow(base, t / torch.clamp(s, min=s_min))


def fsrs3_init_state(
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
    s = clamp(s, s_min, s_max)
    d = clamp(d, d_min, d_max)
    return s, d


def fsrs3_stability_after_success(
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


def fsrs3_stability_after_failure(
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


def sspmmc_s2i(
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


def sspmmc_d2i(
    difficulty: torch.Tensor, d_min: float, d_max: float, d_size: int
) -> torch.Tensor:
    ratio = (difficulty - d_min) / (d_max - d_min)
    idx = torch.floor(ratio * float(d_size))
    return torch.clamp(idx, 0.0, float(d_size - 1)).to(torch.int64)
