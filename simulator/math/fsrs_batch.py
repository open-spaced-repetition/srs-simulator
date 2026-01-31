from __future__ import annotations

import torch


def fsrs6_forgetting_curve(
    decay: torch.Tensor,
    factor: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    s_min: float,
) -> torch.Tensor:
    return torch.pow(1.0 + factor * t / torch.clamp(s, min=s_min), decay)


def fsrs6_init_state(
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


def fsrs6_next_d(
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


def fsrs6_stability_short_term(
    weights: torch.Tensor, s: torch.Tensor, rating: torch.Tensor
) -> torch.Tensor:
    rating_f = rating.to(dtype=weights.dtype)
    sinc = torch.exp(weights[:, 17] * (rating_f - 3.0 + weights[:, 18])) * torch.pow(
        s, -weights[:, 19]
    )
    safe = torch.maximum(sinc, torch.tensor(1.0, device=s.device, dtype=s.dtype))
    scale = torch.where(rating >= 3, safe, sinc)
    return s * scale


def fsrs6_stability_after_success(
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


def fsrs6_stability_after_failure(
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
