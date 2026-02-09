from __future__ import annotations

import logging
from pathlib import Path

import torch

from simulator.benchmark_loader import load_benchmark_weights
from simulator.batched_sweep.utils import format_id_list
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.models.lstm import _resolve_benchmark_weights


def resolve_lstm_paths(
    user_ids: list[int], benchmark_root: Path, *, short_term: bool
) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    kept: list[int] = []
    missing: list[int] = []
    for user_id in user_ids:
        path = _resolve_benchmark_weights(
            user_id, benchmark_root, short_term=short_term
        )
        if path is None:
            missing.append(user_id)
            continue
        paths.append(path)
        kept.append(user_id)
    if missing:
        logging.warning(
            "Skipping %d users missing LSTM weights: %s",
            len(missing),
            format_id_list(missing),
        )
    return paths, kept


def load_fsrs6_weights(
    *,
    repo_root: Path,
    user_ids: list[int],
    benchmark_root: Path,
    benchmark_partition: str | None,
    overrides: dict[str, str] | None,
    short_term: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[int]]:
    partition_key = benchmark_partition or "0"
    weights: list[torch.Tensor] = []
    kept: list[int] = []
    missing: list[int] = []
    for user_id in user_ids:
        try:
            params = load_benchmark_weights(
                repo_root=repo_root,
                benchmark_root=benchmark_root,
                environment="fsrs6",
                user_id=user_id,
                partition_key=partition_key,
                overrides=overrides,
                short_term=short_term,
            )
        except ValueError:
            missing.append(user_id)
            continue
        weights.append(torch.tensor(params, dtype=torch.float32))
        kept.append(user_id)
    if missing:
        logging.warning(
            "Skipping %d users missing FSRS-6 weights: %s",
            len(missing),
            format_id_list(missing),
        )
    if not weights:
        return None, []
    return torch.stack(weights, dim=0).to(device), kept


def build_default_fsrs6_weights(
    *, user_ids: list[int], device: torch.device
) -> torch.Tensor:
    tensor = torch.tensor(DEFAULT_FSRS6_WEIGHTS, dtype=torch.float32, device=device)
    return tensor.repeat(len(user_ids), 1)


def load_fsrs3_weights(
    *,
    repo_root: Path,
    user_ids: list[int],
    benchmark_root: Path,
    benchmark_partition: str | None,
    overrides: dict[str, str] | None,
    short_term: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, list[int]]:
    partition_key = benchmark_partition or "0"
    weights: list[torch.Tensor] = []
    kept: list[int] = []
    missing: list[int] = []
    invalid: list[int] = []
    for user_id in user_ids:
        try:
            params = load_benchmark_weights(
                repo_root=repo_root,
                benchmark_root=benchmark_root,
                environment="fsrs3",
                user_id=user_id,
                partition_key=partition_key,
                overrides=overrides,
                short_term=short_term,
            )
        except ValueError:
            missing.append(user_id)
            continue
        if len(params) != 13:
            invalid.append(user_id)
            continue
        weights.append(torch.tensor(params, dtype=torch.float32))
        kept.append(user_id)
    if missing:
        logging.warning(
            "Skipping %d users missing FSRS-3 weights: %s",
            len(missing),
            format_id_list(missing),
        )
    if invalid:
        logging.warning(
            "Skipping %d users with invalid FSRS-3 weights: %s",
            len(invalid),
            format_id_list(invalid),
        )
    if not weights:
        return None, []
    return torch.stack(weights, dim=0).to(device), kept
