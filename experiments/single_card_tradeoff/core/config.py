from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.retention_sweep.cli_utils import add_benchmark_args
from experiments.single_card_tradeoff.oracles.dp_cache import (
    add_oracle_dp_cache_args,
    OracleDPCacheConfig,
    oracle_dp_cache_config_from_args,
    set_default_oracle_dp_cache_config,
)
from simulator.benchmark_loader import load_benchmark_weights, parse_result_overrides
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.button_usage import load_button_usage_config, normalize_button_usage
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS

SUPPORTED_SINGLE_CARD_ENVS = {"fsrs6_default", "fsrs6"}


@dataclass(frozen=True)
class SingleCardFSRS6Config:
    environment: str
    fsrs_weights: tuple[float, ...]
    first_rating_prob: tuple[float, ...]
    review_rating_prob: tuple[float, ...]
    learning_costs: tuple[float, ...]
    review_costs: tuple[float, ...]
    user_id: int
    benchmark_result: str | None
    benchmark_partition: str
    srs_benchmark_root: str | None
    button_usage: str | None

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "fsrs_weights": list(self.fsrs_weights),
            "first_rating_prob": list(self.first_rating_prob),
            "review_rating_prob": list(self.review_rating_prob),
            "learning_costs": list(self.learning_costs),
            "review_costs": list(self.review_costs),
            "user_id": self.user_id,
            "benchmark_result": self.benchmark_result,
            "benchmark_partition": self.benchmark_partition,
            "srs_benchmark_root": self.srs_benchmark_root,
            "button_usage": self.button_usage,
        }


@dataclass(frozen=True)
class SingleCardRuntimeContext:
    torch_device: str | None
    dp_cache_config: OracleDPCacheConfig
    output_dir: Path | None
    repo_root: Path


def fsrs_config_kwargs(
    fsrs_config: SingleCardFSRS6Config | None,
) -> dict[str, object]:
    if fsrs_config is None:
        return {}
    return {
        "fsrs_weights": fsrs_config.fsrs_weights,
        "first_rating_prob": fsrs_config.first_rating_prob,
        "review_rating_prob": fsrs_config.review_rating_prob,
        "learning_costs": fsrs_config.learning_costs,
        "review_costs": fsrs_config.review_costs,
    }


def add_single_card_fsrs6_config_args(
    parser: argparse.ArgumentParser,
    *,
    include_env: bool = True,
) -> None:
    if include_env:
        parser.add_argument(
            "--env",
            choices=sorted(SUPPORTED_SINGLE_CARD_ENVS),
            default="fsrs6_default",
            help=(
                "FSRS6 environment parameters for single-card rollouts. "
                "'fsrs6' loads --user-id weights from srs-benchmark; "
                "'fsrs6_default' uses built-in defaults."
            ),
        )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="User ID for --env fsrs6 weights and optional --button-usage costs.",
    )
    add_benchmark_args(parser)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=None,
        help=(
            "Optional Anki button usage JSONL. When set, first/review rating "
            "probabilities and learning/review costs are loaded for --user-id."
        ),
    )
    add_oracle_dp_cache_args(parser)


def single_card_runtime_context_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> SingleCardRuntimeContext:
    return SingleCardRuntimeContext(
        torch_device=getattr(args, "torch_device", None),
        dp_cache_config=oracle_dp_cache_config_from_args(args),
        output_dir=output_dir,
        repo_root=repo_root or Path(__file__).resolve().parents[3],
    )


def configure_oracle_dp_cache_from_args(
    args: argparse.Namespace,
) -> OracleDPCacheConfig:
    config = oracle_dp_cache_config_from_args(args)
    set_default_oracle_dp_cache_config(config)
    return config


def _coerce_tuple(
    values: Sequence[float],
    *,
    expected_len: int,
    label: str,
) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != expected_len:
        raise ValueError(f"{label} must contain {expected_len} values.")
    return parsed


def load_single_card_fsrs6_config(
    args: argparse.Namespace,
    *,
    environment: str | None = None,
    repo_root: Path | None = None,
) -> SingleCardFSRS6Config:
    env_name = environment or str(getattr(args, "env", "fsrs6_default"))
    if env_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise ValueError(
            f"Single-card FSRS6 config supports only "
            f"{', '.join(sorted(SUPPORTED_SINGLE_CARD_ENVS))}; got '{env_name}'."
        )
    user_id = int(getattr(args, "user_id", None) or 1)
    benchmark_result = getattr(args, "benchmark_result", None)
    benchmark_partition = str(getattr(args, "benchmark_partition", "0"))
    benchmark_root = getattr(args, "srs_benchmark_root", None)
    root = repo_root or Path(__file__).resolve().parents[3]

    if env_name == "fsrs6":
        weights = load_benchmark_weights(
            repo_root=root,
            benchmark_root=benchmark_root,
            environment="fsrs6",
            user_id=user_id,
            partition_key=benchmark_partition,
            overrides=parse_result_overrides(benchmark_result),
            short_term=False,
        )
        fsrs_weights = _coerce_tuple(
            weights,
            expected_len=21,
            label="FSRS6 benchmark weights",
        )
    else:
        fsrs_weights = tuple(float(value) for value in DEFAULT_FSRS6_WEIGHTS)

    button_usage_path = getattr(args, "button_usage", None)
    button_usage_config = (
        load_button_usage_config(button_usage_path, user_id)
        if button_usage_path is not None
        else None
    )
    usage = normalize_button_usage(button_usage_config)
    return SingleCardFSRS6Config(
        environment=env_name,
        fsrs_weights=fsrs_weights,
        first_rating_prob=_coerce_tuple(
            usage.get("first_rating_prob", DEFAULT_FIRST_RATING_PROB),
            expected_len=4,
            label="first_rating_prob",
        ),
        review_rating_prob=_coerce_tuple(
            usage.get("review_rating_prob", DEFAULT_REVIEW_RATING_PROB),
            expected_len=3,
            label="review_rating_prob",
        ),
        learning_costs=_coerce_tuple(
            usage.get("learn_costs", DEFAULT_STATE_RATING_COSTS.learning),
            expected_len=4,
            label="learning costs",
        ),
        review_costs=_coerce_tuple(
            usage.get("review_costs", DEFAULT_STATE_RATING_COSTS.review),
            expected_len=4,
            label="review costs",
        ),
        user_id=user_id,
        benchmark_result=benchmark_result,
        benchmark_partition=benchmark_partition,
        srs_benchmark_root=str(benchmark_root) if benchmark_root is not None else None,
        button_usage=str(button_usage_path) if button_usage_path is not None else None,
    )
