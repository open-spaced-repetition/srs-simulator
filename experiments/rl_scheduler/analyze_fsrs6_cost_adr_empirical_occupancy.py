from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_engine.fuzz import round_intervals  # noqa: E402
from simulator.batched_engine.multiuser_engine import simulate_multiuser  # noqa: E402
from simulator.batched_sweep.behavior_cost import (  # noqa: E402
    build_behavior_cost,
    load_usage,
)
from simulator.batched_sweep.logging import BatchedSweepLogLane  # noqa: E402
from simulator.batched_sweep.runner import (  # noqa: E402
    _repeat_lstm_weights_for_lanes,
    _repeat_weights_for_lanes,
)
from simulator.batched_sweep.weights import (  # noqa: E402
    load_fsrs6_weights,
    resolve_lstm_paths,
)
from simulator.benchmark_loader import (  # noqa: E402
    parse_result_overrides,
    resolve_benchmark_root,
)
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH  # noqa: E402
from simulator.experiment_infra.gpu_monitor import (  # noqa: E402
    GpuMonitor,
    GpuMonitorSummary,
    disabled_monitor_summary,
)
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    FSRS6CostConditionedADRPolicy,
)
from simulator.math.fsrs import Bounds  # noqa: E402
from simulator.models.fsrs import FSRS6BatchEnvOps  # noqa: E402
from simulator.models.lstm_batch import (  # noqa: E402
    LSTMBatchedEnvOps,
    PackedLSTMWeights,
)
from simulator.schedulers.fsrs6_cost_conditioned_adr import (  # noqa: E402
    FSRS6CostConditionedADRBatchSchedulerOps,
    FSRS6CostConditionedADRBatchState,
)


DEFAULT_RUN_ROOT = Path(
    "artifacts/rl_scheduler/"
    "fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/"
    "fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128"
    "_pop16_gen20_v1_markov_off"
)
DEFAULT_OUT_DIR = Path(
    "artifacts/rl_scheduler/cost_adr_empirical_occupancy_analysis_2026_05_29"
)
DEFAULT_COST_WEIGHTS = (
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
    192.0,
    256.0,
    384.0,
    512.0,
    1024.0,
)
PHASES = ("learn", "review")
S_BINS = ("low_s", "mid_s", "high_s")
D_BINS = ("easy_d", "mid_d", "hard_d")
RETENTION_MIN = 0.30
RETENTION_MAX = 0.995
RETENTION_BINS = 695
LOG_INTERVAL_MIN = 0.0
LOG_INTERVAL_MAX = 9.0
INTERVAL_BINS = 1800
REGION_COUNT = len(S_BINS) * len(D_BINS)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    run_root: Path
    out_dir: Path
    users: tuple[int, ...]
    envs: tuple[str, ...]
    cost_weights: tuple[float, ...]
    days: int
    deck: int
    learn_limit: int
    review_limit: int
    cost_limit_minutes: float
    seed: int
    priority: str
    scheduler_priority: str
    benchmark_root: Path
    benchmark_partition: str
    benchmark_overrides: dict[str, str]
    button_usage: Path | None
    device: torch.device
    fsrs6_max_lanes: int
    lstm_max_lanes: int
    gpu_monitor: bool
    gpu_monitor_interval_seconds: float


@dataclass(slots=True)
class ChunkResult:
    env: str
    lane_records: list[dict[str, Any]]
    cost_phase_count: np.ndarray
    cost_phase_retention_sum: np.ndarray
    cost_phase_interval_sum: np.ndarray
    cost_phase_retention_hist: np.ndarray
    cost_phase_interval_hist: np.ndarray
    region_count: np.ndarray
    region_retention_sum: np.ndarray
    region_interval_sum: np.ndarray
    stats_rows: list[dict[str, Any]]
    max_interval_days: float
    max_raw_interval_days: float
    max_s: float
    max_d: float
    elapsed_seconds: float


class OccupancyCollector:
    def __init__(
        self,
        *,
        env: str,
        lanes: Sequence[BatchedSweepLogLane],
        cost_weights: Sequence[float],
        device: torch.device,
    ) -> None:
        self.env = env
        self.lanes = tuple(lanes)
        self.cost_weights = tuple(cost_weights)
        self.device = device
        self._cost_to_idx = {
            float(weight): idx for idx, weight in enumerate(cost_weights)
        }
        self._lane_cost_idx = torch.tensor(
            [
                self._cost_to_idx[
                    _require_float(
                        lane.fsrs6_cost_adr_goal_cost_weight,
                        "fsrs6_cost_adr_goal_cost_weight",
                    )
                ]
                for lane in lanes
            ],
            device=device,
            dtype=torch.long,
        )

        shape = (len(PHASES), len(cost_weights))
        self.count = torch.zeros(shape, device=device, dtype=torch.float64)
        self.retention_sum = torch.zeros_like(self.count)
        self.interval_sum = torch.zeros_like(self.count)
        self.retention_hist = torch.zeros(
            (*shape, RETENTION_BINS),
            device=device,
            dtype=torch.int64,
        )
        self.interval_hist = torch.zeros(
            (*shape, INTERVAL_BINS),
            device=device,
            dtype=torch.int64,
        )
        lane_shape = (len(PHASES), len(lanes))
        self.lane_count = torch.zeros(lane_shape, device=device, dtype=torch.float64)
        self.lane_retention_sum = torch.zeros_like(self.lane_count)
        self.lane_interval_sum = torch.zeros_like(self.lane_count)
        self.lane_retention_hist = torch.zeros(
            (*lane_shape, RETENTION_BINS),
            device=device,
            dtype=torch.int64,
        )
        self.lane_interval_hist = torch.zeros(
            (*lane_shape, INTERVAL_BINS),
            device=device,
            dtype=torch.int64,
        )
        region_shape = (len(PHASES), len(cost_weights), REGION_COUNT)
        self.region_count = torch.zeros(
            region_shape, device=device, dtype=torch.float64
        )
        self.region_retention_sum = torch.zeros_like(self.region_count)
        self.region_interval_sum = torch.zeros_like(self.region_count)
        self.max_interval_days = torch.tensor(0.0, device=device)
        self.max_raw_interval_days = torch.tensor(0.0, device=device)
        self.max_s = torch.tensor(0.0, device=device)
        self.max_d = torch.tensor(0.0, device=device)

    def collect(
        self,
        *,
        phase_idx: int,
        sched_ops: FSRS6CostConditionedADRBatchSchedulerOps,
        state: FSRS6CostConditionedADRBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        raw_interval: torch.Tensor,
    ) -> None:
        if user_idx.numel() == 0:
            return

        s = state.s[user_idx, card_idx]
        d = state.d[user_idx, card_idx]
        retention = sched_ops._retention_for_state(s, d, user_idx)
        interval = round_intervals(
            raw_interval.to(retention.dtype),
            minimum=1,
        ).to(retention.dtype)
        self.max_interval_days = torch.maximum(self.max_interval_days, interval.max())
        self.max_raw_interval_days = torch.maximum(
            self.max_raw_interval_days,
            raw_interval.max(),
        )
        self.max_s = torch.maximum(self.max_s, s.max())
        self.max_d = torch.maximum(self.max_d, d.max())

        lane_cost_idx = self._lane_cost_idx.index_select(0, user_idx)
        ones = torch.ones_like(retention, dtype=torch.float64)
        phase = torch.full_like(lane_cost_idx, phase_idx)

        flat_cost = phase * len(self.cost_weights) + lane_cost_idx
        self.count.view(-1).scatter_add_(0, flat_cost, ones)
        self.retention_sum.view(-1).scatter_add_(
            0,
            flat_cost,
            retention.to(torch.float64),
        )
        self.interval_sum.view(-1).scatter_add_(
            0,
            flat_cost,
            interval.to(torch.float64),
        )

        retention_bin = torch.floor(
            (retention.clamp(RETENTION_MIN, RETENTION_MAX) - RETENTION_MIN)
            / (RETENTION_MAX - RETENTION_MIN)
            * RETENTION_BINS
        ).to(torch.long)
        retention_bin = retention_bin.clamp(0, RETENTION_BINS - 1)
        log_interval = torch.log10(interval.clamp(min=1.0)).clamp(
            LOG_INTERVAL_MIN,
            LOG_INTERVAL_MAX,
        )
        interval_bin = torch.floor(
            (log_interval - LOG_INTERVAL_MIN)
            / (LOG_INTERVAL_MAX - LOG_INTERVAL_MIN)
            * INTERVAL_BINS
        ).to(torch.long)
        interval_bin = interval_bin.clamp(0, INTERVAL_BINS - 1)

        flat_retention = flat_cost * RETENTION_BINS + retention_bin
        retention_counts = torch.bincount(
            flat_retention,
            minlength=self.retention_hist.numel(),
        )
        self.retention_hist.view(-1).add_(
            retention_counts[: self.retention_hist.numel()]
        )
        flat_interval = flat_cost * INTERVAL_BINS + interval_bin
        interval_counts = torch.bincount(
            flat_interval,
            minlength=self.interval_hist.numel(),
        )
        self.interval_hist.view(-1).add_(interval_counts[: self.interval_hist.numel()])

        flat_lane = phase * len(self.lanes) + user_idx
        self.lane_count.view(-1).scatter_add_(0, flat_lane, ones)
        self.lane_retention_sum.view(-1).scatter_add_(
            0,
            flat_lane,
            retention.to(torch.float64),
        )
        self.lane_interval_sum.view(-1).scatter_add_(
            0,
            flat_lane,
            interval.to(torch.float64),
        )
        flat_lane_retention = flat_lane * RETENTION_BINS + retention_bin
        lane_retention_counts = torch.bincount(
            flat_lane_retention,
            minlength=self.lane_retention_hist.numel(),
        )
        self.lane_retention_hist.view(-1).add_(
            lane_retention_counts[: self.lane_retention_hist.numel()]
        )
        flat_lane_interval = flat_lane * INTERVAL_BINS + interval_bin
        lane_interval_counts = torch.bincount(
            flat_lane_interval,
            minlength=self.lane_interval_hist.numel(),
        )
        self.lane_interval_hist.view(-1).add_(
            lane_interval_counts[: self.lane_interval_hist.numel()]
        )

        bounds = sched_ops._bounds
        x_s = (
            torch.log(s.clamp(bounds.s_min, bounds.s_max)) - sched_ops._log_s_min
        ) / sched_ops._log_s_span
        x_d = (d.clamp(bounds.d_min, bounds.d_max) - bounds.d_min) / sched_ops._d_span
        s_bin = torch.where(x_s <= 0.25, 0, torch.where(x_s < 0.75, 1, 2)).to(
            torch.long
        )
        d_bin = torch.where(x_d <= 0.25, 0, torch.where(x_d < 0.75, 1, 2)).to(
            torch.long
        )
        region = s_bin * len(D_BINS) + d_bin
        flat_region = (
            phase * len(self.cost_weights) + lane_cost_idx
        ) * REGION_COUNT + region
        self.region_count.view(-1).scatter_add_(0, flat_region, ones)
        self.region_retention_sum.view(-1).scatter_add_(
            0,
            flat_region,
            retention.to(torch.float64),
        )
        self.region_interval_sum.view(-1).scatter_add_(
            0,
            flat_region,
            interval.to(torch.float64),
        )

    def finalize(
        self,
        *,
        stats: Sequence[Any],
        elapsed_seconds: float,
    ) -> ChunkResult:
        lane_records: list[dict[str, Any]] = []
        lane_count = self.lane_count.cpu().numpy()
        lane_retention_sum = self.lane_retention_sum.cpu().numpy()
        lane_interval_sum = self.lane_interval_sum.cpu().numpy()
        lane_retention_hist = self.lane_retention_hist.cpu().numpy()
        lane_interval_hist = self.lane_interval_hist.cpu().numpy()
        for lane_idx, lane in enumerate(self.lanes):
            for phase_idx, phase in enumerate(PHASES):
                count = int(lane_count[phase_idx, lane_idx])
                if count <= 0:
                    continue
                lane_records.append(
                    {
                        "environment": self.env,
                        "user_id": lane.user_id,
                        "cost_weight": _require_float(
                            lane.fsrs6_cost_adr_goal_cost_weight,
                            "fsrs6_cost_adr_goal_cost_weight",
                        ),
                        "phase": phase,
                        "events": count,
                        "mean_retention": float(
                            lane_retention_sum[phase_idx, lane_idx] / count
                        ),
                        "median_retention": hist_quantile(
                            lane_retention_hist[phase_idx, lane_idx],
                            RETENTION_MIN,
                            RETENTION_MAX,
                            0.50,
                        ),
                        "mean_interval_days": float(
                            lane_interval_sum[phase_idx, lane_idx] / count
                        ),
                        "median_interval_days": hist_quantile(
                            lane_interval_hist[phase_idx, lane_idx],
                            LOG_INTERVAL_MIN,
                            LOG_INTERVAL_MAX,
                            0.50,
                            log_scale=True,
                        ),
                        "q90_interval_days": hist_quantile(
                            lane_interval_hist[phase_idx, lane_idx],
                            LOG_INTERVAL_MIN,
                            LOG_INTERVAL_MAX,
                            0.90,
                            log_scale=True,
                        ),
                    }
                )

        stats_rows: list[dict[str, Any]] = []
        for lane, stat in zip(self.lanes, stats, strict=True):
            stats_rows.append(
                {
                    "environment": self.env,
                    "user_id": lane.user_id,
                    "cost_weight": _require_float(
                        lane.fsrs6_cost_adr_goal_cost_weight,
                        "fsrs6_cost_adr_goal_cost_weight",
                    ),
                    "total_reviews": int(stat.total_reviews),
                    "total_lapses": int(stat.total_lapses),
                    "total_cost": float(stat.total_cost),
                    "mean_daily_reviews": float(np.mean(stat.daily_reviews)),
                    "mean_daily_new": float(np.mean(stat.daily_new)),
                    "mean_daily_cost_minutes": float(np.mean(stat.daily_cost) / 60.0),
                    "mean_daily_memorized": float(np.mean(stat.daily_memorized)),
                }
            )

        return ChunkResult(
            env=self.env,
            lane_records=lane_records,
            cost_phase_count=self.count.cpu().numpy(),
            cost_phase_retention_sum=self.retention_sum.cpu().numpy(),
            cost_phase_interval_sum=self.interval_sum.cpu().numpy(),
            cost_phase_retention_hist=self.retention_hist.cpu().numpy(),
            cost_phase_interval_hist=self.interval_hist.cpu().numpy(),
            region_count=self.region_count.cpu().numpy(),
            region_retention_sum=self.region_retention_sum.cpu().numpy(),
            region_interval_sum=self.region_interval_sum.cpu().numpy(),
            stats_rows=stats_rows,
            max_interval_days=float(self.max_interval_days.cpu().item()),
            max_raw_interval_days=float(self.max_raw_interval_days.cpu().item()),
            max_s=float(self.max_s.cpu().item()),
            max_d=float(self.max_d.cpu().item()),
            elapsed_seconds=elapsed_seconds,
        )


def parse_users(raw: str) -> tuple[int, ...]:
    users: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid user range {token!r}.")
            users.extend(range(start, end + 1))
        else:
            users.append(int(token))
    result = tuple(dict.fromkeys(users))
    if not result:
        raise ValueError("--users must resolve at least one user.")
    return result


def parse_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def parse_envs(raw: str) -> tuple[str, ...]:
    envs = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"fsrs6", "lstm"}
    invalid = sorted(set(envs) - allowed)
    if invalid:
        raise ValueError(f"Unsupported env(s): {invalid}.")
    if not envs:
        raise ValueError("--envs must contain at least one environment.")
    return envs


def hist_quantile(
    hist: np.ndarray,
    lo: float,
    hi: float,
    quantile: float,
    *,
    log_scale: bool = False,
) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    target = quantile * (total - 1)
    cumulative = np.cumsum(hist, dtype=np.float64)
    idx = int(np.searchsorted(cumulative, target + 1, side="left"))
    idx = min(max(idx, 0), int(hist.shape[0]) - 1)
    center = lo + (idx + 0.5) * (hi - lo) / int(hist.shape[0])
    return float(10**center) if log_scale else float(center)


def policy_path(run_root: Path, user_id: int) -> Path:
    return (
        run_root / "train-overfit" / "train_outputs" / f"user_{user_id}" / "policy.json"
    )


def build_lanes(
    *,
    run_root: Path,
    users: Sequence[int],
    env: str,
    cost_weights: Sequence[float],
) -> list[BatchedSweepLogLane]:
    lanes: list[BatchedSweepLogLane] = []
    for user_id in users:
        for cost_weight in cost_weights:
            lanes.append(
                BatchedSweepLogLane(
                    user_id=user_id,
                    log_root=Path("."),
                    log_dir=Path("."),
                    environment=env,
                    scheduler_name="fsrs6_cost_adr",
                    scheduler_spec="fsrs6_cost_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_cost_adr_policy=policy_path(run_root, user_id),
                    fsrs6_cost_adr_goal_cost_weight=float(cost_weight),
                )
            )
    return lanes


def attach_collector(
    sched_ops: FSRS6CostConditionedADRBatchSchedulerOps,
    collector: OccupancyCollector,
) -> None:
    original_update_learn = sched_ops.update_learn
    original_update_review = sched_ops.update_review

    def update_learn(
        state: FSRS6CostConditionedADRBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        interval = original_update_learn(state, user_idx, card_idx, rating)
        collector.collect(
            phase_idx=0,
            sched_ops=sched_ops,
            state=state,
            user_idx=user_idx,
            card_idx=card_idx,
            raw_interval=interval,
        )
        return interval

    def update_review(
        state: FSRS6CostConditionedADRBatchState,
        user_idx: torch.Tensor,
        card_idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        prev_interval: torch.Tensor,
    ) -> torch.Tensor:
        interval = original_update_review(
            state,
            user_idx,
            card_idx,
            elapsed,
            rating,
            prev_interval,
        )
        collector.collect(
            phase_idx=1,
            sched_ops=sched_ops,
            state=state,
            user_idx=user_idx,
            card_idx=card_idx,
            raw_interval=interval,
        )
        return interval

    setattr(sched_ops, "update_learn", update_learn)
    setattr(sched_ops, "update_review", update_review)


def build_scheduler_ops(
    *,
    lanes: Sequence[BatchedSweepLogLane],
    active_users: Sequence[int],
    fsrs_weights: torch.Tensor,
    device: torch.device,
) -> FSRS6CostConditionedADRBatchSchedulerOps:
    policies = [
        FSRS6CostConditionedADRPolicy.from_json(
            _require_path(lane.fsrs6_cost_adr_policy)
        )
        for lane in lanes
    ]
    policy = policies[0]
    coefficients = torch.tensor(
        [candidate.coefficients for candidate in policies],
        device=device,
        dtype=torch.float32,
    )
    scheduler_weights = _repeat_weights_for_lanes(
        weights=fsrs_weights.to(device),
        active_batch=list(active_users),
        lanes=list(lanes),
    )
    goal_weights = torch.tensor(
        [
            _require_float(
                lane.fsrs6_cost_adr_goal_cost_weight,
                "fsrs6_cost_adr_goal_cost_weight",
            )
            for lane in lanes
        ],
        device=device,
        dtype=torch.float32,
    )
    return FSRS6CostConditionedADRBatchSchedulerOps(
        weights=scheduler_weights,
        policy=policy,
        goal_cost_weight=goal_weights,
        coefficients=coefficients,
        bounds=policy.bounds,
        priority_mode="low_retrievability",
        device=device,
        dtype=torch.float32,
    )


def run_chunk(
    *,
    config: AnalysisConfig,
    env: str,
    users: Sequence[int],
) -> ChunkResult:
    lanes = build_lanes(
        run_root=config.run_root,
        users=users,
        env=env,
        cost_weights=config.cost_weights,
    )
    lane_user_ids = [lane.user_id for lane in lanes]
    fsrs_weights, fsrs_users = load_fsrs6_weights(
        repo_root=REPO_ROOT,
        user_ids=list(users),
        benchmark_root=config.benchmark_root,
        benchmark_partition=config.benchmark_partition,
        overrides=config.benchmark_overrides,
        short_term=False,
        device=config.device,
    )
    if fsrs_weights is None or tuple(fsrs_users) != tuple(users):
        raise RuntimeError(f"Could not load FSRS6 weights for users {list(users)}.")

    if env == "fsrs6":
        lane_weights = _repeat_weights_for_lanes(
            weights=fsrs_weights.to(config.device),
            active_batch=list(users),
            lanes=lanes,
        )
        env_ops = FSRS6BatchEnvOps(
            weights=lane_weights,
            bounds=Bounds(),
            device=lane_weights.device,
            dtype=torch.float32,
        )
    elif env == "lstm":
        lstm_paths, kept = resolve_lstm_paths(
            list(users), config.benchmark_root, short_term=False
        )
        if tuple(kept) != tuple(users):
            raise RuntimeError(f"Could not load LSTM weights for users {list(users)}.")
        packed = PackedLSTMWeights.from_paths(
            lstm_paths,
            use_duration_feature=False,
            device=config.device,
            dtype=torch.float32,
        )
        lane_packed = _repeat_lstm_weights_for_lanes(
            weights=packed,
            active_batch=list(users),
            lanes=lanes,
        )
        env_ops = LSTMBatchedEnvOps(
            lane_packed,
            device=lane_packed.process_0_weight.device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported env {env!r}.")

    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
        review_markov_success_weights,
    ) = load_usage(
        lane_user_ids,
        config.button_usage,
        review_markov_transition=False,
    )
    behavior, cost_model = build_behavior_cost(
        len(lane_user_ids),
        deck_size=config.deck,
        learn_limit=config.learn_limit,
        review_limit=config.review_limit,
        cost_limit_minutes=config.cost_limit_minutes,
        learn_costs=learn_costs.to(env_ops.device),
        review_costs=review_costs.to(env_ops.device),
        first_rating_prob=first_rating_prob.to(env_ops.device),
        review_rating_prob=review_rating_prob.to(env_ops.device),
        learning_rating_prob=learning_rating_prob.to(env_ops.device),
        relearning_rating_prob=relearning_rating_prob.to(env_ops.device),
        state_rating_costs=state_rating_costs.to(env_ops.device),
        review_markov_success_weights=review_markov_success_weights,
        short_term=False,
    )
    sched_ops = build_scheduler_ops(
        lanes=lanes,
        active_users=users,
        fsrs_weights=fsrs_weights,
        device=env_ops.device,
    )
    collector = OccupancyCollector(
        env=env,
        lanes=lanes,
        cost_weights=config.cost_weights,
        device=env_ops.device,
    )
    attach_collector(sched_ops, collector)

    print(
        f"start env={env} users={users[0]}-{users[-1]} "
        f"lanes={len(lanes)} device={env_ops.device}",
        flush=True,
    )
    started = time.perf_counter()
    progress_last = -365

    def progress_callback(completed: int, total: int) -> None:
        nonlocal progress_last
        if completed == total or completed - progress_last >= 365:
            print(
                f"progress env={env} users={users[0]}-{users[-1]} "
                f"day={completed}/{total}",
                flush=True,
            )
            progress_last = completed

    stats = simulate_multiuser(
        days=config.days,
        deck_size=config.deck,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        priority_mode=config.priority,
        seed=config.seed,
        device=env_ops.device,
        dtype=torch.float32,
        fuzz=False,
        progress=False,
        progress_callback=progress_callback,
        short_term_source=None,
    )
    elapsed = time.perf_counter() - started
    print(
        f"done env={env} users={users[0]}-{users[-1]} elapsed={elapsed:.1f}s "
        f"max_interval_days={collector.max_interval_days.item():.1f}",
        flush=True,
    )
    return collector.finalize(stats=stats, elapsed_seconds=elapsed)


def split_users(
    users: Sequence[int],
    *,
    cost_weight_count: int,
    max_lanes: int,
) -> list[tuple[int, ...]]:
    users_per_chunk = max(1, max_lanes // cost_weight_count)
    return [
        tuple(users[index : index + users_per_chunk])
        for index in range(0, len(users), users_per_chunk)
    ]


def combine_results(results: Sequence[ChunkResult]) -> dict[str, Any]:
    by_env: dict[str, dict[str, Any]] = {}
    lane_records: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for env in sorted({result.env for result in results}):
        subset = [result for result in results if result.env == env]
        cost_weight_count = int(subset[0].cost_phase_count.shape[1])
        count = np.zeros((len(PHASES), cost_weight_count), dtype=np.float64)
        retention_sum = np.zeros_like(count)
        interval_sum = np.zeros_like(count)
        retention_hist = np.zeros(
            (len(PHASES), cost_weight_count, RETENTION_BINS),
            dtype=np.int64,
        )
        interval_hist = np.zeros(
            (len(PHASES), cost_weight_count, INTERVAL_BINS),
            dtype=np.int64,
        )
        region_count = np.zeros(
            (len(PHASES), cost_weight_count, REGION_COUNT),
            dtype=np.float64,
        )
        region_retention_sum = np.zeros_like(region_count)
        region_interval_sum = np.zeros_like(region_count)

        for result in subset:
            count += result.cost_phase_count
            retention_sum += result.cost_phase_retention_sum
            interval_sum += result.cost_phase_interval_sum
            retention_hist += result.cost_phase_retention_hist
            interval_hist += result.cost_phase_interval_hist
            region_count += result.region_count
            region_retention_sum += result.region_retention_sum
            region_interval_sum += result.region_interval_sum
            lane_records.extend(result.lane_records)
            stats_rows.extend(result.stats_rows)

        by_env[env] = {
            "count": count,
            "retention_sum": retention_sum,
            "interval_sum": interval_sum,
            "retention_hist": retention_hist,
            "interval_hist": interval_hist,
            "region_count": region_count,
            "region_retention_sum": region_retention_sum,
            "region_interval_sum": region_interval_sum,
            "max_interval_days": max(result.max_interval_days for result in subset),
            "max_raw_interval_days": max(
                result.max_raw_interval_days for result in subset
            ),
            "max_s": max(result.max_s for result in subset),
            "max_d": max(result.max_d for result in subset),
            "elapsed_seconds": sum(result.elapsed_seconds for result in subset),
        }
    return {"by_env": by_env, "lane_records": lane_records, "stats_rows": stats_rows}


def write_outputs(
    *,
    config: AnalysisConfig,
    results: Sequence[ChunkResult],
    gpu_summary: GpuMonitorSummary,
) -> None:
    combined = combine_results(results)
    by_env = combined["by_env"]
    by_cost_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    selected_review_rows: list[dict[str, Any]] = []

    for env, data in by_env.items():
        for cost_idx, cost_weight in enumerate(config.cost_weights):
            for phase_idx, phase in enumerate(PHASES):
                count = int(data["count"][phase_idx, cost_idx])
                if count <= 0:
                    continue
                retention_hist = data["retention_hist"][phase_idx, cost_idx]
                interval_hist = data["interval_hist"][phase_idx, cost_idx]
                row = {
                    "environment": env,
                    "cost_weight": cost_weight,
                    "phase": phase,
                    "events": count,
                    "mean_retention": float(
                        data["retention_sum"][phase_idx, cost_idx] / count
                    ),
                    "retention_q05": hist_quantile(
                        retention_hist,
                        RETENTION_MIN,
                        RETENTION_MAX,
                        0.05,
                    ),
                    "retention_q25": hist_quantile(
                        retention_hist,
                        RETENTION_MIN,
                        RETENTION_MAX,
                        0.25,
                    ),
                    "retention_q50": hist_quantile(
                        retention_hist,
                        RETENTION_MIN,
                        RETENTION_MAX,
                        0.50,
                    ),
                    "retention_q75": hist_quantile(
                        retention_hist,
                        RETENTION_MIN,
                        RETENTION_MAX,
                        0.75,
                    ),
                    "retention_q95": hist_quantile(
                        retention_hist,
                        RETENTION_MIN,
                        RETENTION_MAX,
                        0.95,
                    ),
                    "mean_interval_days": float(
                        data["interval_sum"][phase_idx, cost_idx] / count
                    ),
                    "interval_q05_days": hist_quantile(
                        interval_hist,
                        LOG_INTERVAL_MIN,
                        LOG_INTERVAL_MAX,
                        0.05,
                        log_scale=True,
                    ),
                    "interval_q25_days": hist_quantile(
                        interval_hist,
                        LOG_INTERVAL_MIN,
                        LOG_INTERVAL_MAX,
                        0.25,
                        log_scale=True,
                    ),
                    "interval_q50_days": hist_quantile(
                        interval_hist,
                        LOG_INTERVAL_MIN,
                        LOG_INTERVAL_MAX,
                        0.50,
                        log_scale=True,
                    ),
                    "interval_q75_days": hist_quantile(
                        interval_hist,
                        LOG_INTERVAL_MIN,
                        LOG_INTERVAL_MAX,
                        0.75,
                        log_scale=True,
                    ),
                    "interval_q95_days": hist_quantile(
                        interval_hist,
                        LOG_INTERVAL_MIN,
                        LOG_INTERVAL_MAX,
                        0.95,
                        log_scale=True,
                    ),
                }
                by_cost_rows.append(row)
                if phase == "review" and cost_weight in {
                    0.0,
                    16.0,
                    64.0,
                    128.0,
                    384.0,
                    1024.0,
                }:
                    selected_review_rows.append(row)

        for cost_idx, cost_weight in enumerate(config.cost_weights):
            for phase_idx, phase in enumerate(PHASES):
                total = float(data["count"][phase_idx, cost_idx])
                if total <= 0.0:
                    continue
                for s_idx, s_label in enumerate(S_BINS):
                    for d_idx, d_label in enumerate(D_BINS):
                        region_idx = s_idx * len(D_BINS) + d_idx
                        count = float(
                            data["region_count"][phase_idx, cost_idx, region_idx]
                        )
                        if count <= 0.0:
                            continue
                        region_rows.append(
                            {
                                "environment": env,
                                "cost_weight": cost_weight,
                                "phase": phase,
                                "s_bin": s_label,
                                "d_bin": d_label,
                                "events": int(count),
                                "event_share_percent": count / total * 100.0,
                                "mean_retention": float(
                                    data["region_retention_sum"][
                                        phase_idx,
                                        cost_idx,
                                        region_idx,
                                    ]
                                    / count
                                ),
                                "mean_interval_days": float(
                                    data["region_interval_sum"][
                                        phase_idx,
                                        cost_idx,
                                        region_idx,
                                    ]
                                    / count
                                ),
                            }
                        )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(config.out_dir / "empirical_by_cost_phase.csv", by_cost_rows)
    write_csv(
        config.out_dir / "empirical_state_region_by_cost_phase.csv",
        region_rows,
    )
    write_csv(
        config.out_dir / "empirical_lane_phase_summary.csv",
        combined["lane_records"],
    )
    write_csv(
        config.out_dir / "empirical_simulation_stats_by_lane.csv",
        combined["stats_rows"],
    )

    environment_summary = {
        env: {
            "elapsed_seconds": data["elapsed_seconds"],
            "max_interval_days": data["max_interval_days"],
            "max_raw_interval_days": data["max_raw_interval_days"],
            "max_s": data["max_s"],
            "max_d": data["max_d"],
            "learn_events": int(data["count"][0].sum()),
            "review_events": int(data["count"][1].sum()),
        }
        for env, data in by_env.items()
    }
    summary = {
        "schema_version": 1,
        "source_run_root": str(config.run_root),
        "users": list(config.users),
        "user_count": len(config.users),
        "cost_weights": list(config.cost_weights),
        "method": {
            "description": (
                "Batched replay with scheduler update hooks. Histograms are "
                "weighted by actual scheduling decisions in simulation."
            ),
            "days": config.days,
            "deck": config.deck,
            "learn_limit": config.learn_limit,
            "review_limit": config.review_limit,
            "cost_limit_minutes": config.cost_limit_minutes,
            "seed": config.seed,
            "priority": config.priority,
            "scheduler_priority": config.scheduler_priority,
            "fuzz": False,
            "review_markov_transition": False,
            "fsrs6_max_lanes": config.fsrs6_max_lanes,
            "lstm_max_lanes": config.lstm_max_lanes,
            "retention_bins": RETENTION_BINS,
            "log_interval_bins": INTERVAL_BINS,
        },
        "chunks": [
            {
                "environment": result.env,
                "elapsed_seconds": result.elapsed_seconds,
                "max_interval_days": result.max_interval_days,
                "max_raw_interval_days": result.max_raw_interval_days,
                "max_s": result.max_s,
                "max_d": result.max_d,
            }
            for result in results
        ],
        "environment_summary": environment_summary,
        "selected_review_rows": selected_review_rows,
        "gpu_monitor": gpu_summary.to_dict(),
    }
    (config.out_dir / "empirical_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(environment_summary, indent=2), flush=True)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _require_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("Expected a policy path.")
    return path


def _require_float(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Expected {label}.")
    return float(value)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Cost-ADR policies and compute empirical occupancy-weighted "
            "retention/interval distributions."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--users", default="1-128")
    parser.add_argument("--envs", default="fsrs6,lstm")
    parser.add_argument(
        "--cost-weights",
        default=",".join(format(weight, "g") for weight in DEFAULT_COST_WEIGHTS),
    )
    parser.add_argument("--days", type=positive_int, default=1825)
    parser.add_argument("--deck", type=positive_int, default=10000)
    parser.add_argument("--learn-limit", type=positive_int, default=10)
    parser.add_argument("--review-limit", type=positive_int, default=9999)
    parser.add_argument("--cost-limit-minutes", type=float, default=720.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--priority", choices=("new-first", "review-first"), default="new-first"
    )
    parser.add_argument("--scheduler-priority", default="low_retrievability")
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-partition", default="0")
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--fsrs6-max-lanes",
        type=positive_int,
        default=2048,
        help="Maximum simultaneous FSRS6 lanes. 128 users x 16 weights = 2048.",
    )
    parser.add_argument(
        "--lstm-max-lanes",
        type=positive_int,
        default=512,
        help=(
            "Maximum simultaneous LSTM lanes. Default 512 is deliberately below "
            "the production 1024 cap to reduce VRAM/OOM risk while tracing."
        ),
    )
    parser.add_argument("--gpu-monitor-interval", type=float, default=2.0)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    return AnalysisConfig(
        run_root=args.run_root,
        out_dir=args.out_dir,
        users=parse_users(args.users),
        envs=parse_envs(args.envs),
        cost_weights=parse_floats(args.cost_weights),
        days=args.days,
        deck=args.deck,
        learn_limit=args.learn_limit,
        review_limit=args.review_limit,
        cost_limit_minutes=float(args.cost_limit_minutes),
        seed=args.seed,
        priority=args.priority,
        scheduler_priority=args.scheduler_priority,
        benchmark_root=benchmark_root,
        benchmark_partition=args.benchmark_partition,
        benchmark_overrides=parse_result_overrides(args.benchmark_result),
        button_usage=args.button_usage,
        device=device,
        fsrs6_max_lanes=args.fsrs6_max_lanes,
        lstm_max_lanes=args.lstm_max_lanes,
        gpu_monitor=not args.no_gpu_monitor,
        gpu_monitor_interval_seconds=float(args.gpu_monitor_interval),
    )


def run(config: AnalysisConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    monitor = (
        GpuMonitor(
            output_dir=config.out_dir / "gpu_monitor",
            interval_seconds=config.gpu_monitor_interval_seconds,
        )
        if config.gpu_monitor
        else None
    )
    if monitor is not None:
        monitor.start()
    results: list[ChunkResult] = []
    gpu_summary = disabled_monitor_summary(config.out_dir / "gpu_monitor")
    try:
        print(f"benchmark_root={config.benchmark_root}", flush=True)
        print(f"device={config.device}", flush=True)
        for env in config.envs:
            max_lanes = (
                config.fsrs6_max_lanes if env == "fsrs6" else config.lstm_max_lanes
            )
            chunks = split_users(
                config.users,
                cost_weight_count=len(config.cost_weights),
                max_lanes=max_lanes,
            )
            print(
                f"env={env} max_lanes={max_lanes} chunks={len(chunks)}",
                flush=True,
            )
            for users in chunks:
                results.append(run_chunk(config=config, env=env, users=users))
                if config.device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if monitor is not None:
            gpu_summary = monitor.stop()
    write_outputs(config=config, results=results, gpu_summary=gpu_summary)
    if gpu_summary.shared_memory_spill_detected:
        raise SystemExit(
            "GPU shared-memory spill detected. Reduce --lstm-max-lanes or "
            "--fsrs6-max-lanes and rerun."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args(argv))
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
