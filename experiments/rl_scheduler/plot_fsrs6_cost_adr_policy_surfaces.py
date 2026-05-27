from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.fsrs6_cost_adr_policy import (  # noqa: E402
    DEFAULT_COST_WEIGHTS,
)
from simulator.benchmark_loader import (  # noqa: E402
    load_benchmark_weights,
    parse_result_overrides,
    resolve_benchmark_root,
)
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FSRS6CostConditionedADRPolicy,
    normalized_cost_weight,
)
from simulator.math.fsrs import (  # noqa: E402
    FSRS6Params,
    fsrs6_forgetting_curve,
    fsrs6_next_interval,
)

ZMode = Literal["interval", "log_interval", "retention"]


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    user_id: int
    path: Path
    policy: FSRS6CostConditionedADRPolicy
    cost_weights: tuple[float, ...]
    policy_index: int | None = None


def _parse_csv_floats(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return _cost_weight_tuple(
        tuple(float(item) for item in items), source="--cost-weights"
    )


def _parse_csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def _path_user_id(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"user_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _path_policy_index(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"policy_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return raw


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json_object(path)


def _metadata_user_id(metadata: Mapping[str, Any]) -> int | None:
    raw_user_id = metadata.get("user_id")
    if raw_user_id is not None:
        return int(raw_user_id)
    raw_training_user_ids = metadata.get("training_user_ids")
    if isinstance(raw_training_user_ids, list) and len(raw_training_user_ids) == 1:
        return int(raw_training_user_ids[0])
    return None


def _metadata_policy_index(metadata: Mapping[str, Any]) -> int | None:
    raw_policy_index = metadata.get("portfolio_index")
    if raw_policy_index is None:
        return None
    if isinstance(raw_policy_index, bool) or not isinstance(raw_policy_index, int):
        raise SystemExit("metadata.json field 'portfolio_index' must be an integer.")
    return raw_policy_index


def _metadata_cost_weights(
    metadata: Mapping[str, Any],
    *,
    source: Path,
) -> tuple[float, ...] | None:
    raw_weights = metadata.get("cost_weights")
    if raw_weights is None:
        return None
    return _cost_weight_tuple(raw_weights, source=f"{source} field 'cost_weights'")


def _cost_weight_tuple(raw_weights: Sequence[Any], *, source: str) -> tuple[float, ...]:
    if isinstance(raw_weights, str) or not isinstance(raw_weights, Sequence):
        raise SystemExit(f"{source} must be an array of numbers.")
    weights = tuple(_finite_nonnegative_float(item, source) for item in raw_weights)
    if not weights:
        raise SystemExit(f"{source} must not be empty.")
    if len(set(weights)) != len(weights):
        raise SystemExit(f"{source} must not contain duplicates.")
    return weights


def _finite_nonnegative_float(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(f"{source} must contain only numbers.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SystemExit(f"{source} values must be finite and >= 0.")
    return result


def _resolve_policy_root(args: argparse.Namespace) -> Path:
    if args.policy_root is not None and args.train_run_root is not None:
        raise SystemExit("Use only one of --policy-root or --train-run-root.")
    if args.train_run_root is not None:
        return args.train_run_root / "train-overfit" / "train_outputs"
    if args.policy_root is not None:
        return args.policy_root
    raise SystemExit("Provide --policy-root or --train-run-root.")


def _discover_policies(
    root: Path,
    *,
    users: Sequence[int] | None,
    start_user: int | None,
    end_user: int | None,
    cost_weights: Sequence[float] | None,
) -> list[PolicyEntry]:
    if not root.exists():
        raise SystemExit(f"Policy root not found: {root}")

    user_filter = set(users) if users is not None else None
    entries: list[PolicyEntry] = []
    for path in sorted(root.rglob("policy.json")):
        try:
            policy = FSRS6CostConditionedADRPolicy.from_json(path)
        except ValueError:
            continue
        metadata_path = path.with_name("metadata.json")
        artifact_metadata = _read_optional_json_object(metadata_path)
        user_id = _path_user_id(path)
        if user_id is None:
            user_id = _metadata_user_id(artifact_metadata)
        if user_id is None:
            continue
        if user_filter is not None and user_id not in user_filter:
            continue
        if start_user is not None and user_id < start_user:
            continue
        if end_user is not None and user_id > end_user:
            continue

        entry_cost_weights = (
            tuple(cost_weights)
            if cost_weights is not None
            else _metadata_cost_weights(artifact_metadata, source=metadata_path)
        )
        if entry_cost_weights is None:
            entry_cost_weights = DEFAULT_COST_WEIGHTS

        entries.append(
            PolicyEntry(
                user_id=user_id,
                path=path.resolve(),
                policy=policy,
                cost_weights=entry_cost_weights,
                policy_index=_metadata_policy_index(artifact_metadata)
                if artifact_metadata
                else _path_policy_index(path),
            )
        )

    if not entries:
        raise SystemExit(
            f"No matching FSRS6 Cost-ADR policy.json files found under {root}"
        )
    return sorted(
        entries,
        key=lambda entry: (
            entry.user_id,
            entry.policy_index if entry.policy_index is not None else 10**9,
            str(entry.path),
        ),
    )


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("point count must be >= 2")
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _logspace(start: float, end: float, count: int) -> list[float]:
    if start <= 0.0:
        raise ValueError("S minimum must be > 0 for log-spaced grid.")
    log_start = math.log(start)
    log_end = math.log(end)
    return [math.exp(value) for value in _linspace(log_start, log_end, count)]


def _cost_weight_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _policy_suffix(entry: PolicyEntry) -> str:
    if entry.policy_index is None:
        return ""
    return f"_policy_{entry.policy_index}"


def _scale_visibility(trace_count: int, visible_index: int | None = None) -> list[bool]:
    if trace_count == 0:
        return []
    if visible_index is not None:
        return [index == visible_index for index in range(trace_count)]
    return [index == trace_count - 1 for index in range(trace_count)]


def _interval_z(interval_days: float, z_mode: ZMode) -> float:
    if z_mode == "interval":
        return interval_days
    if z_mode == "retention":
        raise ValueError("retention z mode requires a retention value.")
    return math.log10(max(interval_days, 1e-12))


def _z_axis_title(z_mode: ZMode) -> str:
    if z_mode == "retention":
        return "Retention"
    if z_mode == "log_interval":
        return "log10(interval days)"
    return "Interval days"


def _z_hover_line(z_mode: ZMode) -> str:
    if z_mode == "retention":
        return "retention_z=%{z:.4f}<extra></extra>"
    if z_mode == "log_interval":
        return "log10_interval=%{z:.3f}<extra></extra>"
    return "interval_z=%{z:.3g}<extra></extra>"


def _hovertemplate(
    z_mode: ZMode,
    *,
    show_interval: bool,
    show_retention: bool,
) -> str:
    lines = [
        "cost_weight=%{customdata[0]:g}",
        "normalized_w=%{customdata[1]:.3f}",
        "S=%{x:.3g}",
        "D=%{y:.3g}",
    ]
    if show_interval:
        lines.append("interval_days=%{customdata[2]:.3g}")
    if show_retention:
        lines.append("retention=%{customdata[3]:.4f}")
    lines.append(_z_hover_line(z_mode))
    return "<br>".join(lines)


def _entry_needs_fsrs6_params(entry: PolicyEntry, z_mode: ZMode) -> bool:
    if z_mode == "retention":
        return entry.policy.action_head == ACTION_HEAD_INTERVAL
    return entry.policy.action_head == ACTION_HEAD_RETENTION


def _load_fsrs6_params(
    *,
    user_id: int,
    srs_benchmark_root: Path | None,
    benchmark_result: str | None,
    benchmark_partition: str | None,
) -> FSRS6Params:
    benchmark_root = resolve_benchmark_root(REPO_ROOT, srs_benchmark_root)
    overrides = parse_result_overrides(benchmark_result)
    try:
        weights = load_benchmark_weights(
            repo_root=REPO_ROOT,
            benchmark_root=benchmark_root,
            environment="fsrs6",
            user_id=user_id,
            partition_key=benchmark_partition or "0",
            overrides=overrides,
            short_term=False,
        )
        return FSRS6Params(tuple(weights))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Could not load FSRS6 weights for user {user_id}: {exc}"
        ) from exc


def _load_needed_fsrs6_params(
    entries: Sequence[PolicyEntry],
    *,
    z_mode: ZMode,
    srs_benchmark_root: Path | None,
    benchmark_result: str | None,
    benchmark_partition: str | None,
) -> dict[int, FSRS6Params]:
    needed_user_ids = sorted(
        {entry.user_id for entry in entries if _entry_needs_fsrs6_params(entry, z_mode)}
    )
    return {
        user_id: _load_fsrs6_params(
            user_id=user_id,
            srs_benchmark_root=srs_benchmark_root,
            benchmark_result=benchmark_result,
            benchmark_partition=benchmark_partition,
        )
        for user_id in needed_user_ids
    }


def _evaluate_policy_point(
    *,
    policy: FSRS6CostConditionedADRPolicy,
    fsrs6_params: FSRS6Params | None,
    stability: float,
    difficulty: float,
    cost_weight: float,
) -> tuple[float, float]:
    if policy.action_head == ACTION_HEAD_INTERVAL:
        interval_days = policy.evaluate_interval(
            stability=stability,
            difficulty=difficulty,
            cost_weight=cost_weight,
        )
        retention = (
            math.nan
            if fsrs6_params is None
            else fsrs6_forgetting_curve(fsrs6_params, interval_days, stability)
        )
        return interval_days, retention
    if policy.action_head == ACTION_HEAD_RETENTION:
        retention = policy.evaluate_retention(
            stability=stability,
            difficulty=difficulty,
            cost_weight=cost_weight,
        )
        interval_days = (
            math.nan
            if fsrs6_params is None
            else fsrs6_next_interval(fsrs6_params, stability, retention)
        )
        return interval_days, retention
    raise AssertionError(f"Unexpected action_head={policy.action_head!r}.")


def _surface_z(interval_days: float, retention: float, z_mode: ZMode) -> float:
    if z_mode == "retention":
        return retention
    return _interval_z(interval_days, z_mode)


def _build_surface_arrays(
    *,
    policy: FSRS6CostConditionedADRPolicy,
    fsrs6_params: FSRS6Params | None = None,
    cost_weight: float,
    s_grid: Sequence[float],
    d_grid: Sequence[float],
    z_mode: ZMode,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    z_rows: list[list[float]] = []
    customdata_rows: list[list[list[float]]] = []
    normalized_weight = normalized_cost_weight(
        cost_weight,
        cost_weight_min=policy.cost_weight_min,
        cost_weight_max=policy.cost_weight_max,
    )
    for d_value in d_grid:
        z_row: list[float] = []
        customdata_row: list[list[float]] = []
        for s_value in s_grid:
            interval_days, retention = _evaluate_policy_point(
                policy=policy,
                fsrs6_params=fsrs6_params,
                stability=s_value,
                difficulty=d_value,
                cost_weight=cost_weight,
            )
            z_row.append(_surface_z(interval_days, retention, z_mode))
            customdata_row.append(
                [cost_weight, normalized_weight, interval_days, retention]
            )
        z_rows.append(z_row)
        customdata_rows.append(customdata_row)
    return z_rows, customdata_rows


def _write_user_plot(
    *,
    entry: PolicyEntry,
    out_dir: Path,
    s_points: int,
    d_points: int,
    s_min: float | None,
    s_max: float | None,
    d_min: float | None,
    d_max: float | None,
    opacity: float,
    z_mode: ZMode,
    fsrs6_params: FSRS6Params | None,
) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for this tool. Install dependencies with `uv sync`."
        ) from exc

    if entry.policy.action_head not in {ACTION_HEAD_INTERVAL, ACTION_HEAD_RETENTION}:
        raise SystemExit(
            f"{entry.path} uses action_head={entry.policy.action_head!r}; "
            "this visualizer expects interval or desired_retention policies."
        )
    if _entry_needs_fsrs6_params(entry, z_mode) and fsrs6_params is None:
        raise SystemExit(
            f"{entry.path} needs FSRS6 user weights for --z-mode {z_mode!r}."
        )

    bounds = entry.policy.bounds
    resolved_s_min = s_min if s_min is not None else bounds.s_min
    resolved_s_max = s_max if s_max is not None else bounds.s_max
    resolved_d_min = d_min if d_min is not None else bounds.d_min
    resolved_d_max = d_max if d_max is not None else bounds.d_max
    s_grid = _logspace(resolved_s_min, resolved_s_max, s_points)
    d_grid = _linspace(resolved_d_min, resolved_d_max, d_points)
    cost_weights = tuple(sorted(entry.cost_weights))
    color_values = [math.log1p(weight) for weight in cost_weights]
    color_min = min(color_values)
    color_max = max(color_values)
    if math.isclose(color_min, color_max, rel_tol=0.0, abs_tol=1e-12):
        color_max = color_min + 1.0

    z_min = math.inf
    z_max = -math.inf
    fig = go.Figure()
    show_interval_hover = (
        entry.policy.action_head == ACTION_HEAD_INTERVAL or fsrs6_params is not None
    )
    show_retention_hover = (
        entry.policy.action_head == ACTION_HEAD_RETENTION or fsrs6_params is not None
    )
    for index, (cost_weight, color_value) in enumerate(
        zip(cost_weights, color_values, strict=True)
    ):
        z_values, customdata = _build_surface_arrays(
            policy=entry.policy,
            fsrs6_params=fsrs6_params,
            cost_weight=cost_weight,
            s_grid=s_grid,
            d_grid=d_grid,
            z_mode=z_mode,
        )
        z_min = min(z_min, min(min(row) for row in z_values))
        z_max = max(z_max, max(max(row) for row in z_values))
        surface_color = [[color_value for _ in s_grid] for _ in d_grid]
        fig.add_trace(
            go.Surface(
                x=s_grid,
                y=d_grid,
                z=z_values,
                customdata=customdata,
                surfacecolor=surface_color,
                cmin=color_min,
                cmax=color_max,
                colorscale="Viridis",
                opacity=opacity,
                showlegend=True,
                showscale=index == len(cost_weights) - 1,
                colorbar={
                    "title": {"text": "cost weight"},
                    "x": 1.05,
                    "xanchor": "left",
                    "y": 0.5,
                    "len": 0.72,
                    "thickness": 16,
                    "tickmode": "array",
                    "tickvals": color_values,
                    "ticktext": [_cost_weight_label(weight) for weight in cost_weights],
                },
                name=f"w={_cost_weight_label(cost_weight)}",
                hovertemplate=_hovertemplate(
                    z_mode,
                    show_interval=show_interval_hover,
                    show_retention=show_retention_hover,
                ),
            )
        )

    trace_count = len(cost_weights)
    buttons: list[dict[str, Any]] = [
        {
            "label": "All weights",
            "method": "update",
            "args": [
                {
                    "visible": [True for _ in cost_weights],
                    "showscale": _scale_visibility(trace_count),
                }
            ],
        },
        {
            "label": "Hide all",
            "method": "update",
            "args": [
                {
                    "visible": ["legendonly" for _ in cost_weights],
                    "showscale": [False for _ in cost_weights],
                }
            ],
        },
    ]
    for index, cost_weight in enumerate(cost_weights):
        buttons.append(
            {
                "label": f"w={_cost_weight_label(cost_weight)}",
                "method": "update",
                "args": [
                    {
                        "visible": [
                            True if trace_index == index else "legendonly"
                            for trace_index in range(trace_count)
                        ],
                        "showscale": _scale_visibility(trace_count, index),
                    }
                ],
            }
        )

    if math.isfinite(z_min) and math.isfinite(z_max) and z_min < z_max:
        zaxis: dict[str, Any] = {
            "title": _z_axis_title(z_mode),
            "range": [z_min, z_max],
        }
    else:
        zaxis = {"title": _z_axis_title(z_mode)}
    policy_text = "" if entry.policy_index is None else f", policy={entry.policy_index}"
    fig.update_layout(
        title=(
            f"FSRS6 Cost-ADR {z_mode} policy surfaces: "
            f"user {entry.user_id}{policy_text}"
        ),
        scene={
            "domain": {"x": [0.08, 0.9], "y": [0.0, 0.96]},
            "xaxis": {"title": "Stability S", "type": "log"},
            "yaxis": {"title": "Difficulty D"},
            "zaxis": zaxis,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.08,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ],
        margin={"l": 140, "r": 130, "b": 0, "t": 95},
        legend={
            "title": {"text": "cost-weight surfaces"},
            "x": -0.13,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "itemsizing": "constant",
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / (
        f"fsrs6_cost_adr_policy_surfaces_user_{entry.user_id}"
        f"{_policy_suffix(entry)}.html"
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Plot FSRS6 Cost-ADR policy surfaces by user and cost weight."),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy-root",
        type=Path,
        help="Root containing user_*/policy.json files.",
    )
    source.add_argument(
        "--train-run-root",
        type=Path,
        help="Run root containing train-overfit/train_outputs.",
    )
    parser.add_argument(
        "--users",
        default=None,
        help="Comma-separated user ids to include. Overrides start/end filtering.",
    )
    parser.add_argument("--start-user", type=int, default=None)
    parser.add_argument("--end-user", type=int, default=None)
    parser.add_argument(
        "--cost-weights",
        default=None,
        help=(
            "Comma-separated cost weights to plot. Default: sibling metadata "
            "cost_weights, then the formal 16-weight Cost-ADR grid."
        ),
    )
    parser.add_argument("--s-points", type=int, default=48)
    parser.add_argument("--d-points", type=int, default=48)
    parser.add_argument("--s-min", type=float, default=None)
    parser.add_argument("--s-max", type=float, default=None)
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument(
        "--z-mode",
        choices=("retention", "log_interval", "interval"),
        default="retention",
        help=(
            "Use retention, log10(interval days), or raw interval days on the z axis. "
            "Retention mode converts interval-head policies through the user's FSRS6 "
            "forgetting curve."
        ),
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help=(
            "Optional srs-benchmark root for FSRS6 user weights. Required only when "
            "the selected z mode needs interval/retention conversion."
        ),
    )
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help="Optional benchmark result override, e.g. fsrs6=FSRS-6-recency.",
    )
    parser.add_argument(
        "--benchmark-partition",
        default=None,
        help="Benchmark parameter partition key. Default: 0.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="Surface opacity for overlaid cost-weight curves.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments")
        / "rl_scheduler"
        / "plots"
        / "fsrs6_cost_adr_policy_surfaces",
        help="Directory for generated HTML plots.",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.opacity <= 1.0):
        raise SystemExit("--opacity must satisfy 0 < opacity <= 1.")
    if args.s_points < 2 or args.d_points < 2:
        raise SystemExit("--s-points and --d-points must be >= 2.")
    if args.users is not None and (
        args.start_user is not None or args.end_user is not None
    ):
        raise SystemExit("--users cannot be combined with --start-user/--end-user.")

    policy_root = _resolve_policy_root(args)
    entries = _discover_policies(
        policy_root,
        users=_parse_csv_ints(args.users),
        start_user=args.start_user,
        end_user=args.end_user,
        cost_weights=_parse_csv_floats(args.cost_weights),
    )
    fsrs6_params_by_user = _load_needed_fsrs6_params(
        entries,
        z_mode=args.z_mode,
        srs_benchmark_root=args.srs_benchmark_root,
        benchmark_result=args.benchmark_result,
        benchmark_partition=args.benchmark_partition,
    )
    output_paths = [
        _write_user_plot(
            entry=entry,
            out_dir=args.out_dir,
            s_points=args.s_points,
            d_points=args.d_points,
            s_min=args.s_min,
            s_max=args.s_max,
            d_min=args.d_min,
            d_max=args.d_max,
            opacity=args.opacity,
            z_mode=args.z_mode,
            fsrs6_params=fsrs6_params_by_user.get(entry.user_id),
        )
        for entry in entries
    ]
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
