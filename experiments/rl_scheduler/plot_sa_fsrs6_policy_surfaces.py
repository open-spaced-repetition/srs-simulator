from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.sa_policy import format_float_token, parse_float_token
from simulator.sa_fsrs6_policy import SAFSRS6Policy


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    user_id: int
    baseline_desired_retention: float
    lambda_value: float | None
    path: Path
    policy: SAFSRS6Policy


def _parse_csv_floats(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(float(item) for item in items)


def _parse_csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in items)


def _matches_float(value: float | None, allowed: Sequence[float] | None) -> bool:
    if allowed is None:
        return True
    if value is None:
        return False
    return any(math.isclose(value, item, rel_tol=0.0, abs_tol=1e-9) for item in allowed)


def _path_token_float(path: Path, prefix: str) -> float | None:
    for part in path.parts:
        if not part.startswith(prefix):
            continue
        try:
            return parse_float_token(part[len(prefix) :])
        except ValueError:
            continue
    return None


def _path_user_id(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"user_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


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
    dr_values: Sequence[float] | None,
    lambda_values: Sequence[float] | None,
) -> list[PolicyEntry]:
    if not root.exists():
        raise SystemExit(f"Policy root not found: {root}")

    user_filter = set(users) if users is not None else None
    entries: list[PolicyEntry] = []
    for path in sorted(root.rglob("policy.json")):
        policy = SAFSRS6Policy.from_json(path)
        user_id = _path_user_id(path)
        if user_id is None:
            metadata = policy.metadata or {}
            raw_user_id = metadata.get("user_id")
            user_id = int(raw_user_id) if raw_user_id is not None else None
        if user_id is None:
            continue
        if user_filter is not None and user_id not in user_filter:
            continue
        if start_user is not None and user_id < start_user:
            continue
        if end_user is not None and user_id > end_user:
            continue

        baseline_dr = _path_token_float(path, "dr_")
        if baseline_dr is None:
            baseline_dr = float(policy.baseline_desired_retention)
        if not _matches_float(baseline_dr, dr_values):
            continue

        lambda_value = _path_token_float(path, "lambda_")
        metadata = policy.metadata or {}
        if lambda_value is None and metadata.get("lambda_value") is not None:
            lambda_value = float(metadata["lambda_value"])
        if not _matches_float(lambda_value, lambda_values):
            continue

        entries.append(
            PolicyEntry(
                user_id=user_id,
                baseline_desired_retention=baseline_dr,
                lambda_value=lambda_value,
                path=path.resolve(),
                policy=policy,
            )
        )

    if not entries:
        raise SystemExit(f"No matching SA FSRS-6 policy.json files found under {root}")
    return entries


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


def _lambda_label(value: float | None) -> str:
    if value is None:
        return "none"
    return format_float_token(value)


def _build_surface_z(
    *,
    policy: SAFSRS6Policy,
    s_grid: Sequence[float],
    d_grid: Sequence[float],
) -> list[list[float]]:
    return [
        [policy.evaluate(stability=s_value, difficulty=d_value) for s_value in s_grid]
        for d_value in d_grid
    ]


def _scale_visibility(trace_count: int, visible_index: int | None = None) -> list[bool]:
    if trace_count == 0:
        return []
    if visible_index is not None:
        return [index == visible_index for index in range(trace_count)]
    return [index == trace_count - 1 for index in range(trace_count)]


def _same_bounds(entries: Sequence[PolicyEntry]) -> bool:
    sample = entries[0].policy.bounds
    for entry in entries[1:]:
        bounds = entry.policy.bounds
        if not (
            math.isclose(sample.s_min, bounds.s_min, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(sample.s_max, bounds.s_max, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(sample.d_min, bounds.d_min, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(sample.d_max, bounds.d_max, rel_tol=0.0, abs_tol=1e-9)
        ):
            return False
    return True


def _write_user_plot(
    *,
    user_id: int,
    lambda_value: float | None,
    entries: Sequence[PolicyEntry],
    out_dir: Path,
    s_points: int,
    d_points: int,
    s_min: float | None,
    s_max: float | None,
    d_min: float | None,
    d_max: float | None,
    opacity: float,
) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for this tool. Install dependencies with `uv sync`."
        ) from exc

    if not _same_bounds(entries):
        raise SystemExit(
            f"Policies for user {user_id}, lambda={lambda_value} use different bounds."
        )
    bounds = entries[0].policy.bounds
    resolved_s_min = s_min if s_min is not None else bounds.s_min
    resolved_s_max = s_max if s_max is not None else bounds.s_max
    resolved_d_min = d_min if d_min is not None else bounds.d_min
    resolved_d_max = d_max if d_max is not None else bounds.d_max
    s_grid = _logspace(resolved_s_min, resolved_s_max, s_points)
    d_grid = _linspace(resolved_d_min, resolved_d_max, d_points)

    sorted_entries = sorted(entries, key=lambda entry: entry.baseline_desired_retention)
    dr_values = [entry.baseline_desired_retention for entry in sorted_entries]
    dr_min = min(dr_values)
    dr_max = max(dr_values)
    retention_min = min(entry.policy.retention_min for entry in sorted_entries)
    retention_max = max(entry.policy.retention_max for entry in sorted_entries)

    fig = go.Figure()
    for index, entry in enumerate(sorted_entries):
        dr = entry.baseline_desired_retention
        surface_color = [[dr for _ in s_grid] for _ in d_grid]
        fig.add_trace(
            go.Surface(
                x=s_grid,
                y=d_grid,
                z=_build_surface_z(
                    policy=entry.policy,
                    s_grid=s_grid,
                    d_grid=d_grid,
                ),
                surfacecolor=surface_color,
                cmin=dr_min,
                cmax=dr_max,
                colorscale="Viridis",
                opacity=opacity,
                name=f"DR {dr:.2f}",
                showlegend=True,
                showscale=index == len(sorted_entries) - 1,
                colorbar={
                    "title": {"text": "DR"},
                    "x": 1.05,
                    "xanchor": "left",
                    "y": 0.5,
                    "len": 0.72,
                    "thickness": 16,
                },
                hovertemplate=(
                    "DR=%{surfacecolor:.2f}<br>"
                    "S=%{x:.3g}<br>"
                    "D=%{y:.3g}<br>"
                    "retention=%{z:.3f}<extra></extra>"
                ),
            )
        )

    trace_count = len(sorted_entries)
    buttons = [
        {
            "label": "All DRs",
            "method": "update",
            "args": [
                {
                    "visible": [True for _ in sorted_entries],
                    "showscale": _scale_visibility(trace_count),
                }
            ],
        },
        {
            "label": "Hide all",
            "method": "update",
            "args": [
                {
                    "visible": ["legendonly" for _ in sorted_entries],
                    "showscale": [False for _ in sorted_entries],
                }
            ],
        },
    ]
    for index, entry in enumerate(sorted_entries):
        buttons.append(
            {
                "label": f"DR {entry.baseline_desired_retention:.2f}",
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

    lambda_text = "none" if lambda_value is None else f"{lambda_value:g}"
    fig.update_layout(
        title=(
            f"SA FSRS-6 retention policy surfaces: user {user_id}, lambda={lambda_text}"
        ),
        scene={
            "domain": {"x": [0.08, 0.9], "y": [0.0, 0.96]},
            "xaxis": {"title": "Stability S", "type": "log"},
            "yaxis": {"title": "Difficulty D"},
            "zaxis": {
                "title": "Output retention",
                "range": [retention_min, retention_max],
            },
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
            "title": {"text": "DR surfaces"},
            "x": -0.13,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "itemsizing": "constant",
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        out_dir
        / f"sa_fsrs6_policy_surfaces_user_{user_id}_lambda_{_lambda_label(lambda_value)}.html"
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def _group_entries(
    entries: Iterable[PolicyEntry],
) -> dict[tuple[int, float | None], list[PolicyEntry]]:
    grouped: dict[tuple[int, float | None], list[PolicyEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.user_id, entry.lambda_value)].append(entry)
    return dict(grouped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot SA FSRS-6 policy output retention surfaces by user and DR.",
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy-root",
        type=Path,
        help="Root containing user_*/lambda_*/dr_*/policy.json files.",
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
        "--lambda-values",
        default=None,
        help="Comma-separated lambda values to include. Default: all discovered.",
    )
    parser.add_argument(
        "--dr-values",
        default=None,
        help="Comma-separated baseline DR values to include. Default: all discovered.",
    )
    parser.add_argument("--s-points", type=int, default=48)
    parser.add_argument("--d-points", type=int, default=48)
    parser.add_argument("--s-min", type=float, default=None)
    parser.add_argument("--s-max", type=float, default=None)
    parser.add_argument("--d-min", type=float, default=None)
    parser.add_argument("--d-max", type=float, default=None)
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.45,
        help="Surface opacity for overlaid DR curves.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments")
        / "rl_scheduler"
        / "plots"
        / "sa_fsrs6_policy_surfaces",
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
        dr_values=_parse_csv_floats(args.dr_values),
        lambda_values=_parse_csv_floats(args.lambda_values),
    )
    output_paths = [
        _write_user_plot(
            user_id=user_id,
            lambda_value=lambda_value,
            entries=group_entries,
            out_dir=args.out_dir,
            s_points=args.s_points,
            d_points=args.d_points,
            s_min=args.s_min,
            s_max=args.s_max,
            d_min=args.d_min,
            d_max=args.d_max,
            opacity=args.opacity,
        )
        for (user_id, lambda_value), group_entries in sorted(
            _group_entries(entries).items()
        )
    ]
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
