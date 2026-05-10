from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.fsrs6_adr_policy import (  # noqa: E402
    format_float_token,
    parse_float_token,
)
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy  # noqa: E402

PlotMode = Literal["baseline_dr", "memorization_ratio"]


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    user_id: int
    baseline_desired_retention: float | None
    lambda_value: float | None
    path: Path
    policy: FSRS6ADRPolicy
    portfolio_index: int | None = None
    memorized_average: float | None = None
    deck_size: int | None = None
    memorization_ratio: float | None = None

    @property
    def plot_mode(self) -> PlotMode:
        if self.baseline_desired_retention is not None:
            return "baseline_dr"
        return "memorization_ratio"


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


def _path_portfolio_index(path: Path) -> int | None:
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


def _float_field(raw: Any, field: str, source: Path) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise SystemExit(f"{source} field {field!r} must be a number.")
    return float(raw)


def _int_field(raw: Any, field: str, source: Path) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SystemExit(f"{source} field {field!r} must be an integer.")
    return int(raw)


def _optional_float_field(raw: Any, field: str, source: Path) -> float | None:
    if raw is None:
        return None
    return _float_field(raw, field, source)


def _optional_int_field(raw: Any, field: str, source: Path) -> int | None:
    if raw is None:
        return None
    return _int_field(raw, field, source)


def _resolve_relative_path(raw: Any, *, base: Path, field: str, source: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"{source} field {field!r} must be a non-empty string.")
    resolved = Path(raw)
    if resolved.is_absolute():
        return resolved
    return base / resolved


def _metadata_user_id(metadata: Mapping[str, Any]) -> int | None:
    raw_user_id = metadata.get("user_id")
    if raw_user_id is not None:
        return int(raw_user_id)
    raw_training_user_ids = metadata.get("training_user_ids")
    if isinstance(raw_training_user_ids, list) and len(raw_training_user_ids) == 1:
        return int(raw_training_user_ids[0])
    return None


def _config_snapshot_path(
    metadata_path: Path, metadata: Mapping[str, Any]
) -> Path | None:
    raw_path = metadata.get("config_snapshot_path")
    if raw_path is None:
        return None
    return _resolve_relative_path(
        raw_path,
        base=metadata_path.parent,
        field="config_snapshot_path",
        source=metadata_path,
    )


def _deck_size_from_config(config_path: Path) -> int:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    simulation = raw.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(f"{config_path} must contain [simulation].")
    raw_deck = simulation.get("deck")
    if isinstance(raw_deck, bool) or not isinstance(raw_deck, int):
        raise ValueError(f"{config_path} [simulation].deck must be an integer.")
    if raw_deck <= 0:
        raise ValueError(f"{config_path} [simulation].deck must be positive.")
    return int(raw_deck)


def _portfolio_deck_size(
    *,
    policy_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    fallback_deck_size: int | None,
) -> int:
    config_path = _config_snapshot_path(metadata_path, metadata)
    config_error: str | None = None
    if config_path is not None:
        try:
            return _deck_size_from_config(config_path)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            config_error = str(exc)
    if fallback_deck_size is not None:
        return fallback_deck_size

    message = (
        f"Policy {policy_path} has null baseline_desired_retention and needs "
        "deck size to compute memorized_average / deck_size. "
    )
    if config_path is None:
        message += "Provide --deck-size or metadata.json config_snapshot_path."
    else:
        message += f"Could not read deck from {config_path}: {config_error}"
    raise SystemExit(message)


def _portfolio_metrics(
    *,
    policy_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    fallback_deck_size: int | None,
) -> tuple[int | None, float, int, float]:
    raw_metrics_path = metadata.get("metrics_path", "metrics.json")
    metrics_path = _resolve_relative_path(
        raw_metrics_path,
        base=policy_path.parent,
        field="metrics_path",
        source=metadata_path,
    )
    try:
        metrics_payload = _read_json_object(metrics_path)
    except OSError as exc:
        raise SystemExit(
            f"Could not read portfolio metrics {metrics_path}: {exc}"
        ) from exc
    metrics = metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"{metrics_path} field 'metrics' must be a JSON object.")
    raw_memorized = metrics.get("memorized_average")
    if raw_memorized is None:
        raw_memorized = metrics.get("average_memorization")
    memorized_average = _float_field(
        raw_memorized,
        "metrics.memorized_average",
        metrics_path,
    )
    deck_size = _portfolio_deck_size(
        policy_path=policy_path,
        metadata_path=metadata_path,
        metadata=metadata,
        fallback_deck_size=fallback_deck_size,
    )
    if deck_size <= 0:
        raise SystemExit("--deck-size must be positive.")
    portfolio_index = _optional_int_field(
        metadata.get("portfolio_index", metrics_payload.get("portfolio_index")),
        "portfolio_index",
        metadata_path,
    )
    if portfolio_index is None:
        portfolio_index = _path_portfolio_index(policy_path)
    return (
        portfolio_index,
        memorized_average,
        deck_size,
        memorized_average / deck_size,
    )


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
    deck_size: int | None,
) -> list[PolicyEntry]:
    if not root.exists():
        raise SystemExit(f"Policy root not found: {root}")

    user_filter = set(users) if users is not None else None
    entries: list[PolicyEntry] = []
    for path in sorted(root.rglob("policy.json")):
        policy = FSRS6ADRPolicy.from_json(path)
        metadata_path = path.with_name("metadata.json")
        artifact_metadata = _read_optional_json_object(metadata_path)
        user_id = _path_user_id(path)
        if user_id is None:
            user_id = _metadata_user_id(policy.metadata or {})
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

        baseline_dr = _path_token_float(path, "dr_")
        if baseline_dr is None:
            baseline_dr = policy.baseline_desired_retention
        if (
            baseline_dr is None
            and artifact_metadata.get("baseline_desired_retention") is not None
        ):
            baseline_dr = _optional_float_field(
                artifact_metadata.get("baseline_desired_retention"),
                "baseline_desired_retention",
                metadata_path,
            )
        if baseline_dr is not None:
            baseline_dr = float(baseline_dr)
            if not _matches_float(baseline_dr, dr_values):
                continue
        elif dr_values is not None:
            continue

        lambda_value = _path_token_float(path, "lambda_")
        if lambda_value is None and artifact_metadata.get("lambda_value") is not None:
            lambda_value = float(artifact_metadata["lambda_value"])
        policy_metadata = policy.metadata or {}
        if lambda_value is None and policy_metadata.get("lambda_value") is not None:
            lambda_value = float(policy_metadata["lambda_value"])
        if not _matches_float(lambda_value, lambda_values):
            continue

        portfolio_index = None
        memorized_average = None
        resolved_deck_size = None
        memorization_ratio = None
        if baseline_dr is None:
            (
                portfolio_index,
                memorized_average,
                resolved_deck_size,
                memorization_ratio,
            ) = _portfolio_metrics(
                policy_path=path,
                metadata_path=metadata_path,
                metadata=artifact_metadata,
                fallback_deck_size=deck_size,
            )

        entries.append(
            PolicyEntry(
                user_id=user_id,
                baseline_desired_retention=baseline_dr,
                lambda_value=lambda_value,
                path=path.resolve(),
                policy=policy,
                portfolio_index=portfolio_index,
                memorized_average=memorized_average,
                deck_size=resolved_deck_size,
                memorization_ratio=memorization_ratio,
            )
        )

    if not entries:
        raise SystemExit(f"No matching FSRS6 ADR policy.json files found under {root}")
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
    policy: FSRS6ADRPolicy,
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


def _required_float(value: float | None, *, field: str, path: Path) -> float:
    if value is None:
        raise SystemExit(f"Policy {path} is missing required {field}.")
    return value


def _portfolio_label(entry: PolicyEntry) -> str:
    if entry.portfolio_index is not None:
        return f"Policy {entry.portfolio_index}"
    return entry.path.parent.name


def _portfolio_customdata(
    entry: PolicyEntry, s_count: int, d_count: int
) -> list[list[list[Any]]]:
    portfolio_index = "" if entry.portfolio_index is None else entry.portfolio_index
    memorized_average = _required_float(
        entry.memorized_average,
        field="memorized_average",
        path=entry.path,
    )
    deck_size = _required_float(
        float(entry.deck_size) if entry.deck_size is not None else None,
        field="deck_size",
        path=entry.path,
    )
    ratio = _required_float(
        entry.memorization_ratio,
        field="memorization_ratio",
        path=entry.path,
    )
    row = [
        [_portfolio_label(entry), portfolio_index, memorized_average, deck_size, ratio]
        for _ in range(s_count)
    ]
    return [[list(item) for item in row] for _ in range(d_count)]


def _sorted_entries_for_mode(
    entries: Sequence[PolicyEntry], mode: PlotMode
) -> list[PolicyEntry]:
    if mode == "baseline_dr":
        return sorted(
            entries,
            key=lambda entry: _required_float(
                entry.baseline_desired_retention,
                field="baseline_desired_retention",
                path=entry.path,
            ),
        )
    return sorted(
        entries,
        key=lambda entry: (
            _required_float(
                entry.memorization_ratio,
                field="memorization_ratio",
                path=entry.path,
            ),
            entry.portfolio_index if entry.portfolio_index is not None else 10**9,
            str(entry.path),
        ),
    )


def _color_values_for_mode(
    entries: Sequence[PolicyEntry], mode: PlotMode
) -> list[float]:
    field = (
        "baseline_desired_retention" if mode == "baseline_dr" else "memorization_ratio"
    )
    return [
        _required_float(getattr(entry, field), field=field, path=entry.path)
        for entry in entries
    ]


def _write_user_plot(
    *,
    user_id: int,
    lambda_value: float | None,
    mode: PlotMode,
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
    if any(entry.plot_mode != mode for entry in entries):
        raise SystemExit(
            f"Policies for user {user_id}, lambda={lambda_value} mix plot modes."
        )
    bounds = entries[0].policy.bounds
    resolved_s_min = s_min if s_min is not None else bounds.s_min
    resolved_s_max = s_max if s_max is not None else bounds.s_max
    resolved_d_min = d_min if d_min is not None else bounds.d_min
    resolved_d_max = d_max if d_max is not None else bounds.d_max
    s_grid = _logspace(resolved_s_min, resolved_s_max, s_points)
    d_grid = _linspace(resolved_d_min, resolved_d_max, d_points)

    sorted_entries = _sorted_entries_for_mode(entries, mode)
    color_values = _color_values_for_mode(sorted_entries, mode)
    if mode == "baseline_dr":
        color_title = "DR"
        all_label = "All DRs"
        legend_title = "DR surfaces"
        output_suffix = ""
    else:
        color_title = "memorized/deck"
        all_label = "All policies"
        legend_title = "Policy surfaces"
        output_suffix = "_portfolio"
    color_min = min(color_values)
    color_max = max(color_values)
    retention_min = min(entry.policy.retention_min for entry in sorted_entries)
    retention_max = max(entry.policy.retention_max for entry in sorted_entries)

    fig = go.Figure()
    for index, (entry, color_value) in enumerate(
        zip(sorted_entries, color_values, strict=True)
    ):
        surface_color = [[color_value for _ in s_grid] for _ in d_grid]
        surface_kwargs: dict[str, Any] = {
            "x": s_grid,
            "y": d_grid,
            "z": _build_surface_z(
                policy=entry.policy,
                s_grid=s_grid,
                d_grid=d_grid,
            ),
            "surfacecolor": surface_color,
            "cmin": color_min,
            "cmax": color_max,
            "colorscale": "Viridis",
            "opacity": opacity,
            "showlegend": True,
            "showscale": index == len(sorted_entries) - 1,
            "colorbar": {
                "title": {"text": color_title},
                "x": 1.05,
                "xanchor": "left",
                "y": 0.5,
                "len": 0.72,
                "thickness": 16,
            },
        }
        if mode == "baseline_dr":
            surface_kwargs.update(
                {
                    "name": f"DR {color_value:.2f}",
                    "hovertemplate": (
                        "DR=%{surfacecolor:.2f}<br>"
                        "S=%{x:.3g}<br>"
                        "D=%{y:.3g}<br>"
                        "retention=%{z:.3f}<extra></extra>"
                    ),
                }
            )
        else:
            label = _portfolio_label(entry)
            surface_kwargs.update(
                {
                    "name": f"{label} ({color_value:.4f})",
                    "customdata": _portfolio_customdata(
                        entry,
                        s_count=len(s_grid),
                        d_count=len(d_grid),
                    ),
                    "hovertemplate": (
                        "policy=%{customdata[0]}<br>"
                        "policy_index=%{customdata[1]}<br>"
                        "memorized_average=%{customdata[2]:.3f}<br>"
                        "deck_size=%{customdata[3]:.0f}<br>"
                        "memorized/deck=%{surfacecolor:.4f}<br>"
                        "S=%{x:.3g}<br>"
                        "D=%{y:.3g}<br>"
                        "retention=%{z:.3f}<extra></extra>"
                    ),
                }
            )
        fig.add_trace(go.Surface(**surface_kwargs))

    trace_count = len(sorted_entries)
    buttons = [
        {
            "label": all_label,
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
    for index, (entry, color_value) in enumerate(
        zip(sorted_entries, color_values, strict=True)
    ):
        if mode == "baseline_dr":
            button_label = f"DR {color_value:.2f}"
        else:
            button_label = f"{_portfolio_label(entry)} {color_value:.4f}"
        buttons.append(
            {
                "label": button_label,
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
    title_prefix = (
        "FSRS6 ADR retention policy surfaces"
        if mode == "baseline_dr"
        else "FSRS6 ADR portfolio retention policy surfaces"
    )
    fig.update_layout(
        title=(f"{title_prefix}: user {user_id}, lambda={lambda_text}"),
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
            "title": {"text": legend_title},
            "x": -0.13,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "itemsizing": "constant",
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / (
        f"fsrs6_adr_policy_surfaces_user_{user_id}_"
        f"lambda_{_lambda_label(lambda_value)}{output_suffix}.html"
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def _group_entries(
    entries: Iterable[PolicyEntry],
) -> dict[tuple[int, float | None, PlotMode], list[PolicyEntry]]:
    grouped: dict[tuple[int, float | None, PlotMode], list[PolicyEntry]] = defaultdict(
        list
    )
    for entry in entries:
        grouped[(entry.user_id, entry.lambda_value, entry.plot_mode)].append(entry)
    return dict(grouped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot FSRS6 ADR policy output retention surfaces by user and DR.",
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
    parser.add_argument(
        "--deck-size",
        type=int,
        default=None,
        help=(
            "Fallback deck size for null-baseline ADR portfolio policies when "
            "metadata config_snapshot_path is unavailable."
        ),
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
        / "fsrs6_adr_policy_surfaces",
        help="Directory for generated HTML plots.",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.opacity <= 1.0):
        raise SystemExit("--opacity must satisfy 0 < opacity <= 1.")
    if args.s_points < 2 or args.d_points < 2:
        raise SystemExit("--s-points and --d-points must be >= 2.")
    if args.deck_size is not None and args.deck_size <= 0:
        raise SystemExit("--deck-size must be positive.")
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
        deck_size=args.deck_size,
    )
    output_paths = [
        _write_user_plot(
            user_id=user_id,
            lambda_value=lambda_value,
            mode=mode,
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
        for (user_id, lambda_value, mode), group_entries in sorted(
            _group_entries(entries).items(),
            key=lambda item: (
                item[0][0],
                math.inf if item[0][1] is None else item[0][1],
                item[0][2],
            ),
        )
    ]
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
