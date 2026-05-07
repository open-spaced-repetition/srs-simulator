from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.fsrs6_adr_direct_policy import (
    format_float_token,
    parse_float_token,
)
from simulator.fsrs6_adr_delta_policy import FEATURE_VERSION, FSRS6ADRDeltaPolicy


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    user_id: int
    lambda_value: float | None
    path: Path
    policy: FSRS6ADRDeltaPolicy
    discovered_dr_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SurfaceData:
    desired_retention: float
    z: list[list[float]]
    hovertext: list[list[str]]
    z_min: float
    z_max: float


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
    for part in reversed(path.parts):
        if not part.startswith(prefix):
            continue
        try:
            return parse_float_token(part[len(prefix) :])
        except ValueError:
            continue
    return None


def _path_user_id(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.fullmatch(r"user_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def _load_optional_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_mapping(path)


def _is_fsrs6_adr_delta_policy(path: Path) -> bool:
    raw = _load_json_mapping(path)
    return (
        raw.get("feature_version") == FEATURE_VERSION
        or raw.get("policy_kind") == "fsrs6-adr-delta"
    )


def _load_sibling_metadata(path: Path) -> Mapping[str, Any] | None:
    metadata = _load_optional_json_mapping(path.parent / "metadata.json")
    if metadata is None:
        return None
    scheduler_name = metadata.get("scheduler_name")
    if scheduler_name is not None and scheduler_name != "fsrs6_adr_delta":
        raise ValueError(
            f"Artifact metadata for {path} has scheduler_name={scheduler_name!r}; "
            "expected 'fsrs6_adr_delta'."
        )
    policy_path_raw = metadata.get("policy_path")
    if isinstance(policy_path_raw, str) and policy_path_raw.strip():
        metadata_policy_path = Path(policy_path_raw)
        if not metadata_policy_path.is_absolute():
            metadata_policy_path = path.parent / metadata_policy_path
        if metadata_policy_path.resolve() != path.resolve():
            raise ValueError(
                f"Artifact metadata for {path} points to "
                f"{metadata_policy_path}, not {path}."
            )
    return metadata


def _metadata_user_id(metadata: Mapping[str, Any] | None, path: Path) -> int | None:
    if metadata is None or "training_user_ids" not in metadata:
        return None
    raw = metadata["training_user_ids"]
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"metadata training_user_ids must be an array: {path}")
    if len(raw) != 1:
        raise ValueError(
            f"metadata training_user_ids must contain exactly one user for {path}."
        )
    value = raw[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metadata training_user_ids[0] must be an integer: {path}")
    return int(value)


def _metadata_float(
    metadata: Mapping[str, Any] | None,
    key: str,
    path: Path,
) -> float | None:
    if metadata is None or key not in metadata:
        return None
    return _float(metadata[key], f"metadata.{key} for {path}")


def _discover_metrics_dr_values(policy_path: Path) -> tuple[float, ...]:
    metrics = _load_optional_json_mapping(policy_path.parent / "metrics.json")
    if metrics is None:
        return ()

    settings = metrics.get("settings")
    if isinstance(settings, Mapping):
        raw_values = settings.get("baseline_desired_retention_values")
        if raw_values is not None:
            settings_values = _float_sequence(
                raw_values,
                f"{policy_path.parent / 'metrics.json'} settings values",
            )
            if settings_values:
                return _unique_sorted(settings_values)

    raw_per_dr = metrics.get("per_desired_retention")
    if raw_per_dr is None:
        return ()
    if isinstance(raw_per_dr, str) or not isinstance(raw_per_dr, Sequence):
        raise ValueError(
            f"metrics per_desired_retention must be an array: {policy_path}"
        )
    per_dr_values: list[float] = []
    for index, item in enumerate(raw_per_dr):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"metrics per_desired_retention[{index}] must be an object: "
                f"{policy_path}"
            )
        per_dr_values.append(
            _float(
                item.get("baseline_desired_retention"),
                f"per_desired_retention[{index}].baseline_desired_retention",
            )
        )
    return _unique_sorted(per_dr_values)


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
    lambda_values: Sequence[float] | None,
) -> list[PolicyEntry]:
    root = root.expanduser()
    if not root.exists():
        raise SystemExit(f"Policy root not found: {root}")

    user_filter = set(users) if users is not None else None
    entries: list[PolicyEntry] = []
    for path in sorted(root.rglob("policy.json")):
        if not _is_fsrs6_adr_delta_policy(path):
            continue
        policy = FSRS6ADRDeltaPolicy.from_json(path)
        metadata = _load_sibling_metadata(path)
        path_user_id = _path_user_id(path)
        metadata_user_id = _metadata_user_id(metadata, path)
        user_id = metadata_user_id if metadata_user_id is not None else path_user_id
        if user_id is None:
            continue
        if (
            metadata_user_id is not None
            and path_user_id is not None
            and metadata_user_id != path_user_id
        ):
            raise ValueError(
                f"Policy {path} user_id mismatch: path has {path_user_id}, "
                f"metadata has {metadata_user_id}."
            )
        if user_filter is not None and user_id not in user_filter:
            continue
        if start_user is not None and user_id < start_user:
            continue
        if end_user is not None and user_id > end_user:
            continue

        path_lambda = _path_token_float(path, "lambda_")
        metadata_lambda = _metadata_float(metadata, "lambda_value", path)
        lambda_value = metadata_lambda if metadata_lambda is not None else path_lambda
        if (
            metadata_lambda is not None
            and path_lambda is not None
            and not math.isclose(
                metadata_lambda,
                path_lambda,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                f"Policy {path} lambda mismatch: path has {path_lambda}, "
                f"metadata has {metadata_lambda}."
            )
        if not _matches_float(lambda_value, lambda_values):
            continue

        entries.append(
            PolicyEntry(
                user_id=user_id,
                lambda_value=lambda_value,
                path=path.resolve(),
                policy=policy,
                discovered_dr_values=_discover_metrics_dr_values(path),
            )
        )

    if not entries:
        raise SystemExit(
            f"No matching FSRS6 ADR Delta policy.json files found under {root}"
        )
    _reject_duplicate_entries(entries)
    return entries


def _reject_duplicate_entries(entries: Sequence[PolicyEntry]) -> None:
    seen: dict[tuple[int, int | None], Path] = {}
    for entry in entries:
        key = (
            entry.user_id,
            None
            if entry.lambda_value is None
            else round(float(entry.lambda_value) * 1_000_000),
        )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate FSRS6 ADR Delta policies for "
                f"user={entry.user_id}, lambda={entry.lambda_value}: "
                f"{previous} and {entry.path}"
            )
        seen[key] = entry.path


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


def _desired_retention_values(
    *,
    entry: PolicyEntry,
    requested_values: Sequence[float] | None,
    dr_count: int,
) -> tuple[float, ...]:
    if requested_values is not None:
        values = tuple(float(value) for value in requested_values)
        source = "--dr-values"
    elif entry.discovered_dr_values:
        values = entry.discovered_dr_values
        source = "metrics.json"
    else:
        values = tuple(
            _linspace(entry.policy.retention_min, entry.policy.retention_max, dr_count)
        )
        source = "--dr-count"
    if not values:
        raise ValueError("At least one desired retention value is required.")
    for value in values:
        if not (entry.policy.retention_min <= value <= entry.policy.retention_max):
            raise ValueError(
                f"Desired retention {value:g} from {source} is outside policy "
                f"bounds [{entry.policy.retention_min:g}, "
                f"{entry.policy.retention_max:g}] for {entry.path}."
            )
    return _unique_sorted(values)


def _build_surface_data(
    *,
    policy: FSRS6ADRDeltaPolicy,
    s_grid: Sequence[float],
    d_grid: Sequence[float],
    desired_retention: float,
    z_mode: str,
) -> SurfaceData:
    z: list[list[float]] = []
    hovertext: list[list[str]] = []
    z_min = math.inf
    z_max = -math.inf
    for d_value in d_grid:
        z_row: list[float] = []
        hover_row: list[str] = []
        for s_value in s_grid:
            output_retention = policy.evaluate(
                stability=s_value,
                difficulty=d_value,
                desired_retention=desired_retention,
            )
            adjustment = output_retention - desired_retention
            z_value = output_retention if z_mode == "retention" else adjustment
            z_row.append(z_value)
            hover_row.append(
                f"input DR={desired_retention:.2f}<br>"
                f"S={s_value:.3g}<br>"
                f"D={d_value:.3g}<br>"
                f"output retention={output_retention:.3f}<br>"
                f"delta={adjustment:+.4f}"
            )
            z_min = min(z_min, z_value)
            z_max = max(z_max, z_value)
        z.append(z_row)
        hovertext.append(hover_row)
    return SurfaceData(
        desired_retention=desired_retention,
        z=z,
        hovertext=hovertext,
        z_min=z_min,
        z_max=z_max,
    )


def _scale_visibility(trace_count: int, visible_index: int | None = None) -> list[bool]:
    if trace_count == 0:
        return []
    if visible_index is not None:
        return [index == visible_index for index in range(trace_count)]
    return [index == trace_count - 1 for index in range(trace_count)]


def _z_axis_range(
    *,
    z_mode: str,
    policy: FSRS6ADRDeltaPolicy,
    surfaces: Sequence[SurfaceData],
) -> list[float]:
    if z_mode == "retention":
        return [policy.retention_min, policy.retention_max]
    z_min = min(surface.z_min for surface in surfaces)
    z_max = max(surface.z_max for surface in surfaces)
    if math.isclose(z_min, z_max, rel_tol=0.0, abs_tol=1e-12):
        padding = 0.01
    else:
        padding = 0.05 * (z_max - z_min)
    return [z_min - padding, z_max + padding]


def _z_axis_title(z_mode: str) -> str:
    if z_mode == "retention":
        return "Output retention"
    return "Output retention - input DR"


def _transpose_grid(values: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [list(row) for row in zip(*values, strict=True)]


def _write_policy_plot(
    *,
    entry: PolicyEntry,
    out_dir: Path,
    s_points: int,
    d_points: int,
    s_min: float | None,
    s_max: float | None,
    d_min: float | None,
    d_max: float | None,
    desired_retention_values: Sequence[float],
    opacity: float,
    z_mode: str,
) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required for this tool. Install dependencies with `uv sync`."
        ) from exc

    bounds = entry.policy.bounds
    resolved_s_min = s_min if s_min is not None else bounds.s_min
    resolved_s_max = s_max if s_max is not None else bounds.s_max
    resolved_d_min = d_min if d_min is not None else bounds.d_min
    resolved_d_max = d_max if d_max is not None else bounds.d_max
    s_grid = _logspace(resolved_s_min, resolved_s_max, s_points)
    d_grid = _linspace(resolved_d_min, resolved_d_max, d_points)
    x_grid = [list(s_grid) for _ in d_grid]
    y_grid = [[d_value for _ in s_grid] for d_value in d_grid]

    surfaces = [
        _build_surface_data(
            policy=entry.policy,
            s_grid=s_grid,
            d_grid=d_grid,
            desired_retention=desired_retention,
            z_mode=z_mode,
        )
        for desired_retention in desired_retention_values
    ]
    dr_min = min(desired_retention_values)
    dr_max = max(desired_retention_values)

    fig = go.Figure()
    for index, surface in enumerate(surfaces):
        dr = surface.desired_retention
        surface_color = [[dr for _ in s_grid] for _ in d_grid]
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=surface.z,
                customdata=_transpose_grid(surface.hovertext),
                surfacecolor=surface_color,
                cmin=dr_min,
                cmax=dr_max,
                colorscale="Viridis",
                opacity=opacity,
                name=f"DR {dr:.2f}",
                showlegend=True,
                showscale=index == len(surfaces) - 1,
                colorbar={
                    "title": {"text": "Input DR"},
                    "x": 1.05,
                    "xanchor": "left",
                    "y": 0.5,
                    "len": 0.72,
                    "thickness": 16,
                },
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    trace_count = len(surfaces)
    buttons = [
        {
            "label": "All DRs",
            "method": "update",
            "args": [
                {
                    "visible": [True for _ in surfaces],
                    "showscale": _scale_visibility(trace_count),
                }
            ],
        },
        {
            "label": "Hide all",
            "method": "update",
            "args": [
                {
                    "visible": ["legendonly" for _ in surfaces],
                    "showscale": [False for _ in surfaces],
                }
            ],
        },
    ]
    for index, surface in enumerate(surfaces):
        buttons.append(
            {
                "label": f"DR {surface.desired_retention:.2f}",
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

    lambda_text = "none" if entry.lambda_value is None else f"{entry.lambda_value:g}"
    fig.update_layout(
        title=(
            "FSRS6 ADR Delta-conditioned retention policy surfaces: "
            f"user {entry.user_id}, lambda={lambda_text}"
        ),
        scene={
            "domain": {"x": [0.08, 0.9], "y": [0.0, 0.96]},
            "xaxis": {"title": "Stability S", "type": "log"},
            "yaxis": {"title": "Difficulty D"},
            "zaxis": {
                "title": _z_axis_title(z_mode),
                "range": _z_axis_range(
                    z_mode=z_mode,
                    policy=entry.policy,
                    surfaces=surfaces,
                ),
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
            "title": {"text": "DR slices"},
            "x": -0.13,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "itemsizing": "constant",
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        out_dir / "fsrs6_adr_delta_policy_surfaces_"
        f"user_{entry.user_id}_lambda_{_lambda_label(entry.lambda_value)}_"
        f"z_{z_mode}.html"
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def _unique_sorted(values: Sequence[float]) -> tuple[float, ...]:
    unique: list[float] = []
    for value in sorted(float(item) for item in values):
        if not any(
            math.isclose(value, previous, rel_tol=0.0, abs_tol=1e-9)
            for previous in unique
        ):
            unique.append(value)
    return tuple(unique)


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _float_sequence(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return tuple(
        _float(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FSRS6 ADR Delta-conditioned policy output surfaces by user, "
            "lambda, and input desired retention."
        ),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--policy-root",
        type=Path,
        help="Root containing user_*/lambda_*/policy.json files.",
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
        help=(
            "Comma-separated input desired retention values to plot. Default: "
            "metrics.json training grid when available, otherwise an even grid."
        ),
    )
    parser.add_argument(
        "--dr-count",
        type=int,
        default=9,
        help="Fallback number of evenly spaced DR slices when metrics are absent.",
    )
    parser.add_argument(
        "--z-mode",
        choices=("retention", "adjustment"),
        default="retention",
        help="Plot output retention or output-minus-input adjustment on the z axis.",
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
        help="Surface opacity for overlaid DR slices.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments")
        / "rl_scheduler"
        / "plots"
        / "fsrs6_adr_delta_policy_surfaces",
        help="Directory for generated HTML plots.",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.opacity <= 1.0):
        raise SystemExit("--opacity must satisfy 0 < opacity <= 1.")
    if args.s_points < 2 or args.d_points < 2:
        raise SystemExit("--s-points and --d-points must be >= 2.")
    if args.dr_count < 2:
        raise SystemExit("--dr-count must be >= 2.")
    if args.users is not None and (
        args.start_user is not None or args.end_user is not None
    ):
        raise SystemExit("--users cannot be combined with --start-user/--end-user.")

    requested_dr_values = _parse_csv_floats(args.dr_values)
    if requested_dr_values is not None and not requested_dr_values:
        raise SystemExit("--dr-values must contain at least one value.")
    policy_root = _resolve_policy_root(args)
    entries = _discover_policies(
        policy_root,
        users=_parse_csv_ints(args.users),
        start_user=args.start_user,
        end_user=args.end_user,
        lambda_values=_parse_csv_floats(args.lambda_values),
    )
    output_paths = []
    for entry in sorted(
        entries,
        key=lambda item: (
            item.user_id,
            math.inf if item.lambda_value is None else item.lambda_value,
        ),
    ):
        output_paths.append(
            _write_policy_plot(
                entry=entry,
                out_dir=args.out_dir,
                s_points=args.s_points,
                d_points=args.d_points,
                s_min=args.s_min,
                s_max=args.s_max,
                d_min=args.d_min,
                d_max=args.d_max,
                desired_retention_values=_desired_retention_values(
                    entry=entry,
                    requested_values=requested_dr_values,
                    dr_count=args.dr_count,
                ),
                opacity=args.opacity,
                z_mode=args.z_mode,
            )
        )
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
