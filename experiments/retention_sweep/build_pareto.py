from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert simulate.py logs into a Pareto frontier plot.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing simulate.py JSONL logs.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        default="lstm",
        help="Comma-separated list of environments to compare.",
    )
    parser.add_argument(
        "--sched",
        dest="sched",
        default="fsrs6",
        help=(
            "Comma-separated list of schedulers to plot "
            "(include sspmmc for policies; use fixed@<days> for fixed intervals "
            "or fixed to include all fixed intervals)."
        ),
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="User ID used for default log directory resolution.",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=0.50,
        help="Minimum desired retention to include.",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=0.98,
        help="Maximum desired retention to include.",
    )
    parser.add_argument(
        "--fuzz",
        choices=["on", "off", "any"],
        default="any",
        help="Filter logs by fuzz flag (ignored when --compare-fuzz is set).",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="any",
        help="Filter logs by short-term flag (ignored when --short-term-source is set).",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched", "any"],
        default="any",
        help="Filter logs by short-term source (steps/sched).",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized", "any"],
        default="any",
        help="Filter logs by simulation engine.",
    )
    parser.add_argument(
        "--compare-fuzz",
        action="store_true",
        help="Plot fuzz on/off as separate series when logs include meta.fuzz.",
    )
    parser.add_argument(
        "--compare-short-term",
        action="store_true",
        help="Plot short-term on/off as separate series when logs include meta.short_term.",
    )
    parser.add_argument(
        "--compare-engine",
        action="store_true",
        help="Plot event/vectorized as separate series when logs include meta.engine.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to write simulation_results.json.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save the Pareto frontier plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and only write the results JSON.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide point annotations for scheduler configurations.",
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _load_meta_totals(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = None
    totals = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload.get("data")
            elif payload.get("type") == "totals":
                totals = payload.get("data")
            if meta is not None and totals is not None:
                break
    if meta is None or totals is None:
        raise ValueError(f"Missing meta or totals in {path}")
    return meta, totals


def _resolve_policy_title(
    meta: Dict[str, Any],
    base_dirs: Sequence[Path],
) -> Optional[str]:
    policy_path = meta.get("sspmmc_policy")
    if not policy_path:
        return None
    path = Path(policy_path)
    if not path.is_absolute():
        for base_dir in base_dirs:
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                path = candidate
                break
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            title = payload.get("title")
            if isinstance(title, str) and title.strip():
                return _resolve_sspmmc_label(title.strip(), path.stem)
        except (OSError, json.JSONDecodeError):
            return _resolve_sspmmc_label(None, path.stem)
    return _resolve_sspmmc_label(None, path.stem)


def _resolve_sspmmc_label(title: Optional[str], fallback: str) -> Optional[str]:
    label_map = {
        "balanced": "Balanced",
        "maximum efficiency": "Efficiency",
        "maximum_efficiency": "Efficiency",
        "max efficiency": "Efficiency",
        "max_efficiency": "Efficiency",
        "maximum knowledge": "Knowledge",
        "maximum_knowledge": "Knowledge",
        "max knowledge": "Knowledge",
        "max_knowledge": "Knowledge",
    }
    combined = " ".join(part for part in [title or "", fallback] if part).lower()
    for key, label in label_map.items():
        if key in combined:
            return label
    return None


def _format_scheduler_title(scheduler: str) -> str:
    labels = {
        "anki_sm2": "Anki-SM-2",
        "memrise": "Memrise",
    }
    return labels.get(scheduler, scheduler)


def _iter_log_entries(
    log_dir: Path,
    environment: str,
    scheduler_filter: Optional[set[str]],
    min_retention: float,
    max_retention: float,
    base_dirs: Sequence[Path],
    fuzz_filter: Optional[bool],
    short_term_filter: Optional[bool],
    short_term_source_filter: Optional[str],
    engine_filter: Optional[str],
    fixed_interval_filter: Optional[Sequence[float]] = None,
) -> Iterable[Tuple[Optional[float], Dict[str, Any]]]:
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            meta, totals = _load_meta_totals(path)
        except ValueError:
            continue

        if meta.get("environment") != environment:
            continue
        scheduler = meta.get("scheduler")
        if scheduler_filter and scheduler not in scheduler_filter:
            continue

        fuzz_value = _normalize_bool(meta.get("fuzz"))
        if fuzz_filter is not None:
            if fuzz_value is None or fuzz_value != fuzz_filter:
                continue

        short_term_value = _normalize_bool(meta.get("short_term"))
        if short_term_filter is True:
            if short_term_value is not True:
                continue
        elif short_term_filter is False:
            if short_term_value is True:
                continue

        if short_term_source_filter is not None:
            source_value = meta.get("short_term_source")
            if source_value != short_term_source_filter:
                continue

        engine_value = meta.get("engine")
        if engine_filter is not None:
            if engine_value != engine_filter:
                continue

        fixed_interval = None
        if scheduler == "fixed":
            raw_interval = meta.get("fixed_interval")
            fixed_interval = normalize_fixed_interval(
                float(raw_interval) if raw_interval is not None else None
            )
            if fixed_interval_filter:
                if not any(
                    math.isclose(fixed_interval, value, rel_tol=0.0, abs_tol=1e-6)
                    for value in fixed_interval_filter
                ):
                    continue

        desired = meta.get("desired_retention")
        if scheduler_uses_desired_retention(scheduler):
            desired_value = float(desired or 0.0)
            if desired_value < min_retention or desired_value > max_retention:
                continue
        else:
            desired_value = None

        time_average = totals.get("time_average")
        if time_average is None:
            continue
        time_average = float(time_average)
        if time_average <= 0:
            continue

        if scheduler == "sspmmc":
            title = _resolve_policy_title(meta, base_dirs)
        elif scheduler == "fixed":
            title = f"Ivl={format_float(fixed_interval)}"
        elif scheduler_uses_desired_retention(scheduler):
            title = f"DR={format_float(float(desired_value) * 100)}%"
        else:
            title = _format_scheduler_title(scheduler)

        user_id = meta.get("user_id")
        if user_id is not None:
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                user_id = None

        memorized_average = float(totals.get("memorized_average", 0.0))
        entry = {
            "title": title,
            "reviews_average": float(totals.get("reviews_average", 0.0)),
            "time_average": time_average,
            "memorized_average": memorized_average,
            "memorized_per_minute": memorized_average / time_average,
            "avg_accum_memorized_per_hour": float(
                totals.get("avg_accum_memorized_per_hour", 0.0)
            ),
            "scheduler": scheduler,
            "user_id": user_id,
            "fixed_interval": fixed_interval,
            "fuzz": fuzz_value,
            "short_term": short_term_value,
            "short_term_source": meta.get("short_term_source"),
            "engine": engine_value,
        }
        yield desired_value, entry


def _build_results(
    log_dir: Path,
    environment: str,
    scheduler_filter: Optional[set[str]],
    min_retention: float,
    max_retention: float,
    base_dirs: Sequence[Path],
    fuzz_filter: Optional[bool],
    short_term_filter: Optional[bool],
    short_term_source_filter: Optional[str],
    engine_filter: Optional[str],
    fixed_interval_filter: Optional[Sequence[float]] = None,
    title_prefix: str | None = None,
    dedupe: bool = True,
    dedupe_fuzz: bool = False,
    dedupe_short_term: bool = False,
    dedupe_engine: bool = False,
) -> List[Dict[str, Any]]:
    by_retention: Dict[object, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for desired, entry in _iter_log_entries(
        log_dir,
        environment,
        scheduler_filter,
        min_retention,
        max_retention,
        base_dirs,
        fuzz_filter,
        short_term_filter,
        short_term_source_filter,
        engine_filter,
        fixed_interval_filter,
    ):
        if title_prefix:
            entry["title"] = f"{title_prefix} {entry['title']}"
        entry["environment"] = environment
        if dedupe and desired is not None:
            if dedupe_fuzz or dedupe_short_term or dedupe_engine:
                key = (
                    desired,
                    entry.get("fuzz") if dedupe_fuzz else None,
                    entry.get("short_term") if dedupe_short_term else None,
                    entry.get("engine") if dedupe_engine else None,
                )
                by_retention[key] = entry
            else:
                by_retention[desired] = entry
        else:
            results.append(entry)

    if dedupe:
        deduped = [by_retention[ret] for ret in sorted(by_retention)]
        return deduped + results
    return results


def _format_title(title_base: str, user_ids: Sequence[int]) -> str:
    if user_ids:
        if len(user_ids) == 1:
            return f"{title_base} (user {user_ids[0]})"
        return "{} (users {})".format(
            title_base, ", ".join(str(uid) for uid in user_ids)
        )
    return title_base


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _plot_compare_frontier(
    series: List[Dict[str, Any]],
    output_path: Path,
    title_base: str = "Pareto frontier comparison",
    show_labels: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if show_labels:
        try:
            from adjustText import adjust_text
        except ImportError as exc:
            raise SystemExit(
                "adjustText is required for --show-labels. "
                "Install it with `uv add adjustText`."
            ) from exc

    all_entries = [entry for item in series for entry in item["entries"]]
    if not all_entries:
        raise ValueError("No entries available to plot.")

    min_x = min(entry["memorized_average"] for entry in all_entries)
    max_x = max(entry["memorized_average"] for entry in all_entries)
    max_y = max(entry["memorized_per_minute"] for entry in all_entries)

    x_min = 200 * math.floor(min_x / 200) if min_x else 0
    x_max = 200 * math.ceil(max_x / 200) if max_x else 1
    y_min = 0
    y_max = max_y * 1.03 if max_y else 1

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    scheduler_order: List[str] = []
    environment_order: List[str] = []
    env_fuzz_order: List[tuple[Optional[str], Optional[bool]]] = []
    env_fuzz_short_engine_order: List[
        tuple[Optional[str], Optional[bool], Optional[bool], Optional[str]]
    ] = []
    use_fuzz_linestyles = any(item.get("fuzz") is not None for item in series)
    use_short_term_linestyles = any(
        item.get("short_term") is not None for item in series
    )
    use_engine_linestyles = any(item.get("engine") is not None for item in series)
    for item in series:
        scheduler = item.get("scheduler")
        environment = item.get("environment")
        fuzz_value = item.get("fuzz")
        short_term_value = item.get("short_term")
        engine_value = item.get("engine")
        if scheduler and scheduler not in scheduler_order:
            scheduler_order.append(scheduler)
        if use_fuzz_linestyles or use_short_term_linestyles or use_engine_linestyles:
            key = (
                environment,
                fuzz_value if use_fuzz_linestyles else None,
                short_term_value if use_short_term_linestyles else None,
                engine_value if use_engine_linestyles else None,
            )
            if key not in env_fuzz_short_engine_order:
                env_fuzz_short_engine_order.append(key)
        elif use_fuzz_linestyles:
            key = (environment, fuzz_value)
            if key not in env_fuzz_order:
                env_fuzz_order.append(key)
        elif environment and environment not in environment_order:
            environment_order.append(environment)
    scheduler_colors = {
        scheduler: colors[idx % len(colors)]
        for idx, scheduler in enumerate(scheduler_order)
    }
    environment_linestyles: dict[object, object] = {}
    if use_fuzz_linestyles or use_short_term_linestyles or use_engine_linestyles:
        environment_linestyles = {
            key: linestyles[idx % len(linestyles)]
            for idx, key in enumerate(env_fuzz_short_engine_order)
        }
    elif use_fuzz_linestyles:
        environment_linestyles = {
            key: linestyles[idx % len(linestyles)]
            for idx, key in enumerate(env_fuzz_order)
        }
    else:
        environment_linestyles = {
            environment: linestyles[idx % len(linestyles)]
            for idx, environment in enumerate(environment_order)
        }

    single_point_markers = {
        "anki_sm2": "^",
        "memrise": "s",
    }

    def _select_label_indices(count: int, max_labels: int) -> List[int]:
        if count <= 0:
            return []
        if count <= max_labels:
            return list(range(count))
        step = max(1, math.ceil(count / max_labels))
        indices = list(range(0, count, step))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        return indices

    plt.figure(figsize=(12, 9))
    non_empty_series = [item for item in series if item["entries"]]
    max_labels_total = 40
    max_labels_per_series = max(2, max_labels_total // max(1, len(non_empty_series)))
    texts = []
    label_points_x = []
    label_points_y = []
    avoid_x = []
    avoid_y = []

    for item in series:
        entries = item["entries"]
        if not entries:
            continue
        scheduler = item.get("scheduler")
        environment = item.get("environment")
        fuzz_value = item.get("fuzz")
        if use_fuzz_linestyles or use_short_term_linestyles or use_engine_linestyles:
            style_key = (
                environment,
                fuzz_value if use_fuzz_linestyles else None,
                item.get("short_term") if use_short_term_linestyles else None,
                item.get("engine") if use_engine_linestyles else None,
            )
        else:
            style_key = environment
        is_single_point = scheduler in single_point_markers and len(entries) == 1
        if is_single_point:
            color = "red"
            linestyle = ""
            marker = single_point_markers.get(scheduler, "^")
            linewidth = 2
            markersize = 7.5
        else:
            color = scheduler_colors.get(scheduler, colors[0])
            linestyle = environment_linestyles.get(style_key, linestyles[0])
            marker = "o"
            linewidth = 2
            markersize = 6
        markerfacecolor = None
        markeredgecolor = None
        if is_single_point and use_short_term_linestyles:
            short_term_value = item.get("short_term")
            if short_term_value is not None:
                markerfacecolor = color if short_term_value else "none"
                markeredgecolor = color
        x_vals = [entry["memorized_average"] for entry in entries]
        y_vals = [entry["memorized_per_minute"] for entry in entries]
        avoid_x.extend(x_vals)
        avoid_y.extend(y_vals)
        if len(x_vals) > 1:
            for x0, y0, x1, y1 in zip(x_vals[:-1], y_vals[:-1], x_vals[1:], y_vals[1:]):
                avoid_x.append(x0 + (x1 - x0) * 0.33)
                avoid_y.append(y0 + (y1 - y0) * 0.33)
                avoid_x.append(x0 + (x1 - x0) * 0.66)
                avoid_y.append(y0 + (y1 - y0) * 0.66)
        plt.plot(
            x_vals,
            y_vals,
            label=item["label"],
            linestyle=linestyle,
            marker=marker,
            color=color,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            linewidth=linewidth,
            markersize=markersize,
        )
        if show_labels and scheduler != "sspmmc":
            label_indices = _select_label_indices(len(entries), max_labels_per_series)
            for entry_idx in label_indices:
                entry = entries[entry_idx]
                title = entry.get("title")
                if not title:
                    continue
                texts.append(
                    plt.text(
                        x_vals[entry_idx],
                        y_vals[entry_idx],
                        title,
                        fontsize=9,
                        color="black",
                    )
                )
                label_points_x.append(x_vals[entry_idx])
                label_points_y.append(y_vals[entry_idx])

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    if show_labels and texts:
        import logging
        from matplotlib.patches import FancyArrowPatch

        adjust_logger = logging.getLogger("adjustText")
        previous_level = adjust_logger.level
        adjust_logger.setLevel(logging.ERROR)
        try:
            adjust_text(
                texts,
                x=avoid_x,
                y=avoid_y,
                target_x=label_points_x,
                target_y=label_points_y,
                expand=(1.02, 1.02),
                force_static=(0.05, 0.05),
                force_text=(0.1, 0.1),
                force_pull=(0.06, 0.06),
                max_move=(5, 5),
                lim=200,
            )
        finally:
            adjust_logger.setLevel(previous_level)
        ax = plt.gca()
        for text, target_x, target_y in zip(texts, label_points_x, label_points_y):
            arrow = FancyArrowPatch(
                posA=text.get_position(),
                posB=(target_x, target_y),
                patchA=text,
                transform=ax.transData,
                arrowstyle="-",
                color="gray",
                lw=0.5,
                shrinkA=8,
                shrinkB=4,
            )
            ax.add_patch(arrow)
    plt.xlabel(
        "Memorized cards (average, all days)\n(higher=better)",
        fontsize=18,
        color="black",
    )
    plt.ylabel(
        "Memorized cards (average)/Minutes of studying per day\n(higher=better)",
        fontsize=18,
        color="black",
    )
    plt.xticks(fontsize=16, color="black")
    plt.yticks(fontsize=16, color="black")
    user_ids = sorted(
        {entry["user_id"] for entry in all_entries if entry.get("user_id") is not None}
    )
    envs = sorted(
        {
            item.get("environment")
            for item in series
            if item.get("entries") and item.get("environment")
        }
    )
    title = _format_title(title_base, user_ids)
    if len(envs) == 1:
        title = f"{title} (env {envs[0]})"
    plt.title(title, fontsize=22)
    plt.grid(True, ls="--")
    plt.legend(fontsize=16, loc="lower left", facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()

    repo_root = REPO_ROOT
    user_id = args.user_id or 1
    log_dir = args.log_dir or (
        repo_root / "logs" / "retention_sweep" / f"user_{user_id}"
    )
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    envs = _parse_csv(args.env)

    plot_dir = args.plot_dir or (
        repo_root / "experiments" / "retention_sweep" / "plots" / f"user_{user_id}"
    )

    default_results = (
        "simulation_results_retention_sweep_compare.json"
        if len(envs) > 1
        else "simulation_results_retention_sweep.json"
    )
    results_path = args.results_path or (log_dir / default_results)

    base_dirs = [repo_root, log_dir]
    combined_results: List[Dict[str, Any]] = []
    series: List[Dict[str, Any]] = []
    schedulers = _parse_csv(args.sched) or ["fsrs6"]
    try:
        scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dr_schedulers: List[str] = []
    fixed_intervals: List[float] = []
    include_all_fixed = False
    has_sspmmc = False
    for name, interval, raw in scheduler_specs:
        if name == "sspmmc":
            has_sspmmc = True
            continue
        if name == "fixed":
            if interval is None or raw == "fixed":
                include_all_fixed = True
            else:
                fixed_intervals.append(interval)
            continue
        if name not in dr_schedulers:
            dr_schedulers.append(name)
    if include_all_fixed:
        fixed_intervals = []
    run_dr = bool(dr_schedulers)
    run_sspmmc = has_sspmmc
    run_fixed = include_all_fixed or bool(fixed_intervals)
    if not run_dr and not run_sspmmc and not run_fixed:
        raise SystemExit("No schedulers specified. Use --sched to select plots.")
    fuzz_filter = None
    if not args.compare_fuzz and args.fuzz != "any":
        fuzz_filter = args.fuzz == "on"
    short_term_filter = None
    if args.short_term != "any":
        short_term_filter = args.short_term == "on"
    short_term_source_filter = None
    if args.short_term_source != "any":
        short_term_source_filter = args.short_term_source
        short_term_filter = True
    if args.compare_short_term and short_term_source_filter is not None:
        raise SystemExit(
            "--compare-short-term cannot be combined with --short-term-source."
        )
    engine_filter = None
    if args.engine != "any":
        engine_filter = args.engine
    if args.compare_engine and engine_filter is not None:
        raise SystemExit("--compare-engine cannot be combined with --engine.")
    fuzz_series = [None]
    if args.compare_fuzz:
        fuzz_series = [False, True]
    short_term_series = [short_term_filter]
    if args.compare_short_term:
        short_term_series = [False, True]
    engine_series = [engine_filter]
    if args.compare_engine:
        engine_series = ["event", "vectorized"]
    for env in envs:
        if run_dr:
            for scheduler in dr_schedulers:
                for fuzz_value in fuzz_series:
                    for short_term_value in short_term_series:
                        for engine_value in engine_series:
                            results = _build_results(
                                log_dir,
                                env,
                                {scheduler},
                                args.start_retention,
                                args.end_retention,
                                base_dirs,
                                fuzz_value if args.compare_fuzz else fuzz_filter,
                                short_term_value,
                                short_term_source_filter,
                                engine_value,
                                title_prefix=None,
                                dedupe_fuzz=args.compare_fuzz,
                                dedupe_short_term=args.compare_short_term,
                                dedupe_engine=args.compare_engine,
                            )
                            if not results:
                                continue
                            label_parts = []
                            if len(envs) > 1:
                                label_parts.append(f"env={env}")
                            if len(dr_schedulers) > 1:
                                label_parts.append(f"sched={scheduler}")
                            if args.compare_fuzz:
                                label_parts.append(
                                    "fuzz=on" if fuzz_value else "fuzz=off"
                                )
                            if args.compare_short_term:
                                label_parts.append(
                                    "short-term=on"
                                    if short_term_value
                                    else "short-term=off"
                                )
                            if args.compare_engine:
                                label_parts.append(f"engine={engine_value}")
                            label = " ".join(label_parts) or scheduler
                            series.append(
                                {
                                    "label": label,
                                    "entries": results,
                                    "style": "dr",
                                    "scheduler": scheduler,
                                    "environment": env,
                                    "fuzz": fuzz_value if args.compare_fuzz else None,
                                    "short_term": short_term_value
                                    if args.compare_short_term
                                    else None,
                                    "engine": engine_value
                                    if args.compare_engine
                                    else None,
                                }
                            )
                            combined_results.extend(results)
        if run_fixed:
            interval_filter = None if include_all_fixed else fixed_intervals
            for fuzz_value in fuzz_series:
                for short_term_value in short_term_series:
                    for engine_value in engine_series:
                        fixed_results = _build_results(
                            log_dir,
                            env,
                            {"fixed"},
                            args.start_retention,
                            args.end_retention,
                            base_dirs,
                            fuzz_value if args.compare_fuzz else fuzz_filter,
                            short_term_value,
                            short_term_source_filter,
                            engine_value,
                            fixed_interval_filter=interval_filter,
                            title_prefix=None,
                            dedupe_fuzz=args.compare_fuzz,
                            dedupe_short_term=args.compare_short_term,
                            dedupe_engine=args.compare_engine,
                        )
                        if not fixed_results:
                            continue
                        fixed_results.sort(
                            key=lambda entry: entry.get("fixed_interval")
                            if entry.get("fixed_interval") is not None
                            else 0.0
                        )
                        fixed_label_parts = []
                        if len(envs) > 1:
                            fixed_label_parts.append(f"env={env}")
                        if run_dr or run_sspmmc or len(envs) > 1:
                            fixed_label_parts.append("sched=fixed")
                        if args.compare_fuzz:
                            fixed_label_parts.append(
                                "fuzz=on" if fuzz_value else "fuzz=off"
                            )
                        if args.compare_short_term:
                            fixed_label_parts.append(
                                "short-term=on"
                                if short_term_value
                                else "short-term=off"
                            )
                        if args.compare_engine:
                            fixed_label_parts.append(f"engine={engine_value}")
                        fixed_label = " ".join(fixed_label_parts) or "fixed"
                        series.append(
                            {
                                "label": fixed_label,
                                "entries": fixed_results,
                                "style": "dr",
                                "scheduler": "fixed",
                                "environment": env,
                                "fuzz": fuzz_value if args.compare_fuzz else None,
                                "short_term": short_term_value
                                if args.compare_short_term
                                else None,
                                "engine": engine_value if args.compare_engine else None,
                            }
                        )
                        combined_results.extend(fixed_results)
        if run_sspmmc:
            for fuzz_value in fuzz_series:
                for short_term_value in short_term_series:
                    for engine_value in engine_series:
                        sspmmc_results = _build_results(
                            log_dir,
                            env,
                            {"sspmmc"},
                            args.start_retention,
                            args.end_retention,
                            base_dirs,
                            fuzz_value if args.compare_fuzz else fuzz_filter,
                            short_term_value,
                            short_term_source_filter,
                            engine_value,
                            title_prefix=None,
                            dedupe=False,
                        )
                        if not sspmmc_results:
                            continue
                        sspmmc_results.sort(
                            key=lambda entry: entry["memorized_average"]
                        )
                        ssp_label_parts = ["sched=sspmmc"]
                        if len(envs) > 1:
                            ssp_label_parts.insert(0, f"env={env}")
                        if args.compare_fuzz:
                            ssp_label_parts.append(
                                "fuzz=on" if fuzz_value else "fuzz=off"
                            )
                        if args.compare_short_term:
                            ssp_label_parts.append(
                                "short-term=on"
                                if short_term_value
                                else "short-term=off"
                            )
                        if args.compare_engine:
                            ssp_label_parts.append(f"engine={engine_value}")
                        ssp_label = " ".join(ssp_label_parts)
                        series.append(
                            {
                                "label": ssp_label,
                                "entries": sspmmc_results,
                                "style": "sspmmc",
                                "scheduler": "sspmmc",
                                "environment": env,
                                "fuzz": fuzz_value if args.compare_fuzz else None,
                                "short_term": short_term_value
                                if args.compare_short_term
                                else None,
                                "engine": engine_value if args.compare_engine else None,
                            }
                        )
                        combined_results.extend(sspmmc_results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(combined_results, fh, indent=2, sort_keys=True)

    if args.no_plot:
        print(f"Wrote {len(combined_results)} entries to {results_path}")
        return

    os.environ.setdefault("MPLBACKEND", "Agg")

    plot_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()
    _plot_compare_frontier(
        series,
        plot_dir / "Pareto frontier.png",
        title_base="Pareto frontier",
        show_labels=not args.hide_labels,
    )
    print(f"Wrote {len(combined_results)} entries to {results_path}")


if __name__ == "__main__":
    main()
