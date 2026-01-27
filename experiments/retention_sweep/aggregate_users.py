from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        description=(
            "Aggregate retention_sweep logs across users and plot a Pareto-style "
            "frontier (mean across users)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Root directory containing retention_sweep logs (default logs/retention_sweep).",
    )
    parser.add_argument(
        "--environments",
        default="lstm",
        help="Comma-separated environments to include.",
    )
    parser.add_argument(
        "--schedulers",
        default="fsrs6,anki_sm2",
        help="Comma-separated schedulers to include.",
    )
    parser.add_argument(
        "--min-retention",
        type=float,
        default=0.70,
        help="Minimum desired retention to include.",
    )
    parser.add_argument(
        "--max-retention",
        type=float,
        default=0.99,
        help="Maximum desired retention to include.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to write aggregate results JSON.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Annotate points with DR labels (requires adjustText).",
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _iter_log_paths(log_root: Path) -> Iterable[Path]:
    if not log_root.exists():
        return []
    user_dirs = sorted(path for path in log_root.iterdir() if path.is_dir())
    if user_dirs:
        for user_dir in user_dirs:
            for path in sorted(user_dir.glob("*.jsonl")):
                yield path
    else:
        for path in sorted(log_root.glob("*.jsonl")):
            yield path


def _format_scheduler_title(scheduler: str) -> str:
    labels = {
        "anki_sm2": "Anki-SM-2",
        "memrise": "Memrise",
    }
    return labels.get(scheduler, scheduler)


def _normalize_user_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_user_id_from_path(path: Path) -> Optional[int]:
    parent = path.parent.name
    if parent.startswith("user_"):
        try:
            return int(parent.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    match = re.search(r"_user=(\d+)_", path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _format_title(base: str, user_ids: List[int]) -> str:
    if not user_ids:
        return base
    if len(user_ids) == 1:
        return f"{base} (user {user_ids[0]})"
    user_ids_sorted = sorted(user_ids)
    if user_ids_sorted[-1] - user_ids_sorted[0] + 1 == len(user_ids_sorted):
        return f"{base} (users {user_ids_sorted[0]}-{user_ids_sorted[-1]}, n={len(user_ids_sorted)})"
    return f"{base} (n={len(user_ids_sorted)})"


def main() -> None:
    args = parse_args()

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    envs = _parse_csv(args.environments)
    schedulers = [item for item in _parse_csv(args.schedulers) if item != "sspmmc"]
    if not schedulers:
        schedulers = ["fsrs6"]
    try:
        scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dr_schedulers: List[str] = []
    fixed_intervals: List[float] = []
    include_all_fixed = False
    for name, interval, raw in scheduler_specs:
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
    run_fixed = include_all_fixed or bool(fixed_intervals)
    if not run_dr and not run_fixed:
        raise SystemExit("No schedulers specified. Use --schedulers to select plots.")

    results_path = args.results_path or (
        log_root / "simulation_results_retention_sweep_user_averages.json"
    )
    plot_dir = args.plot_dir or (
        REPO_ROOT / "experiments" / "retention_sweep" / "plots" / "user_averages"
    )

    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]] = {}
    duplicate_count = 0
    all_user_ids: set[int] = set()

    for path in _iter_log_paths(log_root):
        try:
            meta, totals = _load_meta_totals(path)
        except ValueError:
            continue

        environment = meta.get("environment")
        scheduler = meta.get("scheduler")
        if envs and environment not in envs:
            continue
        if scheduler is None:
            continue
        if scheduler == "sspmmc":
            continue
        if scheduler not in {name for name, _, _ in scheduler_specs}:
            continue

        desired = meta.get("desired_retention")
        fixed_interval = None
        if scheduler == "fixed":
            raw_interval = meta.get("fixed_interval")
            fixed_interval = normalize_fixed_interval(
                float(raw_interval) if raw_interval is not None else None
            )
            if run_fixed and fixed_intervals:
                if not any(
                    math.isclose(fixed_interval, value, rel_tol=0.0, abs_tol=1e-6)
                    for value in fixed_intervals
                ):
                    continue
        if scheduler_uses_desired_retention(scheduler):
            if desired is None:
                continue
            try:
                desired = float(desired)
            except (TypeError, ValueError):
                continue
            if desired < args.min_retention or desired > args.max_retention:
                continue
        else:
            desired = None

        time_average = totals.get("time_average")
        if time_average is None:
            continue
        time_average = float(time_average)
        if time_average <= 0:
            continue

        memorized_average = float(totals.get("memorized_average", 0.0))
        memorized_per_minute = memorized_average / time_average
        user_id = _normalize_user_id(meta.get("user_id"))
        if user_id is None:
            user_id = _infer_user_id_from_path(path)
        user_key = user_id if user_id is not None else f"unknown::{path}"
        mtime = path.stat().st_mtime
        if user_id is not None:
            all_user_ids.add(user_id)

        key = (environment, scheduler, desired, fixed_interval)
        group = groups.setdefault(
            key,
            {
                "environment": environment,
                "scheduler": scheduler,
                "desired_retention": desired,
                "fixed_interval": fixed_interval,
                "users": {},
            },
        )
        existing = group["users"].get(user_key)
        if existing is not None:
            duplicate_count += 1
            if mtime <= existing["mtime"]:
                continue
        group["users"][user_key] = {
            "metrics": {
                "memorized_average": memorized_average,
                "memorized_per_minute": memorized_per_minute,
            },
            "mtime": mtime,
        }

    results: List[Dict[str, Any]] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or -1,
            item[0][3] or -1,
        ),
    ):
        users = [item["metrics"] for item in group["users"].values()]
        if not users:
            continue
        scheduler = group["scheduler"]
        desired = group["desired_retention"]
        fixed_interval = group["fixed_interval"]
        if scheduler == "fixed":
            title = f"Ivl={format_float(fixed_interval)}"
        elif scheduler_uses_desired_retention(scheduler):
            title = f"DR={format_float(float(desired) * 100)}%"
        else:
            title = _format_scheduler_title(scheduler)
        results.append(
            {
                "environment": group["environment"],
                "scheduler": group["scheduler"],
                "desired_retention": group["desired_retention"],
                "fixed_interval": group["fixed_interval"],
                "user_count": len(users),
                "memorized_average": mean(item["memorized_average"] for item in users),
                "memorized_per_minute": mean(
                    item["memorized_per_minute"] for item in users
                ),
                "title": title,
            }
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if duplicate_count:
        print(f"Note: skipped {duplicate_count} duplicate logs (kept newest per user).")

    if args.no_plot:
        print(f"Wrote {results_path}")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    series: List[Dict[str, Any]] = []
    for env in envs:
        if run_dr:
            for scheduler in dr_schedulers:
                entries = [
                    entry
                    for entry in results
                    if entry["environment"] == env and entry["scheduler"] == scheduler
                ]
                if scheduler_uses_desired_retention(scheduler):
                    entries.sort(
                        key=lambda entry: entry["desired_retention"]
                        if entry["desired_retention"] is not None
                        else -1
                    )
                label_parts = []
                if len(envs) > 1:
                    label_parts.append(f"env={env}")
                if len(dr_schedulers) > 1:
                    label_parts.append(f"sched={scheduler}")
                label = " ".join(label_parts) or scheduler
                series.append(
                    {
                        "label": label,
                        "entries": entries,
                        "scheduler": scheduler,
                        "environment": env,
                    }
                )
        if run_fixed:
            fixed_entries = [
                entry
                for entry in results
                if entry["environment"] == env and entry["scheduler"] == "fixed"
            ]
            fixed_entries.sort(
                key=lambda entry: entry["fixed_interval"]
                if entry["fixed_interval"] is not None
                else 0.0
            )
            fixed_label_parts = []
            if len(envs) > 1:
                fixed_label_parts.append(f"env={env}")
            if run_dr or len(envs) > 1:
                fixed_label_parts.append("sched=fixed")
            fixed_label = " ".join(fixed_label_parts) or "fixed"
            series.append(
                {
                    "label": fixed_label,
                    "entries": fixed_entries,
                    "scheduler": "fixed",
                    "environment": env,
                }
            )

    _setup_plot_style()
    output_path = plot_dir / "retention_sweep_user_averages.png"
    user_ids = sorted(all_user_ids)
    title = _format_title("Pareto frontier (user averages)", user_ids)
    _plot_compare_frontier(
        series,
        output_path,
        title_base=title,
        user_count=len(user_ids),
        show_labels=args.show_labels,
    )
    print(f"Wrote {results_path}")
    print(f"Saved plot to {output_path}")


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _plot_compare_frontier(
    series: List[Dict[str, Any]],
    output_path: Path,
    title_base: str,
    user_count: int | None = None,
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
    for item in series:
        scheduler = item.get("scheduler")
        environment = item.get("environment")
        if scheduler and scheduler not in scheduler_order:
            scheduler_order.append(scheduler)
        if environment and environment not in environment_order:
            environment_order.append(environment)
    scheduler_colors = {
        scheduler: colors[idx % len(colors)]
        for idx, scheduler in enumerate(scheduler_order)
    }
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
        is_single_point = scheduler in single_point_markers and len(entries) == 1
        if is_single_point:
            color = "red"
            linestyle = ""
            marker = single_point_markers.get(scheduler, "^")
            linewidth = 2
            markersize = 7.5
        else:
            color = scheduler_colors.get(scheduler, colors[0])
            linestyle = environment_linestyles.get(environment, linestyles[0])
            marker = "o"
            linewidth = 2
            markersize = 6
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
    plt.ylabel("Memorized cards (average)/minutes per day", fontsize=18, color="black")
    plt.xticks(fontsize=16, color="black")
    plt.yticks(fontsize=16, color="black")
    plt.title(title_base, fontsize=22)
    if user_count is not None:
        ax = plt.gca()
        ax.text(
            0.98,
            0.98,
            f"Users: {user_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    plt.grid(True, ls="--")
    plt.legend(fontsize=16, loc="lower left", facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
