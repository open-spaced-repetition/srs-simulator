from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from statistics import mean, median
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
        description="Aggregate retention_sweep logs across users and write summary JSON.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Root directory containing retention_sweep logs (default logs/retention_sweep).",
    )
    parser.add_argument(
        "--env",
        "--environments",
        dest="environments",
        default="lstm",
        help="Comma-separated environments to include.",
    )
    parser.add_argument(
        "--sched",
        "--schedulers",
        dest="schedulers",
        default="fsrs6,anki_sm2,memrise",
        help="Comma-separated schedulers to include.",
    )
    parser.add_argument(
        "--start-retention",
        "--min-retention",
        dest="start_retention",
        type=float,
        default=0.50,
        help="Minimum desired retention to include.",
    )
    parser.add_argument(
        "--end-retention",
        "--max-retention",
        dest="end_retention",
        type=float,
        default=0.98,
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
        "--equiv-report",
        action="store_true",
        help="Print summary stats for FSRS-6 equivalence comparisons.",
    )
    parser.add_argument(
        "--equiv-report-path",
        type=Path,
        default=None,
        help="Optional path to write FSRS-6 equivalence summary JSON.",
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
        raise SystemExit(
            "No schedulers specified. Use --sched/--schedulers to select plots."
        )

    results_path = args.results_path or (
        log_root / "simulation_results_retention_sweep_user_averages.json"
    )
    plot_dir = args.plot_dir or (
        REPO_ROOT / "experiments" / "retention_sweep" / "plots" / "user_averages"
    )

    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]] = {}
    duplicate_count = 0

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
                desired = round(float(desired), 2)
            except (TypeError, ValueError):
                continue
            if desired < args.start_retention or desired > args.end_retention:
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

    common_user_ids: Optional[set[int]] = None
    groups_with_users = 0
    for group in groups.values():
        group_user_ids = {
            user_key for user_key in group["users"].keys() if isinstance(user_key, int)
        }
        if not group_user_ids:
            continue
        if common_user_ids is None:
            common_user_ids = set(group_user_ids)
        else:
            common_user_ids &= group_user_ids
        groups_with_users += 1

    if common_user_ids is None:
        raise SystemExit("No user ids found across logs; cannot compute intersections.")
    if not common_user_ids:
        raise SystemExit(
            "No overlapping user ids across configs; intersection is empty."
        )

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
        users = [
            payload["metrics"]
            for user_key, payload in group["users"].items()
            if isinstance(user_key, int) and user_key in common_user_ids
        ]
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
        memorized_average_values = [item["memorized_average"] for item in users]
        memorized_per_minute_values = [item["memorized_per_minute"] for item in users]
        memorized_average_mean = mean(memorized_average_values)
        memorized_per_minute_mean = mean(memorized_per_minute_values)
        memorized_average_median = median(memorized_average_values)
        memorized_per_minute_median = median(memorized_per_minute_values)
        results.append(
            {
                "environment": group["environment"],
                "scheduler": group["scheduler"],
                "desired_retention": group["desired_retention"],
                "fixed_interval": group["fixed_interval"],
                "user_count": len(common_user_ids),
                "memorized_average": memorized_average_mean,
                "memorized_per_minute": memorized_per_minute_mean,
                "memorized_average_mean": memorized_average_mean,
                "memorized_per_minute_mean": memorized_per_minute_mean,
                "memorized_average_median": memorized_average_median,
                "memorized_per_minute_median": memorized_per_minute_median,
                "title": title,
            }
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if duplicate_count:
        print(f"Note: skipped {duplicate_count} duplicate logs (kept newest per user).")
    if groups_with_users:
        print(
            "Note: using intersection of users across configs: "
            f"{len(common_user_ids)} users."
        )

    wants_equiv = args.equiv_report or args.equiv_report_path is not None
    needs_equiv = wants_equiv or not args.no_plot
    equivalent_distributions: List[Dict[str, Any]] = []
    if needs_equiv:
        equivalent_distributions = _compute_equivalent_fsrs6_distributions(
            groups,
            envs,
            common_user_ids,
            baselines=["anki_sm2", "memrise"],
        )

    if not args.no_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        _setup_plot_style()
        for entry in equivalent_distributions:
            env_label = entry["environment"]
            distribution_title = _format_title(
                f"{_format_scheduler_title(entry['baseline'])} vs FSRS-6 equiv distributions",
                sorted(entry["user_ids"]),
            )
            if len(envs) > 1:
                distribution_title = f"{env_label} {distribution_title}"
            suffix = f"_{env_label}" if len(envs) > 1 else ""
            baseline_suffix = entry["baseline"]
            distribution_path = (
                plot_dir
                / f"retention_sweep_equivalent_fsrs6_distributions_{baseline_suffix}{suffix}.png"
            )
            _plot_equivalent_distributions(entry, distribution_path, distribution_title)
            print(f"Saved plot to {distribution_path}")

    if wants_equiv:
        summaries = _summarize_equivalent_distributions(equivalent_distributions)
        _print_equiv_report(summaries)
        if args.equiv_report_path is not None:
            args.equiv_report_path.parent.mkdir(parents=True, exist_ok=True)
            with args.equiv_report_path.open("w", encoding="utf-8") as fh:
                json.dump(summaries, fh, indent=2)
            print(f"Wrote {args.equiv_report_path}")

    if args.no_plot:
        print(f"Wrote {results_path}")
        return
    print(f"Wrote {results_path}")


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _compute_equivalent_fsrs6_distributions(
    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]],
    envs: List[str],
    common_user_ids: set[int],
    baselines: List[str],
) -> List[Dict[str, Any]]:
    distributions: List[Dict[str, Any]] = []
    for env in envs:
        fsrs_users: Dict[int, List[Tuple[float, Dict[str, float]]]] = {}

        for (group_env, scheduler, desired, _), group in groups.items():
            if group_env != env:
                continue
            if scheduler == "fsrs6" and desired is not None:
                for user_key, payload in group["users"].items():
                    if not isinstance(user_key, int):
                        continue
                    fsrs_users.setdefault(user_key, []).append(
                        (float(desired), payload["metrics"])
                    )

        for baseline in baselines:
            baseline_users: Dict[int, Dict[str, float]] = {}
            for (group_env, scheduler, _, _), group in groups.items():
                if group_env != env or scheduler != baseline:
                    continue
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        baseline_users[user_key] = payload["metrics"]

            eligible_users = set(baseline_users) & set(fsrs_users) & common_user_ids
            if not eligible_users:
                continue

            baseline_per_minute: List[float] = []
            fsrs_per_minute: List[float] = []
            fsrs_dr_equiv: List[float] = []
            used_users: set[int] = set()
            for user_id in sorted(eligible_users):
                baseline_metrics = baseline_users[user_id]
                candidates = fsrs_users[user_id]
                if len(candidates) < 2:
                    continue
                target_value = baseline_metrics["memorized_average"]
                candidate_points = [
                    {
                        "dr": dr,
                        "x": payload["memorized_average"],
                        "y": payload["memorized_per_minute"],
                    }
                    for dr, payload in candidates
                ]
                candidate_points.sort(key=lambda item: item["x"])
                lower = None
                upper = None
                for point in candidate_points:
                    if point["x"] <= target_value:
                        lower = point
                    if point["x"] >= target_value and upper is None:
                        upper = point
                if lower is None or upper is None:
                    candidate_points.sort(
                        key=lambda item: abs(item["x"] - target_value)
                    )
                    lower = candidate_points[0]
                    upper = candidate_points[1]
                x1 = lower["x"]
                x2 = upper["x"]
                if math.isclose(x1, x2):
                    dr_equiv = (lower["dr"] + upper["dr"]) / 2.0
                    y_equiv = (lower["y"] + upper["y"]) / 2.0
                else:
                    t = (target_value - x1) / (x2 - x1)
                    t = max(0.0, min(1.0, t))
                    dr_equiv = lower["dr"] + t * (upper["dr"] - lower["dr"])
                    y_equiv = lower["y"] + t * (upper["y"] - lower["y"])

                baseline_per_minute.append(baseline_metrics["memorized_per_minute"])
                fsrs_per_minute.append(y_equiv)
                fsrs_dr_equiv.append(dr_equiv)
                used_users.add(user_id)

            if not used_users:
                continue

            distributions.append(
                {
                    "environment": env,
                    "baseline": baseline,
                    "baseline_per_minute": baseline_per_minute,
                    "fsrs_per_minute": fsrs_per_minute,
                    "fsrs_dr_equiv": fsrs_dr_equiv,
                    "user_ids": used_users,
                }
            )

    return distributions


def _summarize_equivalent_distributions(
    distributions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for entry in distributions:
        baseline_per_minute = entry.get("baseline_per_minute", [])
        fsrs_per_minute = entry.get("fsrs_per_minute", [])
        dr_equiv = entry.get("fsrs_dr_equiv", [])
        if not baseline_per_minute or not fsrs_per_minute:
            continue
        diffs = [
            fsrs_value - baseline_value
            for baseline_value, fsrs_value in zip(baseline_per_minute, fsrs_per_minute)
        ]
        ratios = [
            fsrs_value / baseline_value
            for baseline_value, fsrs_value in zip(baseline_per_minute, fsrs_per_minute)
            if baseline_value > 0
        ]
        if not ratios:
            continue
        pos = sum(1 for value in diffs if value > 0)
        neg = sum(1 for value in diffs if value < 0)
        zero = len(diffs) - pos - neg
        try:
            q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
        except Exception:
            ratios_sorted = sorted(ratios)
            q1 = ratios_sorted[0]
            q3 = ratios_sorted[-1]

        summaries.append(
            {
                "environment": entry.get("environment"),
                "baseline": entry.get("baseline"),
                "user_count": len(entry.get("user_ids", [])),
                "diff_mean": mean(diffs),
                "diff_median": median(diffs),
                "ratio_mean": mean(ratios),
                "ratio_median": median(ratios),
                "ratio_q25": q1,
                "ratio_q75": q3,
                "pos_pct": pos / len(diffs),
                "neg_pct": neg / len(diffs),
                "zero_pct": zero / len(diffs),
                "dr_equiv_mean": mean(dr_equiv) if dr_equiv else None,
                "dr_equiv_median": median(dr_equiv) if dr_equiv else None,
            }
        )
    return summaries


def _print_equiv_report(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        print("No FSRS-6 equivalence summaries available.")
        return
    print("FSRS-6 equivalence summary (matched memorized average)")
    for summary in summaries:
        env = summary.get("environment")
        baseline = _format_scheduler_title(str(summary.get("baseline")))
        user_count = summary.get("user_count")
        pos_pct = summary.get("pos_pct")
        neg_pct = summary.get("neg_pct")
        ratio_mean = summary.get("ratio_mean")
        ratio_median = summary.get("ratio_median")
        ratio_q25 = summary.get("ratio_q25")
        ratio_q75 = summary.get("ratio_q75")
        dr_mean = summary.get("dr_equiv_mean")
        dr_median = summary.get("dr_equiv_median")
        print(
            f"- FSRS-6 vs {baseline} (env={env}): n={user_count}, "
            f"superiority={pos_pct:.1%}, "
            f"mean ratio={ratio_mean:.3f}, "
            f"median ratio={ratio_median:.3f} "
            f"(IQR {ratio_q25:.3f}-{ratio_q75:.3f}), "
            f"mean DR={dr_mean:.3f}, "
            f"median DR={dr_median:.3f}"
        )


def _plot_equivalent_distributions(
    entry: Dict[str, Any],
    output_path: Path,
    title_base: str,
) -> None:
    import matplotlib.pyplot as plt

    baseline_per_minute = entry["baseline_per_minute"]
    fsrs_per_minute = entry["fsrs_per_minute"]
    fsrs_dr_equiv = entry["fsrs_dr_equiv"]
    baseline_label = _format_scheduler_title(entry["baseline"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax_middle = axes[0, 0]
    ax_left = axes[0, 1]
    ax_right = axes[1, 0]
    ax_box = axes[1, 1]
    ax_ratio_hist = axes[2, 0]
    ax_ratio_box = axes[2, 1]

    boxplot_left = ax_left.boxplot(
        [baseline_per_minute, fsrs_per_minute],
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#d9d9d9", "edgecolor": "black"},
        medianprops={"color": "black"},
    )
    for patch, color in zip(boxplot_left["boxes"], ["#1f77b4", "#2ca02c"]):
        patch.set_facecolor(color)
    ax_left.set_xticks([1, 2])
    ax_left.set_xticklabels([baseline_label, "FSRS-6 equiv"])
    ax_left.set_ylabel("Memorized cards/min (average)")
    ax_left.grid(True, axis="y", ls="--", alpha=0.6)

    def _annotate_box_stats(
        ax: Any,
        values_list: List[List[float]],
        x_positions: List[float],
    ) -> Tuple[List[str], List[float], List[float]]:
        labels: List[str] = []
        means: List[float] = []
        medians: List[float] = []
        for x_pos, values in zip(x_positions, values_list):
            if not values:
                continue
            mean_value = mean(values)
            median_value = median(values)
            means.append(mean_value)
            medians.append(median_value)
            labels.append(f"μ={mean_value:.2f}  med={median_value:.2f}")
            ax.scatter(
                [x_pos],
                [mean_value],
                color="black",
                marker="D",
                s=28,
                zorder=4,
            )
            ax.scatter(
                [x_pos],
                [median_value],
                color="white",
                edgecolor="black",
                marker="o",
                s=26,
                zorder=4,
            )
        return labels, means, medians

    left_labels, _, _ = _annotate_box_stats(
        ax_left,
        [baseline_per_minute, fsrs_per_minute],
        [1, 2],
    )
    if left_labels:
        ax_left.legend(
            left_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )

    dr_percent = [value * 100 for value in fsrs_dr_equiv]
    dr_bins = min(15, max(5, len(dr_percent) // 3))
    ax_middle.hist(dr_percent, bins=dr_bins, color="#1f77b4", alpha=0.8)
    ax_middle.set_xlabel("Equivalent FSRS-6 DR (%)")
    ax_middle.set_ylabel("User count")
    ax_middle.grid(True, axis="y", ls="--", alpha=0.6)

    diff_values = [
        fsrs_value - baseline_value
        for baseline_value, fsrs_value in zip(baseline_per_minute, fsrs_per_minute)
    ]
    if diff_values:
        diff_bins = min(15, max(5, len(diff_values) // 3))
        ax_right.hist(diff_values, bins=diff_bins, color="#ff7f0e", alpha=0.8)
    ax_right.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_right.set_xlabel(f"FSRS-6 equiv - {baseline_label} (cards/min)")
    ax_right.set_ylabel("User count")
    ax_right.grid(True, axis="y", ls="--", alpha=0.6)

    if diff_values:
        ax_box.boxplot(
            diff_values,
            vert=True,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": "#ff7f0e", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        ax_box.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_box.set_xticks([1])
        ax_box.set_xticklabels(["Diff"])
        ax_box.set_ylabel(f"FSRS-6 equiv - {baseline_label} (cards/min)")
        ax_box.grid(True, axis="y", ls="--", alpha=0.6)
        diff_positive = sum(value > 0 for value in diff_values)
        diff_ratio = diff_positive / len(diff_values)
        ax_box.text(
            0.98,
            0.98,
            f"diff>0: {diff_ratio:.1%}",
            transform=ax_box.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        diff_labels, _, _ = _annotate_box_stats(ax_box, [diff_values], [1])
        if diff_labels:
            ax_box.legend(
                diff_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
    else:
        ax_box.axis("off")

    ratio_values = [
        fsrs_value / baseline_value
        for baseline_value, fsrs_value in zip(baseline_per_minute, fsrs_per_minute)
        if baseline_value > 0
    ]
    if ratio_values:
        ratio_bins = min(15, max(5, len(ratio_values) // 3))
        ax_ratio_hist.hist(ratio_values, bins=ratio_bins, color="#2ca02c", alpha=0.8)
    ax_ratio_hist.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_ratio_hist.set_xlabel(f"FSRS-6 equiv / {baseline_label} (cards/min)")
    ax_ratio_hist.set_ylabel("User count")
    ax_ratio_hist.grid(True, axis="y", ls="--", alpha=0.6)

    if ratio_values:
        ax_ratio_box.boxplot(
            ratio_values,
            vert=True,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": "#2ca02c", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        ax_ratio_box.axhline(
            1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7
        )
        ax_ratio_box.set_xticks([1])
        ax_ratio_box.set_xticklabels(["Ratio"])
        ax_ratio_box.set_ylabel(f"FSRS-6 equiv / {baseline_label} (cards/min)")
        ax_ratio_box.grid(True, axis="y", ls="--", alpha=0.6)
        ratio_above = sum(value > 1 for value in ratio_values)
        ratio_ratio = ratio_above / len(ratio_values)
        ax_ratio_box.text(
            0.98,
            0.98,
            f"ratio>1: {ratio_ratio:.1%}",
            transform=ax_ratio_box.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ratio_labels, _, _ = _annotate_box_stats(ax_ratio_box, [ratio_values], [1])
        if ratio_labels:
            ax_ratio_box.legend(
                ratio_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
    else:
        ax_ratio_box.axis("off")

    fig.suptitle(title_base, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
