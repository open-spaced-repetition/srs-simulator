from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert simulate.py logs into a Pareto frontier plot.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing simulate.py JSONL logs.",
    )
    parser.add_argument(
        "--environments",
        default="lstm",
        help="Comma-separated list of environments to compare.",
    )
    parser.add_argument(
        "--schedulers",
        default="fsrs6",
        help="Comma-separated list of schedulers to plot (include sspmmc for policies).",
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
        "--sspmmc-root",
        type=Path,
        default=None,
        help="Path to the SSP-MMC-FSRS repo root.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to write simulation_results.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the plot styling setup.",
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
                return title.strip()
        except (OSError, json.JSONDecodeError):
            return path.stem
    return path.stem


def _iter_log_entries(
    log_dir: Path,
    environment: str,
    scheduler_filter: Optional[set[str]],
    min_retention: float,
    max_retention: float,
    base_dirs: Sequence[Path],
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

        desired = float(meta.get("desired_retention", 0.0))
        if scheduler != "sspmmc":
            if desired < min_retention or desired > max_retention:
                continue

        avg_per_hour = totals.get("avg_accum_memorized_per_hour")
        if avg_per_hour is None:
            continue

        if scheduler == "sspmmc":
            title = _resolve_policy_title(meta, base_dirs) or "SSP-MMC"
        else:
            title = f"DR={desired:.2f}"

        entry = {
            "title": title,
            "reviews_average": float(totals.get("reviews_average", 0.0)),
            "time_average": float(totals.get("time_average", 0.0)),
            "memorized_average": float(totals.get("memorized_average", 0.0)),
            "avg_accum_memorized_per_hour": float(avg_per_hour),
            "scheduler": scheduler,
        }
        yield desired if scheduler != "sspmmc" else None, entry


def _build_results(
    log_dir: Path,
    environment: str,
    scheduler_filter: Optional[set[str]],
    min_retention: float,
    max_retention: float,
    base_dirs: Sequence[Path],
    title_prefix: str | None = None,
    dedupe: bool = True,
) -> List[Dict[str, Any]]:
    by_retention: Dict[float, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for desired, entry in _iter_log_entries(
        log_dir, environment, scheduler_filter, min_retention, max_retention, base_dirs
    ):
        if title_prefix:
            entry["title"] = f"{title_prefix} {entry['title']}"
        entry["environment"] = environment
        if dedupe and desired is not None:
            by_retention[desired] = entry
        else:
            results.append(entry)

    if dedupe:
        return [by_retention[ret] for ret in sorted(by_retention)]
    return results


def _plot_compare_frontier(
    series: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    all_entries = [entry for item in series for entry in item["entries"]]
    if not all_entries:
        raise ValueError("No entries available to plot.")

    min_x = min(entry["memorized_average"] for entry in all_entries)
    max_x = max(entry["memorized_average"] for entry in all_entries)
    max_y = max(entry["avg_accum_memorized_per_hour"] for entry in all_entries)

    x_min = 200 * math.floor(min_x / 200) if min_x else 0
    x_max = 200 * math.ceil(max_x / 200) if max_x else 1
    y_min = 0
    y_max = max_y * 1.03 if max_y else 1

    colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#264478"]
    markers = ["o", "s", "D", "^"]

    plt.figure(figsize=(12, 9))
    for idx, item in enumerate(series):
        entries = item["entries"]
        if not entries:
            continue
        color = colors[idx % len(colors)]
        style = item["style"]
        marker = "X" if style == "sspmmc" else markers[idx % len(markers)]
        linestyle = "--" if style == "sspmmc" else "-"
        linewidth = 1.8 if style == "sspmmc" else 2
        markersize = 8 if style == "sspmmc" else 6
        x_vals = [entry["memorized_average"] for entry in entries]
        y_vals = [entry["avg_accum_memorized_per_hour"] for entry in entries]
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

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel("Memorized cards (average, all days)\n(higher=better)", fontsize=18)
    plt.ylabel(
        "Memorized/hours spent (average, all days)\n(higher=better)", fontsize=18
    )
    plt.xticks(fontsize=16, color="black")
    plt.yticks(fontsize=16, color="black")
    plt.title("Pareto frontier comparison", fontsize=22)
    plt.grid(True, ls="--")
    plt.legend(fontsize=16, loc="lower left", facecolor="white")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    log_dir = args.log_dir or (repo_root / "logs" / "retention_sweep")
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    envs = _parse_csv(args.environments)

    ssp_root = args.sspmmc_root or (repo_root.parent / "SSP-MMC-FSRS")
    plot_dir = args.plot_dir or (
        repo_root / "experiments" / "retention_sweep" / "plots"
    )

    default_results = (
        "simulation_results_retention_sweep_compare.json"
        if len(envs) > 1
        else "simulation_results_retention_sweep.json"
    )
    results_path = args.results_path or (
        ssp_root / "outputs" / "checkpoints" / default_results
    )

    base_dirs = [repo_root, ssp_root, log_dir]
    combined_results: List[Dict[str, Any]] = []
    series: List[Dict[str, Any]] = []
    schedulers = _parse_csv(args.schedulers) or ["fsrs6"]
    dr_schedulers = [scheduler for scheduler in schedulers if scheduler != "sspmmc"]
    has_sspmmc = "sspmmc" in schedulers
    run_dr = bool(dr_schedulers)
    run_sspmmc = has_sspmmc
    if not run_dr and not run_sspmmc:
        raise SystemExit("No schedulers specified. Use --schedulers to select plots.")
    for env in envs:
        if run_dr:
            for scheduler in dr_schedulers:
                results = _build_results(
                    log_dir,
                    env,
                    {scheduler},
                    args.min_retention,
                    args.max_retention,
                    base_dirs,
                    title_prefix=None,
                )
                label_parts = []
                if len(dr_schedulers) > 1:
                    label_parts.append(f"sched={scheduler}")
                if len(envs) > 1:
                    label_parts.append(f"env={env}")
                label = " ".join(label_parts) or scheduler
                series.append(
                    {
                        "label": label,
                        "entries": results,
                        "style": "dr",
                    }
                )
                combined_results.extend(results)
        if run_sspmmc:
            sspmmc_results = _build_results(
                log_dir,
                env,
                {"sspmmc"},
                args.min_retention,
                args.max_retention,
                base_dirs,
                title_prefix=None,
                dedupe=False,
            )
            sspmmc_results.sort(key=lambda entry: entry["memorized_average"])
            ssp_label_parts = ["sched=sspmmc"]
            if len(envs) > 1:
                ssp_label_parts.append(f"env={env}")
            ssp_label = " ".join(ssp_label_parts)
            series.append(
                {
                    "label": ssp_label,
                    "entries": sspmmc_results,
                    "style": "sspmmc",
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
    os.chdir(ssp_root)

    for path in (ssp_root, ssp_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from experiments import lib as ssp_lib  # noqa: E402

    plot_dir.mkdir(parents=True, exist_ok=True)
    ssp_lib.setup_environment(args.seed)
    use_compare = run_sspmmc or len(series) != 1
    if use_compare:
        output_path = plot_dir / "Pareto frontier env compare.png"
        _plot_compare_frontier(series, output_path)
    else:
        ssp_lib.PLOTS_DIR = plot_dir
        ssp_lib.plot_pareto_frontier(results_path, [])
    print(f"Wrote {len(combined_results)} entries to {results_path}")


if __name__ == "__main__":
    main()
