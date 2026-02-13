from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare scheduler dominance per user using retention_sweep logs."
        ),
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
        dest="env",
        default="lstm",
        help="Comma-separated environments to include.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to write dominance results JSON.",
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
        "--epsilon",
        type=float,
        default=0.0,
        help="Treat metric differences within this epsilon as ties.",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized", "batched", "any"],
        default="any",
        help="Filter logs by simulation engine.",
    )
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first", "any"],
        default="any",
        help="Filter logs by review priority.",
    )
    parser.add_argument(
        "--fsrs6-default-dr",
        type=float,
        default=0.9,
        help="Desired retention to select FSRS-6 default logs.",
    )
    parser.add_argument(
        "--fsrs3-default-dr",
        type=float,
        default=0.9,
        help="Desired retention to select FSRS-3 default logs.",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="any",
        help="Filter logs by short-term flag.",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched", "any"],
        default="any",
        help="Filter logs by short-term source (steps/sched).",
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _infer_engine(meta: Dict[str, Any], path: Path) -> Optional[str]:
    value = meta.get("engine")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "batch":
            lowered = "batched"
        if lowered in {"event", "vectorized", "batched"}:
            return lowered
    marker = "_engine="
    if marker in path.name:
        try:
            lowered = path.name.split(marker, 1)[1].split("_", 1)[0].strip().lower()
        except IndexError:
            return None
        if lowered == "batch":
            lowered = "batched"
        if lowered in {"event", "vectorized", "batched"}:
            return lowered
    return None


def _infer_priority(meta: Dict[str, Any], path: Path) -> Optional[str]:
    value = meta.get("priority")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"review-first", "new-first"}:
            return lowered
    marker = "_prio="
    if marker in path.name:
        try:
            lowered = path.name.split(marker, 1)[1].split("_", 1)[0].strip().lower()
        except IndexError:
            return None
        if lowered in {"review-first", "new-first"}:
            return lowered
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
    stem = path.name
    marker = "_user="
    if marker in stem:
        try:
            return int(stem.split(marker, 1)[1].split("_", 1)[0])
        except ValueError:
            return None
    return None


def _compare(a: float, b: float, epsilon: float) -> int:
    if abs(a - b) <= epsilon:
        return 0
    return 1 if a > b else -1


def _dominance_label(
    a_metrics: Dict[str, float],
    b_metrics: Dict[str, float],
    *,
    epsilon: float,
) -> str:
    mem_cmp = _compare(
        a_metrics["memorized_average"],
        b_metrics["memorized_average"],
        epsilon,
    )
    eff_cmp = _compare(
        a_metrics["memorized_per_minute"],
        b_metrics["memorized_per_minute"],
        epsilon,
    )
    if mem_cmp == 0 and eff_cmp == 0:
        return "neither"
    if mem_cmp == 0:
        return "a_dominates" if eff_cmp > 0 else "b_dominates"
    if eff_cmp == 0:
        return "a_dominates" if mem_cmp > 0 else "b_dominates"
    if mem_cmp > 0 and eff_cmp > 0:
        return "a_dominates"
    if mem_cmp < 0 and eff_cmp < 0:
        return "b_dominates"
    if mem_cmp > 0 and eff_cmp < 0:
        return "a_b_tradeoff"
    if mem_cmp < 0 and eff_cmp > 0:
        return "b_a_tradeoff"
    return "neither"


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _plot_dominance(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    *,
    a_label: str,
    b_label: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    envs = [item["environment"] for item in results]
    a_dom = [item["a_dominates_pct"] * 100 for item in results]
    b_dom = [item["b_dominates_pct"] * 100 for item in results]
    a_tradeoff = [item["a_b_tradeoff_pct"] * 100 for item in results]
    b_tradeoff = [item["b_a_tradeoff_pct"] * 100 for item in results]
    counts = [item["user_count"] for item in results]

    x = np.arange(len(envs))
    plt.figure(figsize=(10, 6))
    bars_a = plt.bar(x, a_dom, label=f"{a_label} dominates", color="#1f77b4")
    bars_b = plt.bar(
        x,
        b_dom,
        bottom=a_dom,
        label=f"{b_label} dominates",
        color="#2ca02c",
    )
    bars_a_tradeoff = plt.bar(
        x,
        a_tradeoff,
        bottom=[a + b for a, b in zip(a_dom, b_dom)],
        label=f"{a_label} higher memorized, lower efficiency",
        color="#ff7f0e",
    )
    bars_b_tradeoff = plt.bar(
        x,
        b_tradeoff,
        bottom=[a + b + c for a, b, c in zip(a_dom, b_dom, a_tradeoff)],
        label=f"{b_label} higher memorized, lower efficiency",
        color="#d62728",
    )

    def _label_segment(bars, values, bottoms, small_offsets):
        for idx, (bar, value, bottom) in enumerate(zip(bars, values, bottoms)):
            if value <= 0:
                continue
            y = bottom + value / 2
            text = f"{value:.1f}%"
            if value < 3.0:
                y = bottom + value + 1.2 + small_offsets[idx] * 1.4
                small_offsets[idx] += 1
                va = "bottom"
            else:
                va = "center"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va=va,
                fontsize=9,
                color="white" if value >= 5.0 else "black",
            )

    a_bottoms = [0.0] * len(x)
    b_bottoms = a_dom
    a_tradeoff_bottoms = [a + b for a, b in zip(a_dom, b_dom)]
    b_tradeoff_bottoms = [a + b + c for a, b, c in zip(a_dom, b_dom, a_tradeoff)]
    small_offsets = [0] * len(x)
    _label_segment(bars_a, a_dom, a_bottoms, small_offsets)
    _label_segment(bars_b, b_dom, b_bottoms, small_offsets)
    _label_segment(bars_a_tradeoff, a_tradeoff, a_tradeoff_bottoms, small_offsets)
    _label_segment(bars_b_tradeoff, b_tradeoff, b_tradeoff_bottoms, small_offsets)

    for idx, total in enumerate(counts):
        plt.text(
            x[idx],
            102,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x, envs, rotation=20, ha="right")
    plt.ylabel("Users (%)")
    plt.title(title)
    plt.ylim(0, 110)
    plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    envs = _parse_csv(args.env)

    sched_users: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {
        "anki_sm2": {},
        "memrise": {},
        "fsrs6_default": {},
        "fsrs3_default": {},
    }
    duplicate_count = 0
    target_dr = round(float(args.fsrs6_default_dr), 2)
    target_dr_fsrs3 = round(float(args.fsrs3_default_dr), 2)

    for path in _iter_log_paths(log_root):
        try:
            meta, totals = _load_meta_totals(path)
        except ValueError:
            continue

        engine = _infer_engine(meta, path)
        if args.engine != "any" and engine != args.engine:
            continue
        priority = _infer_priority(meta, path)
        if args.priority != "any" and priority != args.priority:
            continue

        environment = meta.get("environment")
        scheduler = meta.get("scheduler")
        if not isinstance(environment, str):
            continue
        if envs and environment not in envs:
            continue
        if scheduler not in {"anki_sm2", "memrise", "fsrs6_default", "fsrs3_default"}:
            continue
        if scheduler == "fsrs6_default":
            desired = meta.get("desired_retention")
            if desired is None:
                continue
            try:
                desired_value = round(float(desired), 2)
            except (TypeError, ValueError):
                continue
            if desired_value != target_dr:
                continue
        if scheduler == "fsrs3_default":
            desired = meta.get("desired_retention")
            if desired is None:
                continue
            try:
                desired_value = round(float(desired), 2)
            except (TypeError, ValueError):
                continue
            if desired_value != target_dr_fsrs3:
                continue

        short_term_value = _normalize_bool(meta.get("short_term"))
        if args.short_term != "any":
            if args.short_term == "on":
                if short_term_value is not True:
                    continue
            elif short_term_value is True:
                continue
        if args.short_term_source != "any":
            source_value = meta.get("short_term_source")
            if source_value != args.short_term_source:
                continue
            if short_term_value is not True:
                continue

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
        if user_id is None:
            continue

        key = (environment, user_id)
        target = sched_users[scheduler]
        existing = target.get(key)
        mtime = path.stat().st_mtime
        if existing is not None:
            duplicate_count += 1
            if mtime <= existing["mtime"]:
                continue
        target[key] = {
            "metrics": {
                "memorized_average": memorized_average,
                "memorized_per_minute": memorized_per_minute,
            },
            "mtime": mtime,
        }

    comparisons = [
        ("anki_sm2", "memrise"),
        ("fsrs6_default", "anki_sm2"),
        ("fsrs6_default", "memrise"),
        ("fsrs3_default", "fsrs6_default"),
    ]
    results: List[Dict[str, Any]] = []
    for left, right in comparisons:
        for env in envs:
            left_for_env = {
                user_id: payload["metrics"]
                for (env_key, user_id), payload in sched_users[left].items()
                if env_key == env
            }
            right_for_env = {
                user_id: payload["metrics"]
                for (env_key, user_id), payload in sched_users[right].items()
                if env_key == env
            }
            eligible_users = set(left_for_env) & set(right_for_env)
            if not eligible_users:
                continue
            counts = {
                "a_dominates": 0,
                "b_dominates": 0,
                "a_b_tradeoff": 0,
                "b_a_tradeoff": 0,
                "neither": 0,
            }
            neither_users: List[int] = []
            for user_id in eligible_users:
                label = _dominance_label(
                    left_for_env[user_id],
                    right_for_env[user_id],
                    epsilon=args.epsilon,
                )
                counts[label] += 1
                if label == "neither":
                    neither_users.append(user_id)
            total = len(eligible_users)
            results.append(
                {
                    "environment": env,
                    "pair": f"{left}_vs_{right}",
                    "left": left,
                    "right": right,
                    "user_count": total,
                    "a_dominates": counts["a_dominates"],
                    "b_dominates": counts["b_dominates"],
                    "a_b_tradeoff": counts["a_b_tradeoff"],
                    "b_a_tradeoff": counts["b_a_tradeoff"],
                    "neither": counts["neither"],
                    "a_dominates_pct": counts["a_dominates"] / total,
                    "b_dominates_pct": counts["b_dominates"] / total,
                    "a_b_tradeoff_pct": counts["a_b_tradeoff"] / total,
                    "b_a_tradeoff_pct": counts["b_a_tradeoff"] / total,
                    "neither_pct": counts["neither"] / total,
                    "neither_users": sorted(neither_users),
                }
            )

    results_path = args.results_path or (
        log_root / "simulation_results_retention_sweep_dominance.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if duplicate_count:
        print(f"Note: skipped {duplicate_count} duplicate logs (kept newest per user).")

    if args.no_plot:
        print(f"Wrote {results_path}")
        return

    plot_dir = args.plot_dir or (
        REPO_ROOT / "experiments" / "retention_sweep" / "plots" / "dominance"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()
    for left, right in comparisons:
        pair_results = [
            item
            for item in results
            if item.get("left") == left and item.get("right") == right
        ]
        if not pair_results:
            continue
        if left == "anki_sm2" and right == "memrise":
            output_name = "dominance_sm2_memrise.png"
        else:
            output_name = f"dominance_{left}_vs_{right}.png"
        output_path = plot_dir / output_name
        if left == "fsrs6_default":
            title = f"{left} (DR={target_dr:.2f}) vs {right} dominance (per user)"
        elif left == "fsrs3_default":
            title = f"{left} (DR={target_dr_fsrs3:.2f}) vs {right} dominance (per user)"
        elif right == "fsrs6_default":
            title = f"{left} vs {right} (DR={target_dr:.2f}) dominance (per user)"
        elif right == "fsrs3_default":
            title = f"{left} vs {right} (DR={target_dr_fsrs3:.2f}) dominance (per user)"
        else:
            title = f"{left} vs {right} dominance (per user)"
        _plot_dominance(
            pair_results,
            output_path,
            title,
            a_label=left,
            b_label=right,
        )
    print(f"Wrote {results_path}")
    for left, right in comparisons:
        if left == "anki_sm2" and right == "memrise":
            output_name = "dominance_sm2_memrise.png"
        else:
            output_name = f"dominance_{left}_vs_{right}.png"
        output_path = plot_dir / output_name
        if output_path.exists():
            print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
