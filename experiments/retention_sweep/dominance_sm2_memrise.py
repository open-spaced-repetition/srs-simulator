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
            "Compare Anki-SM-2 vs Memrise dominance per user using retention_sweep logs."
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
    sm2_metrics: Dict[str, float],
    memrise_metrics: Dict[str, float],
    *,
    epsilon: float,
) -> str:
    mem_cmp = _compare(
        sm2_metrics["memorized_average"],
        memrise_metrics["memorized_average"],
        epsilon,
    )
    eff_cmp = _compare(
        sm2_metrics["memorized_per_minute"],
        memrise_metrics["memorized_per_minute"],
        epsilon,
    )
    if mem_cmp > 0 and eff_cmp > 0:
        return "sm2_dominates"
    if mem_cmp < 0 and eff_cmp < 0:
        return "memrise_dominates"
    if mem_cmp > 0 and eff_cmp < 0:
        return "sm2_memrise_tradeoff"
    if mem_cmp < 0 and eff_cmp > 0:
        return "memrise_sm2_tradeoff"
    return "neither"


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _plot_dominance(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    envs = [item["environment"] for item in results]
    sm2 = [item["sm2_dominates_pct"] * 100 for item in results]
    memrise = [item["memrise_dominates_pct"] * 100 for item in results]
    sm2_tradeoff = [item["sm2_memrise_tradeoff_pct"] * 100 for item in results]
    memrise_tradeoff = [item["memrise_sm2_tradeoff_pct"] * 100 for item in results]
    neither = [item["neither_pct"] * 100 for item in results]
    counts = [item["user_count"] for item in results]

    x = np.arange(len(envs))
    plt.figure(figsize=(10, 6))
    bars_sm2 = plt.bar(x, sm2, label="SM-2 dominates", color="#1f77b4")
    bars_memrise = plt.bar(
        x,
        memrise,
        bottom=sm2,
        label="Memrise dominates",
        color="#2ca02c",
    )
    bars_sm2_tradeoff = plt.bar(
        x,
        sm2_tradeoff,
        bottom=[a + b for a, b in zip(sm2, memrise)],
        label="SM-2 higher memorized, lower efficiency",
        color="#ff7f0e",
    )
    bars_memrise_tradeoff = plt.bar(
        x,
        memrise_tradeoff,
        bottom=[a + b + c for a, b, c in zip(sm2, memrise, sm2_tradeoff)],
        label="Memrise higher memorized, lower efficiency",
        color="#d62728",
    )
    bars_neither = plt.bar(
        x,
        neither,
        bottom=[
            a + b + c + d
            for a, b, c, d in zip(sm2, memrise, sm2_tradeoff, memrise_tradeoff)
        ],
        label="Neither",
        color="#7f7f7f",
    )

    def _label_segment(bars, values, bottoms):
        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue
            y = bottom + value / 2
            text = f"{value:.1f}%"
            if value < 3.0:
                y = bottom + value + 1.2
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

    sm2_bottoms = [0.0] * len(x)
    memrise_bottoms = sm2
    sm2_tradeoff_bottoms = [a + b for a, b in zip(sm2, memrise)]
    memrise_tradeoff_bottoms = [
        a + b + c for a, b, c in zip(sm2, memrise, sm2_tradeoff)
    ]
    neither_bottoms = [
        a + b + c + d
        for a, b, c, d in zip(sm2, memrise, sm2_tradeoff, memrise_tradeoff)
    ]
    _label_segment(bars_sm2, sm2, sm2_bottoms)
    _label_segment(bars_memrise, memrise, memrise_bottoms)
    _label_segment(bars_sm2_tradeoff, sm2_tradeoff, sm2_tradeoff_bottoms)
    _label_segment(bars_memrise_tradeoff, memrise_tradeoff, memrise_tradeoff_bottoms)
    _label_segment(bars_neither, neither, neither_bottoms)

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
    envs = _parse_csv(args.environments)

    sm2_users: Dict[Tuple[str, int], Dict[str, float]] = {}
    memrise_users: Dict[Tuple[str, int], Dict[str, float]] = {}
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
        if scheduler not in {"anki_sm2", "memrise"}:
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
        target = sm2_users if scheduler == "anki_sm2" else memrise_users
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

    results: List[Dict[str, Any]] = []
    for env in envs:
        sm2_for_env = {
            user_id: payload["metrics"]
            for (env_key, user_id), payload in sm2_users.items()
            if env_key == env
        }
        memrise_for_env = {
            user_id: payload["metrics"]
            for (env_key, user_id), payload in memrise_users.items()
            if env_key == env
        }
        eligible_users = set(sm2_for_env) & set(memrise_for_env)
        if not eligible_users:
            continue
        counts = {
            "sm2_dominates": 0,
            "memrise_dominates": 0,
            "sm2_memrise_tradeoff": 0,
            "memrise_sm2_tradeoff": 0,
            "neither": 0,
        }
        for user_id in eligible_users:
            label = _dominance_label(
                sm2_for_env[user_id],
                memrise_for_env[user_id],
                epsilon=args.epsilon,
            )
            counts[label] += 1
        total = len(eligible_users)
        results.append(
            {
                "environment": env,
                "user_count": total,
                "sm2_dominates": counts["sm2_dominates"],
                "memrise_dominates": counts["memrise_dominates"],
                "sm2_memrise_tradeoff": counts["sm2_memrise_tradeoff"],
                "memrise_sm2_tradeoff": counts["memrise_sm2_tradeoff"],
                "neither": counts["neither"],
                "sm2_dominates_pct": counts["sm2_dominates"] / total,
                "memrise_dominates_pct": counts["memrise_dominates"] / total,
                "sm2_memrise_tradeoff_pct": counts["sm2_memrise_tradeoff"] / total,
                "memrise_sm2_tradeoff_pct": counts["memrise_sm2_tradeoff"] / total,
                "neither_pct": counts["neither"] / total,
            }
        )

    results_path = args.results_path or (
        log_root / "simulation_results_retention_sweep_sm2_memrise_dominance.json"
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
    output_path = plot_dir / "dominance_sm2_memrise.png"
    title = "SM-2 vs Memrise dominance (per user)"
    _plot_dominance(results, output_path, title)
    print(f"Wrote {results_path}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
