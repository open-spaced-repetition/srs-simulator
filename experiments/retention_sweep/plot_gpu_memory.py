from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_log(path: Path) -> tuple[dict, dict]:
    meta = {}
    daily = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload.get("data", {}) or {}
            elif payload.get("type") == "daily":
                daily = payload.get("data", {}) or {}
    if not daily:
        raise ValueError("No daily record found in log.")
    return meta, daily


def _plot(meta: dict, daily: dict, *, output: Path | None, show: bool) -> None:
    gpu_bytes = daily.get("gpu_peak_bytes")
    if not gpu_bytes:
        raise ValueError("daily.gpu_peak_bytes missing in log.")
    reviews = daily.get("reviews")
    phase_reviews = daily.get("phase_reviews")
    days = list(range(len(gpu_bytes)))
    gpu_mib = [value / (1024 * 1024) for value in gpu_bytes]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    axes[0].plot(days, gpu_mib, color="tab:purple", label="GPU peak (MiB)")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("MiB")
    axes[0].set_title("Daily GPU peak memory")
    axes[0].legend()

    if reviews:
        axes[1].scatter(reviews, gpu_mib, s=12, alpha=0.7, label="Total reviews/day")
        if phase_reviews:
            long_reviews = phase_reviews
            short_reviews = [r - l for r, l in zip(reviews, phase_reviews)]
            axes[1].scatter(
                long_reviews,
                gpu_mib,
                s=12,
                alpha=0.35,
                label="Long reviews/day",
            )
            axes[1].scatter(
                short_reviews,
                gpu_mib,
                s=12,
                alpha=0.35,
                label="Short reviews/day",
            )
        axes[1].set_xlabel("Reviews/day")
        axes[1].set_ylabel("GPU peak (MiB)")
        axes[1].set_title("GPU peak vs. reviews")
        axes[1].legend()
    else:
        axes[1].axis("off")

    short_reviews = daily.get("short_reviews")
    short_loops = daily.get("short_loops")
    short_per_loop = daily.get("short_reviews_per_loop")
    if short_reviews or short_loops or short_per_loop:
        if short_reviews:
            axes[2].scatter(
                short_reviews,
                gpu_mib,
                s=12,
                alpha=0.7,
                label="Short reviews/day",
            )
        if short_per_loop:
            axes[2].scatter(
                short_per_loop,
                gpu_mib,
                s=12,
                alpha=0.5,
                label="Short reviews/loop",
            )
        if short_loops:
            axes[2].scatter(
                short_loops,
                gpu_mib,
                s=12,
                alpha=0.4,
                label="Short loops/day",
            )
        axes[2].set_xlabel("Short-term activity")
        axes[2].set_ylabel("GPU peak (MiB)")
        axes[2].set_title("GPU peak vs. short-term activity")
        axes[2].legend()
    else:
        axes[2].axis("off")

    title_bits = []
    if meta.get("environment"):
        title_bits.append(f"env={meta['environment']}")
    if meta.get("scheduler"):
        title_bits.append(f"sched={meta['scheduler']}")
    if meta.get("engine"):
        title_bits.append(f"engine={meta['engine']}")
    if title_bits:
        fig.suptitle(" ".join(title_bits))
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot daily GPU peak memory vs review counts.",
        allow_abbrev=False,
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to a log jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window.",
    )
    args = parser.parse_args()

    meta, daily = _load_log(args.log)
    _plot(meta, daily, output=args.out, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
