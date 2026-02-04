from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_csv(path: Path, metric: str) -> tuple[list[int], list[float]]:
    days: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            day_raw = row.get("day", "")
            value_raw = row.get(metric, "")
            if day_raw == "" or value_raw == "":
                continue
            days.append(int(float(day_raw)))
            values.append(float(value_raw) / (1024 * 1024))
    return days, values


def _label_for(path: Path) -> str:
    name = path.stem
    if name.startswith("batch_"):
        name = name[len("batch_") :]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot batch GPU memory usage from CSV logs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a single batch CSV log.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs") / "retention_sweep" / "batch_logs",
        help="Directory containing batch GPU CSV logs.",
    )
    parser.add_argument(
        "--metric",
        choices=["gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"],
        default="gpu_peak_reserved_bytes",
        help="Which GPU memory metric to plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        csv_files = [args.csv]
    else:
        csv_files = sorted(args.log_dir.glob("batch_*.csv"))
    if not csv_files:
        raise SystemExit("No batch GPU CSV logs found.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for path in csv_files:
        days, values = _load_csv(path, args.metric)
        if not values:
            continue
        ax.plot(days, values, label=_label_for(path))
    ax.set_xlabel("Day")
    ax.set_ylabel("MiB")
    ax.set_title(f"Batch GPU memory ({args.metric})")
    if len(csv_files) <= 10:
        ax.legend()
    fig.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
