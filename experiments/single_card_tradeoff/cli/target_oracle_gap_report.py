from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.target_search.comparison import (  # noqa: E402
    compare_target_answers_to_oracle,
    oracle_gap_row,
    read_target_answer_records,
    resolve_target_answers_path,
    summarize_oracle_gaps,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare constrained target answers against an oracle target frontier "
            "answer file."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--candidate-target-answers",
        "--candidate",
        dest="candidate_target_answers",
        type=Path,
        required=True,
        help="Candidate target_answers.csv, or a directory containing it.",
    )
    parser.add_argument(
        "--oracle-target-answers",
        "--oracle",
        dest="oracle_target_answers",
        type=Path,
        required=True,
        help="Oracle target_answers.csv, or a directory containing it.",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance when matching target values.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the candidate answer file directory "
            "or the candidate directory itself."
        ),
    )
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_out_dir(candidate_path: Path) -> Path:
    if candidate_path.is_dir():
        return candidate_path
    return candidate_path.parent


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = resolve_target_answers_path(args.candidate_target_answers)
    oracle_path = resolve_target_answers_path(args.oracle_target_answers)
    if not candidate_path.exists():
        raise SystemExit(f"--candidate-target-answers does not exist: {candidate_path}")
    if not oracle_path.exists():
        raise SystemExit(f"--oracle-target-answers does not exist: {oracle_path}")
    if args.target_tolerance < 0.0 or not math.isfinite(args.target_tolerance):
        raise SystemExit("--target-tolerance must be finite and >= 0.")

    candidate_records = read_target_answer_records(candidate_path)
    oracle_records = read_target_answer_records(oracle_path)
    gaps = compare_target_answers_to_oracle(
        candidate_records,
        oracle_records,
        target_tolerance=args.target_tolerance,
    )
    rows = [oracle_gap_row(gap) for gap in gaps]
    summary = summarize_oracle_gaps(gaps)

    out_dir = args.out_dir or _default_out_dir(args.candidate_target_answers)
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps_path = out_dir / "target_oracle_gaps.csv"
    metadata_path = out_dir / "target_oracle_gaps_metadata.json"
    _write_csv(gaps_path, rows)
    metadata = {
        "candidate_target_answers": str(candidate_path),
        "oracle_target_answers": str(oracle_path),
        "target_tolerance": args.target_tolerance,
        "summary": summary,
        "outputs": {
            "target_oracle_gaps": str(gaps_path),
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote oracle gap report: {gaps_path}")
    print(f"Wrote metadata: {metadata_path}")
    return metadata


def main_from_args(args: argparse.Namespace) -> None:
    run_report(args)


def main() -> None:
    main_from_args(parse_args())


if __name__ == "__main__":
    main()
