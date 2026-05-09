from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl_scheduler.portfolio_selection import (
    LightweightSelectionPool,
    SelectionPayload,
    SelectionPoint,
    hypervolume_2d,
    select_sms_emoa_survivor_indices,
)


@dataclass(frozen=True)
class SelectionCase:
    payload: SelectionPayload


def _case(
    *,
    rng: random.Random,
    candidate_count: int,
    population_size: int,
    candidate_id_offset: int,
) -> SelectionCase:
    baseline_memorized = tuple(rng.uniform(3500.0, 7200.0) for _index in range(23))
    baseline_negative_time = tuple(-rng.uniform(15.0, 140.0) for _index in range(23))
    memorized = tuple(rng.uniform(3500.0, 7400.0) for _index in range(candidate_count))
    time_average = tuple(rng.uniform(12.0, 145.0) for _index in range(candidate_count))
    reference_memorized = min(baseline_memorized) - 500.0
    reference_negative_time = min(baseline_negative_time) - 10.0
    return SelectionCase(
        payload=SelectionPayload(
            baseline_memorized=baseline_memorized,
            baseline_negative_time=baseline_negative_time,
            candidate_ids=tuple(
                candidate_id_offset + index for index in range(candidate_count)
            ),
            memorized=memorized,
            time_average=time_average,
            reference_memorized=reference_memorized,
            reference_negative_time=reference_negative_time,
            population_size=population_size,
        )
    )


def _run_local_selector(case: SelectionCase) -> tuple[tuple[int, ...], float]:
    started = time.perf_counter()
    survivor_indices = select_sms_emoa_survivor_indices(case.payload)
    elapsed = time.perf_counter() - started
    return _survivor_ids(case.payload, survivor_indices), elapsed


def _run_process_pool_selector(
    cases: list[SelectionCase],
    *,
    worker_count: int,
) -> tuple[list[tuple[int, ...]], float, int]:
    with LightweightSelectionPool(max_workers=worker_count) as executor:
        started = time.perf_counter()
        results = executor.map([case.payload for case in cases])
        elapsed = time.perf_counter() - started
        child_rss = sum(_rss_bytes(pid) for pid in executor.child_pids)
    return (
        [
            _survivor_ids(case.payload, result.survivor_indices)
            for case, result in zip(cases, results, strict=True)
        ],
        elapsed,
        child_rss,
    )


def _survivor_ids(
    payload: SelectionPayload,
    survivor_indices: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(payload.candidate_ids[index] for index in survivor_indices)


def _survivor_hv(payload: SelectionPayload, survivor_ids: tuple[int, ...]) -> float:
    by_id = {
        candidate_id: index for index, candidate_id in enumerate(payload.candidate_ids)
    }
    return hypervolume_2d(
        [
            *[
                SelectionPoint(memorized, negative_time)
                for memorized, negative_time in zip(
                    payload.baseline_memorized,
                    payload.baseline_negative_time,
                    strict=True,
                )
            ],
            *[
                SelectionPoint(
                    payload.memorized[by_id[candidate_id]],
                    -payload.time_average[by_id[candidate_id]],
                )
                for candidate_id in survivor_ids
            ],
        ],
        reference=SelectionPoint(
            payload.reference_memorized,
            payload.reference_negative_time,
        ),
    )


def _bench_local(cases: list[SelectionCase]) -> tuple[list[tuple[int, ...]], float]:
    seconds = 0.0
    survivor_ids: list[tuple[int, ...]] = []
    for case in cases:
        ids, elapsed = _run_local_selector(case)
        survivor_ids.append(ids)
        seconds += elapsed
    return survivor_ids, seconds


def _comparison_row(
    *,
    case_name: str,
    backend: str,
    cases: list[SelectionCase],
    baseline_ids: list[tuple[int, ...]],
    baseline_seconds: float,
    candidate_ids: list[tuple[int, ...]],
    candidate_seconds: float,
    child_rss_bytes: int,
) -> dict[str, str]:
    max_hv_diff = max(
        (
            abs(
                _survivor_hv(case.payload, expected)
                - _survivor_hv(case.payload, actual)
            )
            for case, expected, actual in zip(
                cases,
                baseline_ids,
                candidate_ids,
                strict=True,
            )
        ),
        default=0.0,
    )
    speedup = (
        baseline_seconds / candidate_seconds
        if candidate_seconds > 0.0
        else float("inf")
    )
    return {
        "case": case_name,
        "backend": backend,
        "seconds": f"{candidate_seconds:.4f}",
        "speedup_vs_local": f"{speedup:.2f}x",
        "survivor_ids_equal": str(baseline_ids == candidate_ids),
        "max_hv_diff": f"{max_hv_diff:.12g}",
        "parent_rss_mib": f"{_rss_bytes(os.getpid()) / 1048576.0:.1f}",
        "child_rss_mib": f"{child_rss_bytes / 1048576.0:.1f}",
    }


def _rss_bytes(pid: int | None) -> int:
    if pid is None:
        return 0
    statm = Path(f"/proc/{pid}/statm")
    try:
        pages = int(statm.read_text().split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return 0
    return pages * os.sysconf("SC_PAGE_SIZE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact SMS-EMOA survivor selection backends.",
        allow_abbrev=False,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--aggregate-users", type=int, default=128)
    parser.add_argument(
        "--process-workers",
        type=int,
        nargs="*",
        default=[8, 16, 32],
        help="Process-pool worker caps to compare against local selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    scenarios = [
        ("64 -> 32", 64, 32, args.repeats),
        ("96 -> 64", 96, 64, args.repeats),
        ("128 -> 64", 128, 64, args.repeats),
        (
            f"{args.aggregate_users} users * 128 candidates",
            128,
            64,
            args.aggregate_users,
        ),
    ]
    rows = []
    candidate_id_offset = 0
    for name, candidate_count, population_size, repeats in scenarios:
        cases = []
        for _index in range(repeats):
            cases.append(
                _case(
                    rng=rng,
                    candidate_count=candidate_count,
                    population_size=population_size,
                    candidate_id_offset=candidate_id_offset,
                )
            )
            candidate_id_offset += candidate_count
        local_ids, local_seconds = _bench_local(cases)
        rows.append(
            _comparison_row(
                case_name=name,
                backend="local",
                cases=cases,
                baseline_ids=local_ids,
                baseline_seconds=local_seconds,
                candidate_ids=local_ids,
                candidate_seconds=local_seconds,
                child_rss_bytes=0,
            )
        )
        for worker_count in args.process_workers:
            pool_ids, pool_seconds, child_rss_bytes = _run_process_pool_selector(
                cases,
                worker_count=worker_count,
            )
            rows.append(
                _comparison_row(
                    case_name=name,
                    backend=f"process_pool_{worker_count}",
                    cases=cases,
                    baseline_ids=local_ids,
                    baseline_seconds=local_seconds,
                    candidate_ids=pool_ids,
                    candidate_seconds=pool_seconds,
                    child_rss_bytes=child_rss_bytes,
                )
            )

    headers = [
        "case",
        "backend",
        "seconds",
        "speedup_vs_local",
        "survivor_ids_equal",
        "max_hv_diff",
        "parent_rss_mib",
        "child_rss_mib",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _header in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row[header] for header in headers) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
