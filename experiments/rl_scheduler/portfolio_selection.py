from __future__ import annotations

from collections.abc import Sequence
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


SELECTION_PROCESS_POOL_ENV = "FSRS6_PORTFOLIO_SELECTION_PROCESS_POOL"
SELECTION_PROCESS_POOL_WORKERS_ENV = "FSRS6_PORTFOLIO_SELECTION_WORKERS"
LEGACY_ADR_DIRECT_SELECTION_PROCESS_POOL_ENV = (
    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL"
)
LEGACY_ADR_DIRECT_SELECTION_PROCESS_POOL_WORKERS_ENV = (
    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_WORKERS"
)
DEFAULT_SELECTION_PROCESS_POOL_WORKERS = 32
DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS = 8

_DEFAULT_SELECTION_PROCESS_POOL_ENV_VARS = (
    SELECTION_PROCESS_POOL_ENV,
    LEGACY_ADR_DIRECT_SELECTION_PROCESS_POOL_ENV,
)
_DEFAULT_SELECTION_PROCESS_POOL_WORKER_ENV_VARS = (
    SELECTION_PROCESS_POOL_WORKERS_ENV,
    LEGACY_ADR_DIRECT_SELECTION_PROCESS_POOL_WORKERS_ENV,
)

CandidateT = TypeVar("CandidateT")


class CandidateMetricValues(Protocol):
    @property
    def memorized_average(self) -> float: ...

    @property
    def time_average(self) -> float: ...


class SelectionCandidate(Protocol):
    @property
    def candidate_id(self) -> int: ...

    @property
    def metrics(self) -> CandidateMetricValues: ...


@dataclass(frozen=True, slots=True)
class SelectionPoint:
    memorized_average: float
    negative_time_average: float


@dataclass(frozen=True, slots=True)
class SelectionPayload:
    baseline_memorized: tuple[float, ...]
    baseline_negative_time: tuple[float, ...]
    candidate_ids: tuple[int, ...]
    memorized: tuple[float, ...]
    time_average: tuple[float, ...]
    reference_memorized: float
    reference_negative_time: float
    population_size: int


@dataclass(frozen=True, slots=True)
class SelectionResult:
    survivor_indices: tuple[int, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SelectionTask(Generic[CandidateT]):
    candidates: tuple[CandidateT, ...]
    payload: SelectionPayload


class LightweightSelectionPool:
    def __init__(self, *, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive.")
        self._workers = [_SelectionWorkerProcess() for _index in range(max_workers)]
        self._closed = False

    @property
    def child_pids(self) -> tuple[int, ...]:
        return tuple(worker.pid for worker in self._workers if worker.pid is not None)

    def map(self, payloads: Sequence[SelectionPayload]) -> list[SelectionResult]:
        if self._closed:
            raise RuntimeError("selection pool is closed.")
        if not payloads:
            return []
        results: list[SelectionResult | None] = [None for _payload in payloads]
        errors: list[BaseException] = []
        tasks: queue.Queue[tuple[int, SelectionPayload]] = queue.Queue()
        for index, payload in enumerate(payloads):
            tasks.put((index, payload))

        def run_worker(worker: _SelectionWorkerProcess) -> None:
            while True:
                try:
                    index, payload = tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    results[index] = worker.run(payload)
                except BaseException as exc:
                    errors.append(exc)
                    return
                finally:
                    tasks.task_done()

        threads = [
            threading.Thread(target=run_worker, args=(worker,))
            for worker in self._workers[: min(len(self._workers), len(payloads))]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        missing = [index for index, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"selection pool did not finish tasks: {missing!r}")
        return [result for result in results if result is not None]

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.close()

    def __enter__(self) -> LightweightSelectionPool:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown()


def assert_selection_worker_is_lightweight() -> None:
    if "torch" in sys.modules:
        raise RuntimeError("selection worker imported torch unexpectedly")


def selection_payload_from_candidate_metrics(
    *,
    baseline_points: Sequence[SelectionPoint],
    candidates: Sequence[SelectionCandidate],
    population_size: int,
    reference: SelectionPoint,
) -> SelectionPayload:
    return SelectionPayload(
        baseline_memorized=tuple(point.memorized_average for point in baseline_points),
        baseline_negative_time=tuple(
            point.negative_time_average for point in baseline_points
        ),
        candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
        memorized=tuple(
            candidate.metrics.memorized_average for candidate in candidates
        ),
        time_average=tuple(candidate.metrics.time_average for candidate in candidates),
        reference_memorized=reference.memorized_average,
        reference_negative_time=reference.negative_time_average,
        population_size=population_size,
    )


def dominates(lhs: SelectionPoint, rhs: SelectionPoint) -> bool:
    no_worse = (
        lhs.memorized_average >= rhs.memorized_average
        and lhs.negative_time_average >= rhs.negative_time_average
    )
    strictly_better = (
        lhs.memorized_average > rhs.memorized_average
        or lhs.negative_time_average > rhs.negative_time_average
    )
    return no_worse and strictly_better


def non_dominated_indices(points: Sequence[SelectionPoint]) -> list[int]:
    if not points:
        return []
    sorted_points = sorted(
        enumerate(points),
        key=lambda item: (
            -item[1].memorized_average,
            -item[1].negative_time_average,
            item[0],
        ),
    )
    max_y_from_greater_x = float("-inf")
    indices: list[int] = []
    start = 0
    while start < len(sorted_points):
        x_value = sorted_points[start][1].memorized_average
        end = start + 1
        while (
            end < len(sorted_points)
            and sorted_points[end][1].memorized_average == x_value
        ):
            end += 1
        group = sorted_points[start:end]
        group_max_y = max(point.negative_time_average for _index, point in group)
        if group_max_y > max_y_from_greater_x:
            indices.extend(
                index
                for index, point in group
                if point.negative_time_average == group_max_y
            )
        max_y_from_greater_x = max(max_y_from_greater_x, group_max_y)
        start = end
    indices.sort()
    return indices


def hypervolume_2d(
    points: Sequence[SelectionPoint],
    *,
    reference: SelectionPoint,
) -> float:
    frontier = _frontier_points_sorted(points, reference=reference)
    if not frontier:
        return 0.0
    hv = 0.0
    previous_x = reference.memorized_average
    for _index, point in frontier:
        width = max(0.0, point.memorized_average - previous_x)
        height = max(0.0, point.negative_time_average - reference.negative_time_average)
        hv += width * height
        previous_x = max(previous_x, point.memorized_average)
    return float(hv)


def exclusive_hypervolume_contributions(
    *,
    baseline_points: Sequence[SelectionPoint],
    candidate_points: Sequence[SelectionPoint],
    reference: SelectionPoint,
) -> list[float]:
    contributions = [0.0 for _candidate in candidate_points]
    if not candidate_points:
        return contributions
    all_points = [*baseline_points, *candidate_points]
    baseline_count = len(baseline_points)
    frontier = _frontier_points_sorted(all_points, reference=reference)
    coordinate_counts: dict[tuple[float, float], int] = {}
    for _index, point in frontier:
        key = (point.memorized_average, point.negative_time_average)
        coordinate_counts[key] = coordinate_counts.get(key, 0) + 1
    for global_index, point in frontier:
        if global_index < baseline_count:
            continue
        candidate_index = global_index - baseline_count
        key = (point.memorized_average, point.negative_time_average)
        if coordinate_counts[key] > 1:
            continue
        left_boundary = max(
            [
                reference.memorized_average,
                *[
                    other.memorized_average
                    for other_index, other in enumerate(all_points)
                    if other_index != global_index
                    and other.memorized_average < point.memorized_average
                    and other.negative_time_average >= point.negative_time_average
                ],
            ]
        )
        if left_boundary >= point.memorized_average:
            continue
        local_reference = SelectionPoint(
            memorized_average=left_boundary,
            negative_time_average=reference.negative_time_average,
        )
        blockers = [
            SelectionPoint(
                memorized_average=min(other.memorized_average, point.memorized_average),
                negative_time_average=min(
                    other.negative_time_average,
                    point.negative_time_average,
                ),
            )
            for other_index, other in enumerate(all_points)
            if other_index != global_index
            and min(other.memorized_average, point.memorized_average) > left_boundary
            and min(other.negative_time_average, point.negative_time_average)
            > reference.negative_time_average
        ]
        rectangle_area = (point.memorized_average - left_boundary) * (
            point.negative_time_average - reference.negative_time_average
        )
        contributions[candidate_index] = max(
            0.0,
            rectangle_area - hypervolume_2d(blockers, reference=local_reference),
        )
    return contributions


def baseline_aware_candidate_ranks(
    *,
    baseline_points: Sequence[SelectionPoint],
    candidate_points: Sequence[SelectionPoint],
) -> list[int]:
    ranks = [-1 for _candidate in candidate_points]
    baseline_dominated = {
        index
        for index, candidate in enumerate(candidate_points)
        if any(dominates(baseline, candidate) for baseline in baseline_points)
    }
    remaining = [
        index
        for index in range(len(candidate_points))
        if index not in baseline_dominated
    ]
    rank = 0
    while remaining:
        layer_points = [
            *baseline_points,
            *[candidate_points[index] for index in remaining],
        ]
        nd = non_dominated_indices(layer_points)
        selected = [
            remaining[index - len(baseline_points)]
            for index in nd
            if index >= len(baseline_points)
        ]
        if not selected:
            break
        for index in selected:
            ranks[index] = rank
        selected_set = set(selected)
        remaining = [index for index in remaining if index not in selected_set]
        rank += 1
    worst_rank = rank + len(candidate_points) + 1
    return [rank if rank >= 0 else worst_rank for rank in ranks]


def select_sms_emoa_survivor_indices(payload: SelectionPayload) -> tuple[int, ...]:
    _validate_payload(payload)
    alive = list(range(len(payload.candidate_ids)))
    while len(alive) > payload.population_size:
        baseline_points = _baseline_points(payload)
        candidate_points = [
            SelectionPoint(
                memorized_average=payload.memorized[index],
                negative_time_average=-payload.time_average[index],
            )
            for index in alive
        ]
        ranks = baseline_aware_candidate_ranks(
            baseline_points=baseline_points,
            candidate_points=candidate_points,
        )
        contributions = exclusive_hypervolume_contributions(
            baseline_points=baseline_points,
            candidate_points=candidate_points,
            reference=SelectionPoint(
                memorized_average=payload.reference_memorized,
                negative_time_average=payload.reference_negative_time,
            ),
        )
        worst_rank = max(ranks)
        removal_candidates = [
            index for index, rank in enumerate(ranks) if rank == worst_rank
        ]
        remove_local_index = min(
            removal_candidates,
            key=lambda index: (
                contributions[index],
                payload.memorized[alive[index]],
                -payload.time_average[alive[index]],
                -payload.candidate_ids[alive[index]],
            ),
        )
        del alive[remove_local_index]
    return tuple(alive)


def select_sms_emoa_payload_timed(payload: SelectionPayload) -> SelectionResult:
    started = time.perf_counter()
    survivor_indices = select_sms_emoa_survivor_indices(payload)
    return SelectionResult(
        survivor_indices=survivor_indices,
        elapsed_seconds=time.perf_counter() - started,
    )


def selection_process_pool_worker_count(
    job_count: int,
    *,
    worker_env_vars: Sequence[str] = _DEFAULT_SELECTION_PROCESS_POOL_WORKER_ENV_VARS,
    default_workers: int = DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
) -> int:
    if job_count <= 1:
        return 0
    cpu_count = os.cpu_count() or 1
    env_name, raw_worker_count = _first_env_value(worker_env_vars)
    if raw_worker_count:
        try:
            worker_count = int(raw_worker_count)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a positive integer.") from exc
        if worker_count < 1:
            raise ValueError(f"{env_name} must be a positive integer.")
    else:
        worker_count = default_workers
    return min(job_count, cpu_count, worker_count)


def selection_process_pool_enabled(
    job_count: int,
    *,
    enabled_env_vars: Sequence[str] = _DEFAULT_SELECTION_PROCESS_POOL_ENV_VARS,
    default_min_jobs: int = DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
) -> bool:
    if job_count <= 1:
        return False
    env_name, raw_enabled = _first_env_value(enabled_env_vars)
    raw_enabled = raw_enabled.lower()
    if raw_enabled in {"1", "true", "yes", "on"}:
        return True
    if raw_enabled in {"0", "false", "no", "off"}:
        return False
    if raw_enabled:
        raise ValueError(f"{env_name} must be 1/true/on or 0/false/off.")
    return job_count >= default_min_jobs


def selection_executor(
    job_count: int,
    *,
    enabled_env_vars: Sequence[str] = _DEFAULT_SELECTION_PROCESS_POOL_ENV_VARS,
    worker_env_vars: Sequence[str] = _DEFAULT_SELECTION_PROCESS_POOL_WORKER_ENV_VARS,
    default_min_jobs: int = DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    default_workers: int = DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
) -> LightweightSelectionPool | None:
    if not selection_process_pool_enabled(
        job_count,
        enabled_env_vars=enabled_env_vars,
        default_min_jobs=default_min_jobs,
    ):
        return None
    max_workers = selection_process_pool_worker_count(
        job_count,
        worker_env_vars=worker_env_vars,
        default_workers=default_workers,
    )
    if max_workers <= 1:
        return None
    return LightweightSelectionPool(max_workers=max_workers)


def select_survivors_for_generation(
    *,
    tasks: Sequence[SelectionTask[CandidateT]],
    executor: LightweightSelectionPool | None,
) -> tuple[list[list[CandidateT]], list[float]]:
    if executor is None:
        results = [select_sms_emoa_payload_timed(task.payload) for task in tasks]
    else:
        results = executor.map([task.payload for task in tasks])
    survivors = [
        [task.candidates[index] for index in result.survivor_indices]
        for task, result in zip(tasks, results, strict=True)
    ]
    elapsed = [result.elapsed_seconds for result in results]
    return survivors, elapsed


def worker_main() -> int:
    assert_selection_worker_is_lightweight()
    while True:
        payload_data = _read_frame(sys.stdin.buffer)
        if payload_data is None:
            return 0
        try:
            payload = pickle.loads(payload_data)
            result: Any = select_sms_emoa_payload_timed(payload)
        except BaseException as exc:  # pragma: no cover - exercised through parent
            result = exc
        _write_frame(sys.stdout.buffer, pickle.dumps(result, protocol=5))
        sys.stdout.buffer.flush()


class _SelectionWorkerProcess:
    def __init__(self) -> None:
        command = [
            sys.executable,
            "-c",
            "from experiments.rl_scheduler.portfolio_selection import worker_main; "
            "raise SystemExit(worker_main())",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def run(self, payload: SelectionPayload) -> SelectionResult:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("selection worker pipes are unavailable.")
        _write_frame(
            self._process.stdin,
            pickle.dumps(payload, protocol=5),
        )
        self._process.stdin.flush()
        result_data = _read_frame(self._process.stdout)
        if result_data is None:
            stderr = self._read_stderr()
            raise RuntimeError(f"selection worker exited unexpectedly: {stderr}")
        result = pickle.loads(result_data)
        if isinstance(result, BaseException):
            raise result
        if not isinstance(result, SelectionResult):
            raise TypeError(f"selection worker returned {type(result).__name__}.")
        return result

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write(struct.pack("!Q", 0))
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return ""


def _read_frame(stream: Any) -> bytes | None:
    header = _read_exact(stream, 8)
    if header is None:
        return None
    size = struct.unpack("!Q", header)[0]
    if size == 0:
        return None
    data = _read_exact(stream, size)
    if data is None:
        raise EOFError("incomplete selection worker frame.")
    return data


def _write_frame(stream: Any, data: bytes) -> None:
    stream.write(struct.pack("!Q", len(data)))
    stream.write(data)


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _frontier_points_sorted(
    points: Sequence[SelectionPoint],
    *,
    reference: SelectionPoint,
) -> list[tuple[int, SelectionPoint]]:
    contributing = [
        (index, point)
        for index, point in enumerate(points)
        if point.memorized_average > reference.memorized_average
        and point.negative_time_average > reference.negative_time_average
    ]
    if not contributing:
        return []
    local_frontier = non_dominated_indices([point for _index, point in contributing])
    frontier = [contributing[index] for index in local_frontier]
    frontier.sort(
        key=lambda item: (
            item[1].memorized_average,
            item[1].negative_time_average,
            item[0],
        )
    )
    return frontier


def _baseline_points(payload: SelectionPayload) -> list[SelectionPoint]:
    return [
        SelectionPoint(
            memorized_average=memorized,
            negative_time_average=negative_time,
        )
        for memorized, negative_time in zip(
            payload.baseline_memorized,
            payload.baseline_negative_time,
            strict=True,
        )
    ]


def _validate_payload(payload: SelectionPayload) -> None:
    if payload.population_size < 1:
        raise ValueError("population_size must be positive.")
    baseline_count = len(payload.baseline_memorized)
    if len(payload.baseline_negative_time) != baseline_count:
        raise ValueError("baseline point arrays must have the same length.")
    candidate_count = len(payload.candidate_ids)
    if len(payload.memorized) != candidate_count:
        raise ValueError("candidate memorized array length mismatch.")
    if len(payload.time_average) != candidate_count:
        raise ValueError("candidate time_average array length mismatch.")


def _first_env_value(env_vars: Sequence[str]) -> tuple[str, str]:
    if not env_vars:
        raise ValueError("At least one environment variable name is required.")
    for env_name in env_vars:
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return env_name, raw
    return env_vars[0], ""
