from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent import futures

from tqdm import tqdm


@dataclass(frozen=True)
class FanoutJob:
    user_id: int
    cmd: list[str]
    env: dict[str, str]
    echo_prefix: str = ""


@dataclass
class FanoutBars:
    users_bar: tqdm
    overall_bar: tqdm | None
    worker_bars: dict[int, tqdm]
    lock: threading.RLock | None
    worker_position_base: int
    use_parent_progress: bool

    def close(self) -> None:
        self.users_bar.close()
        if self.overall_bar is not None:
            self.overall_bar.close()
        for bar in self.worker_bars.values():
            bar.close()

    def ensure_worker_bar(self, slot: int) -> tqdm | None:
        if not self.use_parent_progress:
            return None
        bar = self.worker_bars.get(slot)
        if bar is not None:
            return bar
        bar = tqdm(
            total=0,
            desc="idle",
            unit="day",
            file=sys.stderr,
            ascii=True,
            position=self.worker_position_base + slot,
            leave=False,
        )
        self.worker_bars[slot] = bar
        return bar


def create_fanout_bars(
    *,
    user_count: int,
    show_overall: bool,
    overall_total: int | None,
    use_parent_progress: bool,
    max_parallel: int,
) -> FanoutBars:
    overall_bar = None
    users_position = 0
    if show_overall:
        overall_bar = tqdm(
            total=overall_total,
            desc="overall",
            unit="day",
            file=sys.stderr,
            ascii=True,
            position=0,
            leave=True,
        )
        users_position = 1

    users_bar = tqdm(
        total=user_count,
        desc="users",
        unit="user",
        file=sys.stderr,
        ascii=True,
        position=users_position,
        leave=True,
    )

    worker_bars: dict[int, tqdm] = {}
    lock = None
    worker_position_base = users_position + 1
    if use_parent_progress:
        lock = threading.RLock()
        tqdm.set_lock(lock)
    return FanoutBars(
        users_bar=users_bar,
        overall_bar=overall_bar,
        worker_bars=worker_bars,
        lock=lock,
        worker_position_base=worker_position_base,
        use_parent_progress=use_parent_progress,
    )


def _reset_worker_bar(progress_bar: tqdm, label: str | None = None) -> None:
    progress_bar.reset(total=0)
    if label is not None:
        progress_bar.set_description_str(label)
    progress_bar.refresh()


def run_fanout(
    *,
    user_ids: list[int],
    max_parallel: int,
    dry_run: bool,
    show_commands: bool,
    fail_fast: bool,
    sleep_seconds: float,
    use_parent_progress: bool,
    bars: FanoutBars,
    build_job: Callable[[int, int | None, int], FanoutJob],
    run_job: Callable[
        [FanoutJob, tqdm | None, tqdm | None, threading.RLock | None], int
    ],
) -> tuple[int, int | None]:
    """Run jobs for all user_ids.

    Returns (failures, first_failure_code).
    """
    failures = 0
    first_failure_code: int | None = None

    # Sequential (including dry-run): preserve legacy behavior.
    if dry_run or max_parallel == 1:
        for index, user_id in enumerate(user_ids):
            job = build_job(user_id, None, index)
            if dry_run or show_commands:
                bars.users_bar.write(
                    f"[{user_id}] {job.echo_prefix}{' '.join(job.cmd)}"
                )
            if not dry_run:
                returncode = run_job(job, None, None, None)
                if returncode != 0:
                    failures += 1
                    bars.users_bar.write(
                        f"[{user_id}] FAILED with exit code {returncode}"
                    )
                    if first_failure_code is None:
                        first_failure_code = returncode
                    if fail_fast:
                        bars.users_bar.update(1)
                        return failures, first_failure_code
            bars.users_bar.update(1)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        return failures, first_failure_code

    # Parallel.
    pending: dict[futures.Future[int], tuple[int, int]] = {}
    available_slots = list(range(max_parallel))
    user_iter: Iterator[int] = iter(user_ids)
    stop_scheduling = False

    def submit_next(executor: futures.Executor) -> bool:
        nonlocal failures, first_failure_code
        try:
            user_id = next(user_iter)
        except StopIteration:
            return False
        if not available_slots:
            return False
        slot = available_slots.pop(0)
        job = build_job(user_id, slot, 0)
        if show_commands:
            bars.users_bar.write(f"[{user_id}] {job.echo_prefix}{' '.join(job.cmd)}")
        worker_bar = bars.ensure_worker_bar(slot)
        if worker_bar is not None:
            _reset_worker_bar(worker_bar, f"u{user_id}")
        future = executor.submit(
            run_job,
            job,
            worker_bar,
            bars.overall_bar,
            bars.lock,
        )
        pending[future] = (user_id, slot)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return True

    with futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for _ in range(max_parallel):
            if not submit_next(executor):
                break
        while pending:
            done, _ = futures.wait(
                pending,
                return_when=futures.FIRST_COMPLETED,
            )
            for future in done:
                user_id, slot = pending.pop(future)
                try:
                    returncode = future.result()
                except futures.CancelledError:
                    bars.users_bar.update(1)
                    worker_bar = bars.worker_bars.get(slot)
                    if worker_bar is not None:
                        _reset_worker_bar(worker_bar, "idle")
                    available_slots.append(slot)
                    continue
                except Exception as exc:  # pragma: no cover
                    failures += 1
                    bars.users_bar.write(f"[{user_id}] FAILED with exception: {exc}")
                    if first_failure_code is None:
                        first_failure_code = 1
                    if fail_fast:
                        stop_scheduling = True
                    bars.users_bar.update(1)
                    worker_bar = bars.worker_bars.get(slot)
                    if worker_bar is not None:
                        _reset_worker_bar(worker_bar, "idle")
                    available_slots.append(slot)
                    continue

                if returncode != 0:
                    failures += 1
                    bars.users_bar.write(
                        f"[{user_id}] FAILED with exit code {returncode}"
                    )
                    if first_failure_code is None:
                        first_failure_code = returncode
                    if fail_fast:
                        stop_scheduling = True

                bars.users_bar.update(1)
                worker_bar = bars.worker_bars.get(slot)
                if worker_bar is not None:
                    _reset_worker_bar(worker_bar, "idle")
                available_slots.append(slot)

            if stop_scheduling:
                for future in list(pending):
                    future.cancel()
                continue
            while len(pending) < max_parallel:
                if not submit_next(executor):
                    break

    return failures, first_failure_code
