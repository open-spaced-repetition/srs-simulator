from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

from tqdm import tqdm

from simulator.subprocess_progress import progress_event_from_payload, try_parse_json


def run_command_with_progress(
    *,
    cmd: list[str],
    env: dict[str, str],
    progress_bar: tqdm | None,
    overall_bar: tqdm | None,
    progress_lock: threading.RLock | None,
    write_line: Callable[[str], None] | None = None,
) -> int:
    """Run a subprocess and optionally consume JSON progress events from stdout.

    If progress_bar is None, stdout is not captured and the subprocess inherits the
    parent's stdio (legacy sequential behavior).
    """
    if progress_bar is None:
        result = subprocess.run(cmd, check=False, env=env)
        return result.returncode

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

    last_label: str | None = None
    overall_label: str | None = None
    overall_completed = 0

    def _write(text: str) -> None:
        if write_line is not None:
            write_line(text)
        else:
            progress_bar.write(text)

    if process.stdout is not None:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            payload = try_parse_json(line)
            if payload is None:
                if progress_lock is not None:
                    with progress_lock:
                        _write(line)
                else:
                    _write(line)
                continue
            event = progress_event_from_payload(payload)
            if event is None:
                continue
            label = event.label
            completed = event.completed
            total = event.total

            if progress_lock is not None:
                progress_lock.acquire()
            try:
                if overall_bar is not None:
                    if label and label != overall_label:
                        overall_label = label
                        overall_completed = 0
                    if completed < overall_completed:
                        overall_completed = 0
                    delta_overall = completed - overall_completed
                    if delta_overall > 0:
                        overall_bar.update(delta_overall)
                        overall_completed = completed

                if label and label != last_label:
                    reset_total = total if total > 0 else progress_bar.total
                    progress_bar.reset(total=reset_total)
                    progress_bar.set_description_str(label)
                    last_label = label
                    if completed > 0:
                        progress_bar.update(completed)
                    else:
                        progress_bar.refresh()
                    continue

                if total > 0 and progress_bar.total != total:
                    progress_bar.total = total
                if completed < progress_bar.n:
                    progress_bar.reset(total=progress_bar.total)
                    if completed > 0:
                        progress_bar.update(completed)
                    else:
                        progress_bar.refresh()
                elif completed > progress_bar.n:
                    progress_bar.update(completed - progress_bar.n)
            finally:
                if progress_lock is not None:
                    progress_lock.release()

    return process.wait()
