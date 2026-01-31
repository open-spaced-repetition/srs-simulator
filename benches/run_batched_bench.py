from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Scenario:
    name: str
    args: list[str]


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "experiments" / "retention_sweep" / "run_sweep_users_batched.py"

SCENARIOS: list[Scenario] = [
    Scenario(
        name="batched_lstm_fsrs6",
        args=[
            "--env",
            "lstm",
            "--sched",
            "fsrs6",
            "--start-user",
            "1",
            "--end-user",
            "100",
            "--batch-size",
            "50",
            "--start-retention",
            "0.80",
            "--end-retention",
            "0.80",
            "--step",
            "0.01",
            "--days",
            "365",
            "--deck",
            "1000",
        ],
    ),
    Scenario(
        name="batched_fsrs6_fsrs6",
        args=[
            "--env",
            "fsrs6",
            "--sched",
            "fsrs6",
            "--start-user",
            "1",
            "--end-user",
            "100",
            "--batch-size",
            "50",
            "--start-retention",
            "0.80",
            "--end-retention",
            "0.80",
            "--step",
            "0.01",
            "--days",
            "365",
            "--deck",
            "1000",
        ],
    ),
]


def _iter_scenarios(names: Iterable[str] | None) -> list[Scenario]:
    if names:
        requested = {name.strip() for name in names if name.strip()}
        scenarios = [scenario for scenario in SCENARIOS if scenario.name in requested]
        if not scenarios:
            available = ", ".join(s.name for s in SCENARIOS)
            raise SystemExit(f"No matching scenarios. Available: {available}")
        return scenarios
    return list(SCENARIOS)


def _run_once(args: list[str], env: dict[str, str]) -> float:
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(SWEEP), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed with exit code {completed.returncode}.")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batched retention sweep baselines."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to run (repeatable).",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat each scenario N times."
    )
    parser.add_argument(
        "--srs-benchmark-root",
        default=None,
        help="Path to the srs-benchmark repo (passed to run_sweep_users_batched.py).",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device to pass to run_sweep_users_batched.py.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated CUDA device indices to distribute batches across.",
    )
    args = parser.parse_args()

    scenarios = _iter_scenarios(args.scenario)
    env = dict(os.environ)

    for scenario in scenarios:
        times: list[float] = []
        for _ in range(max(1, args.repeat)):
            cmd = list(scenario.args)
            cmd.extend(["--no-progress", "--no-log"])
            if args.srs_benchmark_root:
                cmd.extend(["--srs-benchmark-root", args.srs_benchmark_root])
            if args.torch_device:
                cmd.extend(["--torch-device", args.torch_device])
            if args.cuda_devices:
                cmd.extend(["--cuda-devices", args.cuda_devices])
            elapsed = _run_once(cmd, env)
            times.append(elapsed)
        best = min(times)
        avg = sum(times) / len(times)
        print(f"{scenario.name}: wall best {best:.2f}s avg {avg:.2f}s")


if __name__ == "__main__":
    main()
