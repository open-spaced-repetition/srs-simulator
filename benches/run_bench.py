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
SIMULATE = ROOT / "simulate.py"

SCENARIOS: list[Scenario] = [
    Scenario(
        name="event_fsrs6_fsrs6",
        args=[
            "--engine",
            "event",
            "--environment",
            "fsrs6",
            "--scheduler",
            "fsrs6",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
    Scenario(
        name="event_lstm_lstm",
        args=[
            "--engine",
            "event",
            "--environment",
            "lstm",
            "--scheduler",
            "lstm",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
    Scenario(
        name="vectorized_lstm_lstm",
        args=[
            "--engine",
            "vectorized",
            "--environment",
            "lstm",
            "--scheduler",
            "lstm",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
    Scenario(
        name="vectorized_fsrs6_fsrs6",
        args=[
            "--engine",
            "vectorized",
            "--environment",
            "fsrs6",
            "--scheduler",
            "fsrs6",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
    Scenario(
        name="vectorized_lstm_fsrs6",
        args=[
            "--engine",
            "vectorized",
            "--environment",
            "lstm",
            "--scheduler",
            "fsrs6",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
    Scenario(
        name="vectorized_fsrs6_lstm",
        args=[
            "--engine",
            "vectorized",
            "--environment",
            "fsrs6",
            "--scheduler",
            "lstm",
            "--user-id",
            "1",
            "--days",
            "365",
            "--deck",
            "1000",
            "--seed",
            "42",
        ],
    ),
]


def _iter_scenarios(
    names: Iterable[str] | None, override_engine: str | None
) -> list[Scenario]:
    if names:
        requested = {name.strip() for name in names if name.strip()}
        scenarios = [scenario for scenario in SCENARIOS if scenario.name in requested]
        if not scenarios:
            available = ", ".join(s.name for s in SCENARIOS)
            raise SystemExit(f"No matching scenarios. Available: {available}")
    else:
        scenarios = list(SCENARIOS)

    if override_engine is None:
        return scenarios
    overridden: list[Scenario] = []
    for scenario in scenarios:
        args = list(scenario.args)
        if "--engine" in args:
            idx = args.index("--engine")
            if idx + 1 < len(args):
                args[idx + 1] = override_engine
        else:
            args = ["--engine", override_engine, *args]
        overridden.append(Scenario(name=scenario.name, args=args))
    return overridden


def _parse_simulation_time(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.startswith("Simulation time:"):
            suffix = line.split("Simulation time:", 1)[1].strip()
            if suffix.endswith("s"):
                suffix = suffix[:-1]
            try:
                return float(suffix)
            except ValueError:
                return None
    return None


def _run_once(args: list[str], env: dict[str, str]) -> tuple[float, float | None]:
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(SIMULATE), *args],
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
    sim_time = _parse_simulation_time(completed.stdout)
    return elapsed, sim_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulator perf baselines.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to run (repeatable).",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized"],
        default=None,
        help="Override scenario engine.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat each scenario N times."
    )
    parser.add_argument(
        "--srs-benchmark-root",
        default=None,
        help="Path to the srs-benchmark repo (passed to simulate.py).",
    )
    args = parser.parse_args()

    scenarios = _iter_scenarios(args.scenario, args.engine)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    for scenario in scenarios:
        times: list[float] = []
        sim_times: list[float] = []
        for _ in range(max(1, args.repeat)):
            cmd = list(scenario.args)
            cmd.extend(["--no-progress", "--no-plot", "--no-log"])
            if args.srs_benchmark_root:
                cmd.extend(["--srs-benchmark-root", args.srs_benchmark_root])
            elapsed, sim_time = _run_once(cmd, env)
            times.append(elapsed)
            if sim_time is not None:
                sim_times.append(sim_time)
        best = min(times)
        avg = sum(times) / len(times)
        label = scenario.name
        sim_best = min(sim_times) if sim_times else None
        sim_avg = sum(sim_times) / len(sim_times) if sim_times else None
        print(f"{label}: wall best {best:.2f}s avg {avg:.2f}s", end="")
        if sim_best is not None and sim_avg is not None:
            print(f" | sim best {sim_best:.2f}s avg {sim_avg:.2f}s")
        else:
            print()


if __name__ == "__main__":
    main()
