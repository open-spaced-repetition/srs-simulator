from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from simulator.defaults import (
    DEFAULT_DAYS,
    DEFAULT_END_RETENTION,
    DEFAULT_RETENTION_STEP,
    DEFAULT_START_RETENTION,
)


@dataclass(frozen=True, slots=True)
class RunSweepOverrides:
    """Subset of run_sweep.py flags we care about in parent fanout scripts.

    This is intentionally small: it only includes flags that affect how many
    child runs we expect to execute and how long each run takes (days), plus
    SSP-MMC policy selection (affects number of runs).
    """

    start_retention: float
    end_retention: float
    step: float
    days: int
    sspmmc_policy: Path | None
    sspmmc_policy_dir: Path | None
    sspmmc_policies: str | None
    sspmmc_policy_glob: str
    sspmmc_max: int | None


def parse_run_sweep_overrides(extra_args: Sequence[str]) -> RunSweepOverrides:
    """Parse known run_sweep.py flags from a forwarded argv tail.

    Unknown args are ignored on purpose: callers often forward everything after
    `--` to run_sweep.py but only need a few values for progress estimation.
    """

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--start-retention", type=float, default=DEFAULT_START_RETENTION
    )
    parser.add_argument("--end-retention", type=float, default=DEFAULT_END_RETENTION)
    parser.add_argument("--step", type=float, default=DEFAULT_RETENTION_STEP)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--sspmmc-policy", type=Path, default=None)
    parser.add_argument("--sspmmc-policy-dir", type=Path, default=None)
    parser.add_argument("--sspmmc-policies", default=None)
    parser.add_argument("--sspmmc-policy-glob", default="*.json")
    parser.add_argument("--sspmmc-max", type=int, default=None)

    ns, _unknown = parser.parse_known_args(list(extra_args))
    return RunSweepOverrides(
        start_retention=ns.start_retention,
        end_retention=ns.end_retention,
        step=ns.step,
        days=ns.days,
        sspmmc_policy=ns.sspmmc_policy,
        sspmmc_policy_dir=ns.sspmmc_policy_dir,
        sspmmc_policies=ns.sspmmc_policies,
        sspmmc_policy_glob=ns.sspmmc_policy_glob,
        sspmmc_max=ns.sspmmc_max,
    )
