from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra.status import collect_run_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect machine-readable RL scheduler experiment run evidence.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root, usually <output_root>/<run_id>.",
    )
    parser.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="Exit non-zero when the collected status is not fully passed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = collect_run_status(args.run_root)
    print(json.dumps(status, indent=2, sort_keys=True))
    if args.fail_on_failed and not status["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
