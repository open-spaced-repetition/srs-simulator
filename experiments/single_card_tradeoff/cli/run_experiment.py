from __future__ import annotations

import argparse
from pathlib import Path

from experiments.single_card_tradeoff.core.workflow_config import (
    SingleCardWorkflowStage,
)
from experiments.single_card_tradeoff.core.workflow_runner import run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal stages for a single-card tradeoff workflow.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in SingleCardWorkflowStage] + ["all"],
        default=SingleCardWorkflowStage.DRY_RUN.value,
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used under the workflow stage-record root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected stage task plan without executing commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_workflow(
        config_path=args.config,
        stage=args.stage,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
