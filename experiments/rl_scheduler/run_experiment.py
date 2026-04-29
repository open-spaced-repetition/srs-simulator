from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import StageName
from simulator.experiment_infra.runner import run_all, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TOML-driven RL scheduler experiment infrastructure stages.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a checked-in experiment TOML config.",
    )
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in StageName] + ["all"],
        default=StageName.DRY_RUN.value,
        help="Experiment stage to execute.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable run id for reproducible stage output paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = ["uv", "run", "python", *sys.argv]
    if args.stage == "all":
        result = run_all(
            config_path=args.config,
            repo_root=REPO_ROOT,
            run_id=args.run_id,
        )
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return result.exit_code

    stage = StageName(args.stage)
    result = run_stage(
        config_path=args.config,
        stage=stage,
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        command=command,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
