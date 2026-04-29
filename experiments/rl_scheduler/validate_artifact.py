from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import validate_scheduler_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate scheduler artifact metadata.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to scheduler artifact metadata JSON.",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Also require referenced artifact files to exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = validate_scheduler_artifact(
        args.metadata,
        require_files=args.require_files,
    )
    print(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
