from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import StageName
from simulator.experiment_infra.schemas import ExperimentConfig


DEFAULT_SELECTOR_MAX_LANES_PER_BATCH = 8192


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    name: str
    command: tuple[str, ...] | None = None
    skipped: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command) if self.command is not None else None,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manifest-driven portfolio workflow from one command: "
            "baseline DR selection, FSRS6 baseline sweep, then formal stages."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a portfolio experiment TOML config.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Formal experiment run id. Defaults to the config name.",
    )
    parser.add_argument(
        "--baseline-run-id",
        default=None,
        help=(
            "Run id to stamp on generated FSRS6 baseline logs. Defaults to a "
            "stable id derived from the baseline DR manifest filename."
        ),
    )
    parser.add_argument(
        "--selector-max-lanes-per-batch",
        type=int,
        default=DEFAULT_SELECTOR_MAX_LANES_PER_BATCH,
        help=(
            "Lane cap for select_fsrs6_baseline_drs.py. Defaults to "
            f"{DEFAULT_SELECTOR_MAX_LANES_PER_BATCH}."
        ),
    )
    parser.add_argument(
        "--baseline-max-lanes-per-batch",
        type=int,
        default=None,
        help=(
            "Optional global lane cap for the baseline sweep. When omitted, "
            "the experiment config's sweep env_overrides remain in effect."
        ),
    )
    parser.add_argument(
        "--formal-stage",
        choices=[stage.value for stage in StageName] + ["all"],
        default="all",
        help="Formal run_experiment.py stage to execute after baseline preparation.",
    )
    parser.add_argument(
        "--force-manifest",
        action="store_true",
        help="Regenerate the baseline DR manifest even if it already exists.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not generate the baseline DR manifest.",
    )
    parser.add_argument(
        "--skip-baseline-sweep",
        action="store_true",
        help="Do not run the manifest-driven FSRS6 baseline sweep.",
    )
    parser.add_argument(
        "--skip-formal-stages",
        action="store_true",
        help="Stop after baseline preparation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selector_max_lanes_per_batch < 1:
        raise SystemExit("--selector-max-lanes-per-batch must be >= 1.")
    if (
        args.baseline_max_lanes_per_batch is not None
        and args.baseline_max_lanes_per_batch < 1
    ):
        raise SystemExit("--baseline-max-lanes-per-batch must be >= 1.")
    if args.force_manifest and args.skip_manifest:
        raise SystemExit("--force-manifest cannot be combined with --skip-manifest.")

    config_path = args.config
    config = ExperimentConfig.from_toml(config_path)
    manifest = config.baseline_dr_selection.manifest
    if manifest is None:
        raise SystemExit("[baseline_dr_selection].manifest is required.")

    manifest_path = _resolve_repo_path(manifest)
    formal_run_id = args.run_id or config.name
    baseline_run_id = args.baseline_run_id or _default_baseline_run_id(manifest)

    steps = build_workflow_steps(
        config_path=config_path,
        config=config,
        manifest_path=manifest_path,
        formal_run_id=formal_run_id,
        baseline_run_id=baseline_run_id,
        selector_max_lanes_per_batch=args.selector_max_lanes_per_batch,
        baseline_max_lanes_per_batch=args.baseline_max_lanes_per_batch,
        formal_stage=args.formal_stage,
        force_manifest=args.force_manifest,
        skip_manifest=args.skip_manifest,
        skip_baseline_sweep=args.skip_baseline_sweep,
        skip_formal_stages=args.skip_formal_stages,
    )

    results: list[dict[str, Any]] = []
    for step in steps:
        if step.skipped:
            results.append({**step.to_dict(), "exit_code": None})
            continue
        if step.command is None:
            raise RuntimeError(f"Workflow step {step.name} has no command.")
        print(f"\n[{step.name}] {shlex.join(step.command)}", flush=True)
        if args.dry_run:
            results.append({**step.to_dict(), "exit_code": None})
            continue
        completed = subprocess.run(step.command, cwd=REPO_ROOT, check=False)
        results.append({**step.to_dict(), "exit_code": completed.returncode})
        if completed.returncode != 0:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "failed_step": step.name,
                        "exit_code": completed.returncode,
                        "steps": results,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return completed.returncode

    print(
        json.dumps(
            {
                "passed": True,
                "config": str(config_path),
                "run_id": formal_run_id,
                "baseline_run_id": baseline_run_id,
                "manifest": str(manifest_path),
                "steps": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_workflow_steps(
    *,
    config_path: Path,
    config: ExperimentConfig,
    manifest_path: Path,
    formal_run_id: str,
    baseline_run_id: str,
    selector_max_lanes_per_batch: int,
    baseline_max_lanes_per_batch: int | None,
    formal_stage: str,
    force_manifest: bool,
    skip_manifest: bool,
    skip_baseline_sweep: bool,
    skip_formal_stages: bool,
) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    if skip_manifest:
        steps.append(
            WorkflowStep(
                name="select-baseline-drs",
                skipped=True,
                reason="disabled by --skip-manifest",
            )
        )
    elif manifest_path.exists() and not force_manifest:
        steps.append(
            WorkflowStep(
                name="select-baseline-drs",
                skipped=True,
                reason=f"manifest already exists: {manifest_path}",
            )
        )
    else:
        steps.append(
            WorkflowStep(
                name="select-baseline-drs",
                command=tuple(
                    [
                        "uv",
                        "run",
                        "python",
                        "experiments/rl_scheduler/select_fsrs6_baseline_drs.py",
                        "--config",
                        str(config_path),
                        "--max-lanes-per-batch",
                        str(selector_max_lanes_per_batch),
                    ]
                ),
            )
        )

    if skip_baseline_sweep:
        steps.append(
            WorkflowStep(
                name="sweep-fsrs6-baseline",
                skipped=True,
                reason="disabled by --skip-baseline-sweep",
            )
        )
    else:
        steps.append(
            WorkflowStep(
                name="sweep-fsrs6-baseline",
                command=tuple(
                    _baseline_sweep_command(
                        config_path=config_path,
                        config=config,
                        manifest_path=manifest_path,
                        baseline_run_id=baseline_run_id,
                        baseline_max_lanes_per_batch=baseline_max_lanes_per_batch,
                    )
                ),
            )
        )

    if skip_formal_stages:
        steps.append(
            WorkflowStep(
                name="formal-stages",
                skipped=True,
                reason="disabled by --skip-formal-stages",
            )
        )
    else:
        steps.append(
            WorkflowStep(
                name="formal-stages",
                command=(
                    "uv",
                    "run",
                    "python",
                    "experiments/rl_scheduler/run_experiment.py",
                    "--config",
                    str(config_path),
                    "--stage",
                    formal_stage,
                    "--run-id",
                    formal_run_id,
                ),
            )
        )

    return steps


def _baseline_sweep_command(
    *,
    config_path: Path,
    config: ExperimentConfig,
    manifest_path: Path,
    baseline_run_id: str,
    baseline_max_lanes_per_batch: int | None,
) -> list[str]:
    environments = config.baseline.environments or config.sweep_batched.envs
    if not environments:
        raise ValueError("baseline.environments or sweep.envs must be configured.")
    command = [
        "uv",
        "run",
        "python",
        "experiments/retention_sweep/run_sweep_users_batched.py",
        "--config",
        str(config_path),
        "--run-id",
        baseline_run_id,
        "--env",
        ",".join(environments),
        "--sched",
        config.baseline.scheduler,
        "--fsrs6-dr-manifest",
        str(manifest_path),
        "--log-dir",
        str(config.baseline.log_root),
        "--log-layout",
        config.sweep_batched.log_layout,
        "--no-progress",
    ]
    if baseline_max_lanes_per_batch is not None:
        command.extend(
            [
                "--max-lanes-per-batch",
                str(baseline_max_lanes_per_batch),
            ]
        )
    return command


def _default_baseline_run_id(manifest: Path) -> str:
    stem = manifest.stem
    if stem.startswith("fsrs6_"):
        stem = stem[len("fsrs6_") :]
    return f"fsrs6_baseline_{stem}"


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
