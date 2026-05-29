from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.fsrs6_cost_conditioned_adr_policy import (
    ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
    FSRS6CostConditionedADRPolicy,
)


BASE_CONFIG = Path(
    "experiments/rl_scheduler/configs/"
    "fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1.toml"
)
OUTPUT_ROOT = Path("artifacts/rl_scheduler/fsrs6_cost_adr_init_sensitivity_users_1_8")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_USERS = (1, 2, 3, 4, 5, 6, 7, 8)
PARAMETER_COUNT = 15
REPORT_SCRIPT = Path(
    "experiments/rl_scheduler/analyze_fsrs6_cost_adr_init_sensitivity.py"
)


@dataclass(frozen=True, slots=True)
class InitCondition:
    name: str
    label: str
    description: str
    initial_mean_source: str | None = None
    initial_policy_root: Path | None = None

    def config_line(self) -> str:
        if self.initial_mean_source is not None:
            return f'initial_mean_source = "{self.initial_mean_source}"'
        if self.initial_policy_root is not None:
            return f'initial_policy_root = "{self.initial_policy_root.as_posix()}"'
        raise ValueError(f"Init condition {self.name} has no source.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strict FSRS6 Cost-ADR initialization-sensitivity experiment."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated optimizer/simulation seeds.",
    )
    parser.add_argument(
        "--conditions",
        default="first8_mean,constant_r90,zero",
        help="Comma-separated condition names.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when an existing all_summary.json says the run passed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without executing stages.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not run the summary/report script after experiment stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _repo_path(args.output_root)
    base_config = _repo_path(args.base_config)
    seeds = _parse_ints(args.seeds, "--seeds")
    condition_names = _parse_names(args.conditions)
    zero_policy_root = output_root / "initial_policies" / "zero"
    conditions = _conditions(zero_policy_root)
    selected_conditions = [
        _condition_by_name(conditions, name) for name in condition_names
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    _write_zero_policy_root(zero_policy_root)

    manifest = {
        "base_config": _relative(base_config),
        "output_root": _relative(output_root),
        "optimizer_seeds": seeds,
        "simulation_seed": 42,
        "conditions": [
            {
                "name": condition.name,
                "label": condition.label,
                "description": condition.description,
                "initial_mean_source": condition.initial_mean_source,
                "initial_policy_root": _relative(
                    _repo_path(condition.initial_policy_root)
                )
                if condition.initial_policy_root is not None
                else None,
            }
            for condition in selected_conditions
        ],
        "runs": [],
    }

    generated_config_dir = output_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    for seed in seeds:
        for condition in selected_conditions:
            run_id = _run_id(condition=condition, seed=seed)
            run_root = output_root / run_id
            config_path = generated_config_dir / f"{run_id}.toml"
            config_path.write_text(
                _render_config(
                    base_config=base_config,
                    condition=condition,
                    seed=seed,
                    run_id=run_id,
                    output_root=args.output_root,
                    train_run_root=args.output_root / run_id,
                ),
                encoding="utf-8",
            )
            manifest["runs"].append(
                {
                    "condition": condition.name,
                    "seed": seed,
                    "run_id": run_id,
                    "run_root": _relative(run_root),
                    "config_path": _relative(config_path),
                }
            )
            if not args.force and _run_already_passed(run_root):
                print(f"[skip] {run_id} already passed", flush=True)
                continue
            commands.append(
                [
                    "uv",
                    "run",
                    "python",
                    "experiments/rl_scheduler/run_experiment.py",
                    "--config",
                    config_path.as_posix(),
                    "--stage",
                    "all",
                    "--run-id",
                    run_id,
                ]
            )

    manifest_path = output_root / "init_sensitivity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote manifest: {_relative(manifest_path)}", flush=True)

    for command in commands:
        print(f"\n$ {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode

    if not args.skip_analysis:
        analysis_command = [
            "uv",
            "run",
            "python",
            REPORT_SCRIPT.as_posix(),
            "--root",
            args.output_root.as_posix(),
        ]
        print(f"\n$ {shlex.join(analysis_command)}", flush=True)
        if not args.dry_run:
            completed = subprocess.run(analysis_command, cwd=REPO_ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


def _conditions(zero_policy_root: Path) -> tuple[InitCondition, ...]:
    return (
        InitCondition(
            name="first8_mean",
            label="first8 interval-implied R mean",
            description="Current default built-in first-8 interval-implied-retention mean.",
            initial_mean_source="first8_interval_implied_r_mean_v1",
        ),
        InitCondition(
            name="constant_r90",
            label="constant R=0.90",
            description="Hand-crafted constant desired-retention 0.90 starting point.",
            initial_mean_source="retention_baseline_cost_decay_v1",
        ),
        InitCondition(
            name="zero",
            label="all-zero coefficients",
            description="All coefficients set to zero; generated as policy JSONs.",
            initial_policy_root=zero_policy_root,
        ),
    )


def _condition_by_name(
    conditions: tuple[InitCondition, ...],
    name: str,
) -> InitCondition:
    for condition in conditions:
        if condition.name == name:
            return condition
    allowed = ", ".join(condition.name for condition in conditions)
    raise SystemExit(f"Unknown condition {name!r}; allowed: {allowed}.")


def _write_zero_policy_root(policy_root: Path) -> None:
    for user_id in DEFAULT_USERS:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=(0.0,) * PARAMETER_COUNT,
            action_head=ACTION_HEAD_RETENTION,
            feature_version=FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
            cost_weight_min=0.0,
            cost_weight_max=1024.0,
            retention_min=0.30,
            retention_max=0.995,
            title="FSRS6 Cost-ADR zero initializer",
        )
        policy.write_json(policy_root / f"user_{user_id}" / "policy.json")


def _render_config(
    *,
    base_config: Path,
    condition: InitCondition,
    seed: int,
    run_id: str,
    output_root: Path,
    train_run_root: Path,
) -> str:
    lines = base_config.read_text(encoding="utf-8").splitlines()
    rendered: list[str] = []
    section = ""
    inserted_initial_source = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]")
            if section != "training.policy_search":
                inserted_initial_source = False
            rendered.append(line)
            continue
        if section == "":
            if stripped.startswith("name ="):
                rendered.append(f'name = "{run_id}"')
                continue
            if stripped.startswith("output_root ="):
                rendered.append(f'output_root = "{output_root.as_posix()}"')
                continue
        if section == "training.policy_search":
            if stripped.startswith(
                (
                    "initial_mean_source =",
                    "initial_policy =",
                    "initial_policy_root =",
                    "initial_policy_template =",
                    "initial_policy_train_run_root =",
                )
            ):
                continue
            if not inserted_initial_source and stripped.startswith(
                "initial_policy_expand_bounds ="
            ):
                rendered.append(condition.config_line())
                inserted_initial_source = True
        if section == "training.optimizer" and stripped.startswith("seed ="):
            rendered.append(f"seed = {seed}")
            continue
        if section == "sweep":
            if stripped.startswith("run_id ="):
                rendered.append(f'run_id = "{run_id}"')
                continue
            if stripped.startswith("fsrs6_cost_adr_train_run_root ="):
                rendered.append(
                    f'fsrs6_cost_adr_train_run_root = "{train_run_root.as_posix()}"'
                )
                continue
        rendered.append(line)
    return "\n".join(rendered) + "\n"


def _run_already_passed(run_root: Path) -> bool:
    summary_path = run_root / "all" / "all_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("passed"))


def _run_id(*, condition: InitCondition, seed: int) -> str:
    return (
        "fsrs6_cost_adr_init_sensitivity_"
        f"{condition.name}_seed{seed}_users_1_8_pop16_gen20_v1"
    )


def _parse_ints(raw: str, name: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{name} must not be empty.")
    return [int(value) for value in values]


def _parse_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("--conditions must not be empty.")
    return values


def _repo_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("path must not be None")
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return REPO_ROOT / expanded


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
