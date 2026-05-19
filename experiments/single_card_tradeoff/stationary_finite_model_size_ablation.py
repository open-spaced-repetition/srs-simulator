from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.single_card_tradeoff.oracle_stationary_finite_distill import (
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_SUPERVISION,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
)
from experiments.single_card_tradeoff.tradeoff import (
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
    FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float


DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation"
)
DEFAULT_CANDIDATES = [
    "r16d2",
    "r12d2",
    "r10d2",
    "r8d2",
    "r8d1",
    "r6d1",
]
DEFAULT_SUB216_CANDIDATES = [
    "r5d1",
    "r4d1",
    "r3d1",
    "mlp8",
    "mlp6",
    "mlp4",
    "linear",
    "quadratic",
]
DEFAULT_CANDIDATE_SETS = {
    "default": DEFAULT_CANDIDATES,
    "sub216": DEFAULT_SUB216_CANDIDATES,
    "all": [*DEFAULT_CANDIDATES, *DEFAULT_SUB216_CANDIDATES],
}


@dataclass(frozen=True)
class Candidate:
    label: str
    network: str
    hidden_size: int
    network_depth: int
    family: str

    @property
    def arch_label(self) -> str:
        if self.network == "residual":
            return f"residual:{self.hidden_size}:{self.network_depth}"
        if self.network == "mlp":
            return f"mlp:{self.hidden_size}"
        return self.network

    def variant(self, epochs: int) -> str:
        return f"sf_train5_{self.label}_e{epochs}"


CANDIDATES = {
    "r16d2": Candidate("r16d2", "residual", 16, 2, "residual"),
    "r12d2": Candidate("r12d2", "residual", 12, 2, "residual"),
    "r10d2": Candidate("r10d2", "residual", 10, 2, "residual"),
    "r8d2": Candidate("r8d2", "residual", 8, 2, "residual"),
    "r8d1": Candidate("r8d1", "residual", 8, 1, "residual"),
    "r6d1": Candidate("r6d1", "residual", 6, 1, "residual"),
    "r5d1": Candidate("r5d1", "residual", 5, 1, "residual"),
    "r4d1": Candidate("r4d1", "residual", 4, 1, "residual"),
    "r3d1": Candidate("r3d1", "residual", 3, 1, "residual"),
    "mlp8": Candidate("mlp8", "mlp", 8, 1, "mlp"),
    "mlp6": Candidate("mlp6", "mlp", 6, 1, "mlp"),
    "mlp4": Candidate("mlp4", "mlp", 4, 1, "mlp"),
    "linear": Candidate("linear", "linear", 1, 1, "structured"),
    "quadratic": Candidate("quadratic", "quadratic", 1, 1, "structured"),
}


def _csv_floats(values: Sequence[float]) -> str:
    return ",".join(format_float(value) for value in values)


def _parse_int_csv(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"{name} contains invalid integer '{item}'.") from exc
    if not values:
        raise SystemExit(f"{name} must include at least one value.")
    return values


def _parse_candidate_csv(raw: str) -> list[Candidate]:
    selected: list[Candidate] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        try:
            selected.append(CANDIDATES[key])
        except KeyError as exc:
            valid = ",".join(CANDIDATES)
            raise SystemExit(
                f"Unknown candidate '{key}'. Valid values: {valid}"
            ) from exc
    if not selected:
        raise SystemExit("--candidates must include at least one value.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the stationary finite distill model-size ablation while holding "
            "all non-network variables to the current default distillation recipe."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--env", default="fsrs6_default")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-seeds", default="42,43,44")
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--candidate-set",
        choices=sorted(DEFAULT_CANDIDATE_SETS),
        default="default",
        help=(
            "Named candidate set to run when --candidates is omitted. "
            "default preserves the aligned residual table; sub216 adds the "
            "smaller residual, MLP, and structured sweep."
        ),
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help=f"Comma-separated candidate keys. Valid values: {','.join(CANDIDATES)}",
    )
    parser.add_argument(
        "--summary-prefix",
        default="model_size_ablation",
        help=(
            "Prefix for the aggregate CSVs. The default writes "
            "model_size_ablation_summary.csv and model_size_ablation_seed_metrics.csv."
        ),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument(
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    )
    parser.add_argument("--train-eval-particles", type=int, default=10_000)
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing candidate checkpoints and rerun only tradeoff evals.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train checkpoints and rebuild summaries from existing eval CSVs.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _train_candidate(args: argparse.Namespace, candidate: Candidate) -> Path:
    policy_path = args.out_dir / f"{candidate.label}_e{args.epochs}_policy.pt"
    train_path = args.out_dir / f"{candidate.label}_e{args.epochs}_train_results.csv"
    cmd = [
        sys.executable,
        str(
            REPO_ROOT
            / "experiments/single_card_tradeoff/oracle_stationary_finite_distill.py"
        ),
        "--env",
        args.env,
        "--days",
        str(args.days),
        "--deck-scale",
        str(args.deck_scale),
        "--seed",
        str(args.train_seed),
        "--cost-weights",
        _csv_floats(DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS),
        "--action-retentions",
        _csv_floats(DEFAULT_TARGET_RETENTIONS),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--supervision",
        DEFAULT_DISTILL_SUPERVISION,
        "--table-samples-per-weight",
        str(args.table_samples_per_weight),
        "--obs-mode",
        "oracle_stationary",
        "--network",
        candidate.network,
        "--hidden-size",
        str(candidate.hidden_size),
        "--network-depth",
        str(candidate.network_depth),
        "--eval-particles",
        str(args.train_eval_particles),
        "--out",
        str(train_path),
        "--model-out",
        str(policy_path),
    ]
    if args.torch_device:
        cmd.extend(["--torch-device", args.torch_device])
    if args.no_progress:
        cmd.append("--no-progress")
    _run(cmd, cwd=REPO_ROOT)
    return policy_path


def _eval_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    *,
    policy_path: Path,
    seed: int,
) -> tuple[Path, Path]:
    suffix = "" if seed == args.train_seed else f"_seed{seed}"
    results_path = (
        args.out_dir / f"{candidate.label}_e{args.epochs}{suffix}_tradeoff_results.csv"
    )
    auc_path = (
        args.out_dir
        / f"{candidate.label}_e{args.epochs}{suffix}_tradeoff_regret_auc.csv"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments/single_card_tradeoff/tradeoff.py"),
        "--env",
        args.env,
        "--sched",
        f"fsrs6_default,{FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER}",
        "--days",
        str(args.days),
        "--particles",
        str(args.eval_particles),
        "--deck-scale",
        str(args.deck_scale),
        "--seed",
        str(seed),
        "--target-retentions",
        _csv_floats(DEFAULT_TARGET_RETENTIONS),
        "--oracle-stationary-finite-distill-policy",
        str(policy_path),
        "--oracle-stationary-finite-distill-cost-weights",
        _csv_floats(DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS),
        "--out",
        str(results_path),
        "--regret-auc-out",
        str(auc_path),
        "--no-plot",
    ]
    if args.torch_device:
        cmd.extend(["--torch-device", args.torch_device])
    if args.no_progress:
        cmd.append("--no-progress")
    _run(cmd, cwd=REPO_ROOT)
    return results_path, auc_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_auc_row(path: Path) -> dict[str, str]:
    for row in _read_csv_rows(path):
        if (
            row["baseline_scheduler"] == "fsrs6_default"
            and row["scheduler"] == FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER
        ):
            return row
    raise RuntimeError(f"Missing fsrs6_default comparison row in {path}")


def _parameter_count(policy_path: Path) -> int:
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    return int(sum(value.numel() for value in checkpoint["model_state_dict"].values()))


def _policy_metadata(policy_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    return {
        "parameter_count": _parameter_count(policy_path),
        "epochs": int(checkpoint["distill_epochs"]),
        "steps_per_epoch": int(checkpoint["distill_steps_per_epoch"]),
        "final_ce_loss": float(checkpoint["final_ce_loss"]),
        "train_teacher_action_agreement": float(
            checkpoint["final_teacher_action_agreement"]
        ),
        "eval_teacher_action_agreement": float(
            checkpoint["eval_teacher_action_agreement"]
        ),
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _metric_value(row: dict[str, str], key: str) -> float:
    if key in row and row[key] != "":
        return float(row[key])
    legacy_map = {
        "same_target_time_saved_auc": ("time_regret_auc", -1.0),
        "relative_same_target_time_saved_auc_percent": (
            "relative_regret_auc_percent",
            -1.0,
        ),
    }
    legacy_key, sign = legacy_map[key]
    return sign * float(row[legacy_key])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for variant_rows in by_variant.values():
        first = variant_rows[0]
        time_values = [float(row["same_target_time_saved_auc"]) for row in variant_rows]
        relative_values = [
            float(row["relative_same_target_time_saved_auc_percent"])
            for row in variant_rows
        ]
        coverage_values = [float(row["span_coverage_percent"]) for row in variant_rows]
        covered_counts = ",".join(
            str(int(row["covered_target_count"])) for row in variant_rows
        )
        summary_rows.append(
            {
                "variant": first["variant"],
                "arch_label": first["arch_label"],
                "family": first["family"],
                "parameter_count": first["parameter_count"],
                "compression_vs_1452_percent": (
                    (1.0 - float(first["parameter_count"]) / 1452.0) * 100.0
                ),
                "train_weight_count": len(
                    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
                ),
                "train_cost_weights": _csv_floats(
                    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
                ),
                "epochs": first["epochs"],
                "steps_per_epoch": first["steps_per_epoch"],
                "final_ce_loss": first["final_ce_loss"],
                "train_teacher_action_agreement": first[
                    "train_teacher_action_agreement"
                ],
                "eval_teacher_action_agreement": first["eval_teacher_action_agreement"],
                "seed_count": len(variant_rows),
                "same_target_time_saved_auc_mean": _mean(time_values),
                "same_target_time_saved_auc_std": _stdev(time_values),
                "relative_same_target_time_saved_auc_percent_mean": _mean(
                    relative_values
                ),
                "relative_same_target_time_saved_auc_percent_std": _stdev(
                    relative_values
                ),
                "span_coverage_percent_mean": _mean(coverage_values),
                "span_coverage_percent_std": _stdev(coverage_values),
                "covered_target_count_values": covered_counts,
                "policy_path": first["policy_path"],
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    candidate_keys = (
        args.candidates
        if args.candidates is not None
        else ",".join(DEFAULT_CANDIDATE_SETS[args.candidate_set])
    )
    candidates = _parse_candidate_csv(candidate_keys)
    eval_seeds = _parse_int_csv(args.eval_seeds, name="--eval-seeds")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        policy_path = args.out_dir / f"{candidate.label}_e{args.epochs}_policy.pt"
        if not args.skip_train:
            policy_path = _train_candidate(args, candidate)
        if not policy_path.exists():
            raise SystemExit(f"Missing policy checkpoint: {policy_path}")
        metadata = _policy_metadata(policy_path)

        for seed in eval_seeds:
            suffix = "" if seed == args.train_seed else f"_seed{seed}"
            auc_path = (
                args.out_dir
                / f"{candidate.label}_e{args.epochs}{suffix}_tradeoff_regret_auc.csv"
            )
            if not args.skip_eval:
                _, auc_path = _eval_candidate(
                    args,
                    candidate,
                    policy_path=policy_path,
                    seed=seed,
                )
            row = _read_auc_row(auc_path)
            seed_rows.append(
                {
                    "variant": candidate.variant(args.epochs),
                    "arch_label": candidate.arch_label,
                    "family": candidate.family,
                    "parameter_count": metadata["parameter_count"],
                    "compression_vs_1452_percent": (
                        (1.0 - float(metadata["parameter_count"]) / 1452.0) * 100.0
                    ),
                    "train_weight_count": len(
                        DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
                    ),
                    "train_cost_weights": _csv_floats(
                        DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
                    ),
                    "epochs": metadata["epochs"],
                    "steps_per_epoch": metadata["steps_per_epoch"],
                    "final_ce_loss": metadata["final_ce_loss"],
                    "train_teacher_action_agreement": metadata[
                        "train_teacher_action_agreement"
                    ],
                    "eval_teacher_action_agreement": metadata[
                        "eval_teacher_action_agreement"
                    ],
                    "seed": seed,
                    "span_coverage_percent": float(row["span_coverage_percent"]),
                    "same_target_time_saved_auc": float(
                        _metric_value(row, "same_target_time_saved_auc")
                    ),
                    "relative_same_target_time_saved_auc_percent": float(
                        _metric_value(
                            row,
                            "relative_same_target_time_saved_auc_percent",
                        )
                    ),
                    "covered_target_count": int(row["covered_target_count"]),
                    "target_count": int(row["target_count"]),
                    "auc_path": auc_path,
                    "policy_path": policy_path,
                }
            )

    seed_fieldnames = [
        "variant",
        "arch_label",
        "family",
        "parameter_count",
        "compression_vs_1452_percent",
        "train_weight_count",
        "train_cost_weights",
        "epochs",
        "steps_per_epoch",
        "final_ce_loss",
        "train_teacher_action_agreement",
        "eval_teacher_action_agreement",
        "seed",
        "span_coverage_percent",
        "same_target_time_saved_auc",
        "relative_same_target_time_saved_auc_percent",
        "covered_target_count",
        "target_count",
        "auc_path",
        "policy_path",
    ]
    summary_fieldnames = [
        "variant",
        "arch_label",
        "family",
        "parameter_count",
        "compression_vs_1452_percent",
        "train_weight_count",
        "train_cost_weights",
        "epochs",
        "steps_per_epoch",
        "final_ce_loss",
        "train_teacher_action_agreement",
        "eval_teacher_action_agreement",
        "seed_count",
        "same_target_time_saved_auc_mean",
        "same_target_time_saved_auc_std",
        "relative_same_target_time_saved_auc_percent_mean",
        "relative_same_target_time_saved_auc_percent_std",
        "span_coverage_percent_mean",
        "span_coverage_percent_std",
        "covered_target_count_values",
        "policy_path",
    ]
    seed_path = args.out_dir / f"{args.summary_prefix}_seed_metrics.csv"
    summary_path = args.out_dir / f"{args.summary_prefix}_summary.csv"
    _write_csv(seed_path, seed_rows, seed_fieldnames)
    _write_csv(summary_path, _build_summary_rows(seed_rows), summary_fieldnames)
    print(f"Wrote {seed_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
