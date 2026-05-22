from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.cli.fsrs6_adr_train_multiuser import (  # noqa: E402
    _evaluate_single_policy,
    build_jobs,
    load_user_configs,
    parse_float_list,
    parse_user_ids,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    MIN_TARGET_RETENTION,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from simulator.fsrs6_adr_policy import (  # noqa: E402
    FEATURE_VERSION_LOG_POLY,
    FSRS6ADRPolicy,
)
from simulator.scheduler_spec import format_float  # noqa: E402


DEFAULT_COST_WEIGHTS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
    192.0,
    256.0,
    320.0,
    384.0,
    512.0,
    1024.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Monte Carlo variance for trained single-card FSRS6 ADR "
            "policies by repeating fixed-policy rollout with different seeds."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--env", default="fsrs6")
    parser.add_argument("--user-ids", default="1,2,3,4,5,6,7,8")
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
    )
    parser.add_argument("--train-run-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/fsrs6_adr_mc_variance"),
    )
    parser.add_argument("--button-usage", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default="0")
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--reference-particles", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--job-batch-size", type=int, default=8192)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--retention-min", type=float, default=None)
    parser.add_argument("--retention-max", type=float, default=None)
    parser.add_argument("--review-markov-transition", action="store_true")
    parser.add_argument(
        "--exact-memory",
        action="store_true",
        help=(
            "Use exact memorized-day accumulation. Omit this to match default "
            "ADR training rollouts."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    add_run_monitoring_args(parser)
    return parser.parse_args()


def _load_metadata(train_run_root: Path) -> dict[str, Any]:
    path = train_run_root / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"Invalid ADR train metadata: {path}")
    return raw


def _load_training_summary(
    train_run_root: Path,
) -> dict[tuple[int, float], dict[str, str]]:
    path = train_run_root / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[tuple[int, float], dict[str, str]] = {}
    for row in rows:
        out[(int(row["user_id"]), float(row["lambda_value"]))] = row
    return out


def _sample_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _sample_mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _percentile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return finite[low]
    weight = position - low
    return finite[low] * (1.0 - weight) + finite[high] * weight


def _optional_float(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
) -> list[dict[str, Any]]:
    groups = sorted({row[group_key] for row in rows})
    out: list[dict[str, Any]] = []
    for group in groups:
        selected = [row for row in rows if row[group_key] == group]
        objective_std = [float(row["objective_std"]) for row in selected]
        abs_reference_error = [
            float(row["abs_objective_mean_minus_reference"])
            for row in selected
            if row["abs_objective_mean_minus_reference"] is not None
        ]
        out.append(
            {
                group_key: group,
                "policy_count": len(selected),
                "mean_objective_std": _sample_mean(objective_std),
                "median_objective_std": _percentile(objective_std, 0.5),
                "p90_objective_std": _percentile(objective_std, 0.9),
                "max_objective_std": max(objective_std),
                "mean_abs_objective_mean_minus_reference": (
                    _sample_mean(abs_reference_error) if abs_reference_error else None
                ),
                "p90_abs_objective_mean_minus_reference": _percentile(
                    abs_reference_error, 0.9
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    if args.particles <= 0:
        raise SystemExit("--particles must be > 0.")
    if args.repeats <= 1:
        raise SystemExit("--repeats must be > 1.")
    if args.reference_particles < 0:
        raise SystemExit("--reference-particles must be >= 0.")
    if args.job_batch_size < 0:
        raise SystemExit("--job-batch-size must be >= 0.")

    train_run_root = args.train_run_root.expanduser()
    if not train_run_root.exists():
        raise SystemExit(f"ADR train run root not found: {train_run_root}")
    metadata = _load_metadata(train_run_root)
    feature_version = args.feature_version or str(
        metadata.get("feature_version") or FEATURE_VERSION_LOG_POLY
    )
    retention_min = float(
        args.retention_min
        if args.retention_min is not None
        else metadata.get("retention_min", MIN_TARGET_RETENTION)
    )
    retention_max = float(
        args.retention_max
        if args.retention_max is not None
        else metadata.get("retention_max", 0.98)
    )

    user_ids = parse_user_ids(args.user_ids)
    cost_weights = parse_float_list(args.cost_weights, name="--cost-weights")
    configs = load_user_configs(args, user_ids)
    jobs = build_jobs(
        user_ids=user_ids,
        cost_weights=cost_weights,
        configs=configs,
        out_dir=train_run_root,
    )
    policies = []
    for job in jobs:
        path = job.output_dir / "policy.json"
        if not path.exists():
            raise SystemExit(f"ADR policy not found: {path}")
        policies.append(FSRS6ADRPolicy.from_json(path))

    device = (
        torch.device(args.torch_device)
        if args.torch_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )

    coefficients = torch.tensor(
        [policy.coefficients for policy in policies],
        device=device,
        dtype=torch.float64,
    )
    training_summary = _load_training_summary(train_run_root)

    values_by_job: list[dict[str, list[float]]] = [
        {
            "objective": [],
            "card_expected_retrievability": [],
            "card_minutes_per_day": [],
            "card_reviews_per_day": [],
        }
        for _job in jobs
    ]

    sample_path = args.out_dir / "samples.csv"
    sample_fields = [
        "repeat_index",
        "seed",
        "user_id",
        "lambda_value",
        "objective",
        "card_expected_retrievability",
        "card_minutes_per_day",
        "card_reviews_per_day",
        "card_total_reviews",
        "card_total_lapses",
        "observed_retention",
    ]
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        for repeat_index in range(args.repeats):
            seed = int(args.seed + repeat_index * args.seed_stride)
            metrics, objectives = _evaluate_single_policy(
                jobs=jobs,
                configs=configs,
                coefficients=coefficients,
                days=args.days,
                particles_per_group=args.particles,
                feature_version=feature_version,
                retention_min=retention_min,
                retention_max=retention_max,
                exact_memory=args.exact_memory,
                job_batch_size=args.job_batch_size,
                seed=seed,
                review_markov_transition=args.review_markov_transition,
                scheduler_priority="low_retrievability",
                device=device,
                dtype=torch.float64,
            )
            for job_index, (job, metric, objective) in enumerate(
                zip(jobs, metrics, objectives, strict=True)
            ):
                values_by_job[job_index]["objective"].append(objective)
                values_by_job[job_index]["card_expected_retrievability"].append(
                    metric.card_expected_retrievability
                )
                values_by_job[job_index]["card_minutes_per_day"].append(
                    metric.card_minutes_per_day
                )
                values_by_job[job_index]["card_reviews_per_day"].append(
                    metric.card_reviews_per_day
                )
                writer.writerow(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "user_id": job.user_id,
                        "lambda_value": format_float(job.lambda_value),
                        "objective": objective,
                        "card_expected_retrievability": (
                            metric.card_expected_retrievability
                        ),
                        "card_minutes_per_day": metric.card_minutes_per_day,
                        "card_reviews_per_day": metric.card_reviews_per_day,
                        "card_total_reviews": metric.card_total_reviews,
                        "card_total_lapses": metric.card_total_lapses,
                        "observed_retention": metric.observed_retention,
                    }
                )
            if not args.no_progress:
                print(
                    f"repeat={repeat_index + 1}/{args.repeats} seed={seed}",
                    flush=True,
                )

    reference_by_job: list[dict[str, float | None]] = [
        {
            "objective": None,
            "card_expected_retrievability": None,
            "card_minutes_per_day": None,
            "card_reviews_per_day": None,
        }
        for _job in jobs
    ]
    if args.reference_particles > 0:
        reference_metrics, reference_objectives = _evaluate_single_policy(
            jobs=jobs,
            configs=configs,
            coefficients=coefficients,
            days=args.days,
            particles_per_group=args.reference_particles,
            feature_version=feature_version,
            retention_min=retention_min,
            retention_max=retention_max,
            exact_memory=args.exact_memory,
            job_batch_size=args.job_batch_size,
            seed=int(args.seed + 900_000),
            review_markov_transition=args.review_markov_transition,
            scheduler_priority="low_retrievability",
            device=device,
            dtype=torch.float64,
        )
        reference_rows = []
        for job_index, (job, metric, objective) in enumerate(
            zip(jobs, reference_metrics, reference_objectives, strict=True)
        ):
            reference_by_job[job_index] = {
                "objective": objective,
                "card_expected_retrievability": metric.card_expected_retrievability,
                "card_minutes_per_day": metric.card_minutes_per_day,
                "card_reviews_per_day": metric.card_reviews_per_day,
            }
            reference_rows.append(
                {
                    "user_id": job.user_id,
                    "lambda_value": format_float(job.lambda_value),
                    "particles": args.reference_particles,
                    "seed": int(args.seed + 900_000),
                    "objective": objective,
                    "card_expected_retrievability": metric.card_expected_retrievability,
                    "card_minutes_per_day": metric.card_minutes_per_day,
                    "card_reviews_per_day": metric.card_reviews_per_day,
                    "card_total_reviews": metric.card_total_reviews,
                    "card_total_lapses": metric.card_total_lapses,
                    "observed_retention": metric.observed_retention,
                }
            )
        _write_csv(
            args.out_dir / "reference.csv",
            reference_rows,
            fieldnames=[
                "user_id",
                "lambda_value",
                "particles",
                "seed",
                "objective",
                "card_expected_retrievability",
                "card_minutes_per_day",
                "card_reviews_per_day",
                "card_total_reviews",
                "card_total_lapses",
                "observed_retention",
            ],
        )

    policy_rows: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs):
        values = values_by_job[job_index]
        objective_mean = _sample_mean(values["objective"])
        objective_var = _sample_variance(values["objective"])
        objective_std = math.sqrt(objective_var)
        reference_objective = reference_by_job[job_index]["objective"]
        train_row = training_summary.get((job.user_id, job.lambda_value))
        objective_improvement = _optional_float(train_row, "objective_improvement")
        abs_reference_error = (
            abs(objective_mean - float(reference_objective))
            if reference_objective is not None
            else None
        )
        policy_rows.append(
            {
                "user_id": job.user_id,
                "lambda_value": job.lambda_value,
                "particles": args.particles,
                "repeats": args.repeats,
                "objective_mean": objective_mean,
                "objective_variance": objective_var,
                "objective_std": objective_std,
                "objective_se_mean": objective_std / math.sqrt(args.repeats),
                "reference_particles": (
                    args.reference_particles
                    if reference_objective is not None
                    else None
                ),
                "reference_objective": reference_objective,
                "objective_mean_minus_reference": (
                    objective_mean - float(reference_objective)
                    if reference_objective is not None
                    else None
                ),
                "abs_objective_mean_minus_reference": abs_reference_error,
                "card_expected_retrievability_mean": _sample_mean(
                    values["card_expected_retrievability"]
                ),
                "card_expected_retrievability_std": math.sqrt(
                    _sample_variance(values["card_expected_retrievability"])
                ),
                "card_minutes_per_day_mean": _sample_mean(
                    values["card_minutes_per_day"]
                ),
                "card_minutes_per_day_std": math.sqrt(
                    _sample_variance(values["card_minutes_per_day"])
                ),
                "card_reviews_per_day_mean": _sample_mean(
                    values["card_reviews_per_day"]
                ),
                "card_reviews_per_day_std": math.sqrt(
                    _sample_variance(values["card_reviews_per_day"])
                ),
                "train_summary_objective_improvement": objective_improvement,
                "objective_std_div_abs_train_improvement": (
                    objective_std / abs(objective_improvement)
                    if objective_improvement is not None
                    and abs(objective_improvement) > 0.0
                    else None
                ),
                "train_summary_passed": (
                    train_row.get("passed") if train_row is not None else None
                ),
            }
        )

    policy_fields = [
        "user_id",
        "lambda_value",
        "particles",
        "repeats",
        "objective_mean",
        "objective_variance",
        "objective_std",
        "objective_se_mean",
        "reference_particles",
        "reference_objective",
        "objective_mean_minus_reference",
        "abs_objective_mean_minus_reference",
        "card_expected_retrievability_mean",
        "card_expected_retrievability_std",
        "card_minutes_per_day_mean",
        "card_minutes_per_day_std",
        "card_reviews_per_day_mean",
        "card_reviews_per_day_std",
        "train_summary_objective_improvement",
        "objective_std_div_abs_train_improvement",
        "train_summary_passed",
    ]
    _write_csv(
        args.out_dir / "policy_variance_summary.csv",
        policy_rows,
        fieldnames=policy_fields,
    )

    by_user = _group_summary(policy_rows, group_key="user_id")
    by_lambda = _group_summary(policy_rows, group_key="lambda_value")
    _write_csv(
        args.out_dir / "variance_by_user.csv",
        by_user,
        fieldnames=list(by_user[0].keys()) if by_user else ["user_id"],
    )
    _write_csv(
        args.out_dir / "variance_by_lambda.csv",
        by_lambda,
        fieldnames=list(by_lambda[0].keys()) if by_lambda else ["lambda_value"],
    )

    objective_std_values = [float(row["objective_std"]) for row in policy_rows]
    objective_se_values = [float(row["objective_se_mean"]) for row in policy_rows]
    reference_errors = [
        float(row["abs_objective_mean_minus_reference"])
        for row in policy_rows
        if row["abs_objective_mean_minus_reference"] is not None
    ]
    ratios = [
        float(row["objective_std_div_abs_train_improvement"])
        for row in policy_rows
        if row["objective_std_div_abs_train_improvement"] is not None
        and math.isfinite(float(row["objective_std_div_abs_train_improvement"]))
    ]
    aggregate = {
        "schema_version": 1,
        "experiment": "fsrs6_adr_mc_variance",
        "train_run_root": str(train_run_root),
        "env": args.env,
        "user_ids": user_ids,
        "cost_weights": [float(value) for value in cost_weights],
        "policy_count": len(policy_rows),
        "particles": args.particles,
        "repeats": args.repeats,
        "reference_particles": args.reference_particles,
        "exact_memory": bool(args.exact_memory),
        "review_markov_transition": bool(args.review_markov_transition),
        "feature_version": feature_version,
        "retention_min": retention_min,
        "retention_max": retention_max,
        "mean_objective_std": _sample_mean(objective_std_values),
        "median_objective_std": _percentile(objective_std_values, 0.5),
        "p90_objective_std": _percentile(objective_std_values, 0.9),
        "p95_objective_std": _percentile(objective_std_values, 0.95),
        "max_objective_std": max(objective_std_values),
        "mean_objective_se_mean": _sample_mean(objective_se_values),
        "p90_objective_se_mean": _percentile(objective_se_values, 0.9),
        "mean_abs_objective_mean_minus_reference": (
            _sample_mean(reference_errors) if reference_errors else None
        ),
        "p90_abs_objective_mean_minus_reference": _percentile(reference_errors, 0.9),
        "max_abs_objective_mean_minus_reference": (
            max(reference_errors) if reference_errors else None
        ),
        "median_objective_std_div_abs_train_improvement": _percentile(ratios, 0.5),
        "p90_objective_std_div_abs_train_improvement": _percentile(ratios, 0.9),
        "policies_where_objective_std_exceeds_abs_train_improvement": sum(
            1 for value in ratios if value > 1.0
        ),
        "policies_with_training_summary": len(ratios),
        "outputs": {
            "samples": str(sample_path),
            "policy_variance_summary": str(
                args.out_dir / "policy_variance_summary.csv"
            ),
            "variance_by_user": str(args.out_dir / "variance_by_user.csv"),
            "variance_by_lambda": str(args.out_dir / "variance_by_lambda.csv"),
        },
    }
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
