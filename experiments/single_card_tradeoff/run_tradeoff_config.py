from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import tomllib

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import format_float


@dataclass(frozen=True)
class TradeoffRunConfig:
    config_path: Path
    name: str
    seed: int
    env: str
    user_ids: tuple[int, ...]
    schedulers: tuple[str, ...]
    days: int
    particles: int
    deck_scale: int
    target_retentions: tuple[float, ...]
    oracle_cost_weights: tuple[float, ...] | None
    button_usage: Path | None
    review_markov_transition: bool
    torch_device: str | None
    scheduler_priority: str
    benchmark_partition: str
    srs_benchmark_root: Path | None
    fsrs6_adr_policy: Path | None
    fsrs6_adr_policy_root: Path | None
    fsrs6_adr_train_run_root: Path | None
    fsrs6_adr_policy_manifest: Path | None
    fsrs6_adr_lambda_values: tuple[float, ...] | None
    distill_policy_template: str | None
    distill_cost_weights: tuple[float, ...] | None
    out_root: Path
    no_plot: bool
    no_progress: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TOML-configured single-card tradeoff experiment.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--users",
        default=None,
        help="Optional comma-separated user subset overriding experiment.user_ids.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun users even when their results and regret AUC CSVs exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-user tradeoff commands without running them.",
    )
    parser.add_argument(
        "--no-multiuser-batch",
        action="store_true",
        help="Use the old one-subprocess-per-user execution mode.",
    )
    return parser.parse_args()


def _table(
    raw: Mapping[str, Any], key: str, *, required: bool = True
) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing [{key}] table.")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[{key}] must be a TOML table.")
    return value


def _str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _str(value, label)


def _int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return bool(value)


def _float(value: Any, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _int_list(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an integer array.")
    parsed = tuple(_int(item, f"{label}[]") for item in value)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return parsed


def _str_list(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a string array.")
    parsed = tuple(_str(item, f"{label}[]") for item in value)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return parsed


def _float_list(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a numeric array.")
    parsed = tuple(_float(item, f"{label}[]") for item in value)
    if not parsed:
        raise ValueError(f"{label} must not be empty.")
    return parsed


def _optional_float_list(value: Any, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _float_list(value, label)


def _path(value: Any, label: str, *, base_path: Path) -> Path:
    raw = _str(value, label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path


def _optional_path(value: Any, label: str, *, base_path: Path) -> Path | None:
    if value is None:
        return None
    return _path(value, label, base_path=base_path)


def load_config(path: Path) -> TradeoffRunConfig:
    config_path = path.expanduser()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Config must be a TOML table.")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"schema_version must be 1, got {schema_version!r}.")
    base_path = REPO_ROOT
    experiment = _table(raw, "experiment")
    fsrs6_adr = _table(raw, "fsrs6_adr", required=False)
    distill = _table(raw, "stationary_finite_distill", required=False)
    outputs = _table(raw, "outputs")
    return TradeoffRunConfig(
        config_path=config_path,
        name=_str(raw.get("name"), "name"),
        seed=_int(raw.get("seed", 42), "seed"),
        env=_str(experiment.get("env", "fsrs6"), "experiment.env"),
        user_ids=_int_list(experiment.get("user_ids"), "experiment.user_ids"),
        schedulers=_str_list(experiment.get("schedulers"), "experiment.schedulers"),
        days=_int(experiment.get("days", 1825), "experiment.days"),
        particles=_int(experiment.get("particles", 10_000), "experiment.particles"),
        deck_scale=_int(experiment.get("deck_scale", 10_000), "experiment.deck_scale"),
        target_retentions=_float_list(
            experiment.get("target_retentions"),
            "experiment.target_retentions",
        ),
        oracle_cost_weights=_optional_float_list(
            experiment.get("oracle_cost_weights"),
            "experiment.oracle_cost_weights",
        ),
        button_usage=_optional_path(
            experiment.get("button_usage"),
            "experiment.button_usage",
            base_path=base_path,
        ),
        review_markov_transition=_bool(
            experiment.get("review_markov_transition", False),
            "experiment.review_markov_transition",
        ),
        torch_device=_optional_str(
            experiment.get("torch_device"), "experiment.torch_device"
        ),
        scheduler_priority=_str(
            experiment.get("scheduler_priority", "low_retrievability"),
            "experiment.scheduler_priority",
        ),
        benchmark_partition=_str(
            experiment.get("benchmark_partition", "0"),
            "experiment.benchmark_partition",
        ),
        srs_benchmark_root=_optional_path(
            experiment.get("srs_benchmark_root"),
            "experiment.srs_benchmark_root",
            base_path=base_path,
        ),
        fsrs6_adr_policy=_optional_path(
            fsrs6_adr.get("policy"),
            "fsrs6_adr.policy",
            base_path=base_path,
        ),
        fsrs6_adr_policy_root=_optional_path(
            fsrs6_adr.get("policy_root"),
            "fsrs6_adr.policy_root",
            base_path=base_path,
        ),
        fsrs6_adr_train_run_root=_optional_path(
            fsrs6_adr.get("train_run_root"),
            "fsrs6_adr.train_run_root",
            base_path=base_path,
        ),
        fsrs6_adr_policy_manifest=_optional_path(
            fsrs6_adr.get("policy_manifest"),
            "fsrs6_adr.policy_manifest",
            base_path=base_path,
        ),
        fsrs6_adr_lambda_values=_optional_float_list(
            fsrs6_adr.get("lambda_values"),
            "fsrs6_adr.lambda_values",
        ),
        distill_policy_template=_optional_str(
            distill.get("policy_template"),
            "stationary_finite_distill.policy_template",
        ),
        distill_cost_weights=_optional_float_list(
            distill.get("cost_weights"),
            "stationary_finite_distill.cost_weights",
        ),
        out_root=_path(outputs.get("root"), "outputs.root", base_path=base_path),
        no_plot=_bool(experiment.get("no_plot", True), "experiment.no_plot"),
        no_progress=_bool(
            experiment.get("no_progress", True), "experiment.no_progress"
        ),
    )


def _parse_user_subset(
    value: str | None, *, configured: Sequence[int]
) -> tuple[int, ...]:
    if value is None or not value.strip():
        return tuple(configured)
    configured_set = set(configured)
    users: list[int] = []
    for item in value.split(","):
        try:
            user_id = int(item.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid --users entry: {item!r}") from exc
        if user_id not in configured_set:
            raise SystemExit(
                f"User {user_id} is not in the configured user_ids {list(configured)}."
            )
        users.append(user_id)
    if not users:
        raise SystemExit("--users must include at least one user.")
    return tuple(users)


def _csv_token(values: Iterable[float]) -> str:
    return ",".join(format_float(float(value)) for value in values)


def _user_dir(config: TradeoffRunConfig, user_id: int) -> Path:
    return config.out_root / f"user_{user_id}"


def _distill_policy_path(config: TradeoffRunConfig, user_id: int) -> Path | None:
    if config.distill_policy_template is None:
        return None
    path = Path(config.distill_policy_template.format(user_id=user_id)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _distill_policy_template(config: TradeoffRunConfig) -> str | None:
    if config.distill_policy_template is None:
        return None
    if "{user_id}" not in config.distill_policy_template:
        raise ValueError(
            "stationary_finite_distill.policy_template must contain {user_id}."
        )
    path = Path(config.distill_policy_template).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _tradeoff_command(config: TradeoffRunConfig, user_id: int) -> list[str]:
    user_dir = _user_dir(config, user_id)
    command = [
        sys.executable,
        "experiments/single_card_tradeoff/tradeoff.py",
        "--env",
        config.env,
        "--user-id",
        str(user_id),
        "--sched",
        ",".join(config.schedulers),
        "--target-retentions",
        _csv_token(config.target_retentions),
        "--days",
        str(config.days),
        "--particles",
        str(config.particles),
        "--deck-scale",
        str(config.deck_scale),
        "--seed",
        str(config.seed),
        "--scheduler-priority",
        config.scheduler_priority,
        "--benchmark-partition",
        config.benchmark_partition,
        "--out",
        str(user_dir / "results.csv"),
        "--regret-auc-out",
        str(user_dir / "regret_auc.csv"),
    ]
    if config.button_usage is not None:
        command.extend(["--button-usage", str(config.button_usage)])
    if config.review_markov_transition:
        command.append("--review-markov-transition")
    if config.torch_device is not None:
        command.extend(["--torch-device", config.torch_device])
    if config.oracle_cost_weights is not None:
        command.extend(
            ["--oracle-cost-weights", _csv_token(config.oracle_cost_weights)]
        )
    if config.srs_benchmark_root is not None:
        command.extend(["--srs-benchmark-root", str(config.srs_benchmark_root)])
    if config.fsrs6_adr_policy is not None:
        command.extend(["--fsrs6-adr-policy", str(config.fsrs6_adr_policy)])
    if config.fsrs6_adr_policy_root is not None:
        command.extend(["--fsrs6-adr-policy-root", str(config.fsrs6_adr_policy_root)])
    if config.fsrs6_adr_train_run_root is not None:
        command.extend(
            ["--fsrs6-adr-train-run-root", str(config.fsrs6_adr_train_run_root)]
        )
    if config.fsrs6_adr_policy_manifest is not None:
        command.extend(
            ["--fsrs6-adr-policy-manifest", str(config.fsrs6_adr_policy_manifest)]
        )
    if config.fsrs6_adr_lambda_values is not None:
        command.extend(
            ["--fsrs6-adr-lambda-values", _csv_token(config.fsrs6_adr_lambda_values)]
        )
    distill_policy = _distill_policy_path(config, user_id)
    if distill_policy is not None:
        command.extend(
            ["--oracle-stationary-finite-distill-policy", str(distill_policy)]
        )
    if config.distill_cost_weights is not None:
        command.extend(
            [
                "--oracle-stationary-finite-distill-cost-weights",
                _csv_token(config.distill_cost_weights),
            ]
        )
    if config.no_plot:
        command.append("--no-plot")
    if config.no_progress:
        command.append("--no-progress")
    return command


def _multiuser_tradeoff_command(
    config: TradeoffRunConfig,
    user_ids: Sequence[int],
) -> list[str]:
    if not user_ids:
        raise ValueError("user_ids must not be empty.")
    command = [
        sys.executable,
        "experiments/single_card_tradeoff/tradeoff.py",
        "--env",
        config.env,
        "--user-ids",
        ",".join(str(user_id) for user_id in user_ids),
        "--sched",
        ",".join(config.schedulers),
        "--target-retentions",
        _csv_token(config.target_retentions),
        "--days",
        str(config.days),
        "--particles",
        str(config.particles),
        "--deck-scale",
        str(config.deck_scale),
        "--seed",
        str(config.seed),
        "--scheduler-priority",
        config.scheduler_priority,
        "--benchmark-partition",
        config.benchmark_partition,
        "--out",
        str(config.out_root / "combined_results.csv"),
        "--regret-auc-out",
        str(config.out_root / "combined_regret_auc.csv"),
    ]
    if config.button_usage is not None:
        command.extend(["--button-usage", str(config.button_usage)])
    if config.review_markov_transition:
        command.append("--review-markov-transition")
    if config.torch_device is not None:
        command.extend(["--torch-device", config.torch_device])
    if config.oracle_cost_weights is not None:
        command.extend(
            ["--oracle-cost-weights", _csv_token(config.oracle_cost_weights)]
        )
    if config.srs_benchmark_root is not None:
        command.extend(["--srs-benchmark-root", str(config.srs_benchmark_root)])
    if config.fsrs6_adr_policy is not None:
        command.extend(["--fsrs6-adr-policy", str(config.fsrs6_adr_policy)])
    if config.fsrs6_adr_policy_root is not None:
        command.extend(["--fsrs6-adr-policy-root", str(config.fsrs6_adr_policy_root)])
    if config.fsrs6_adr_train_run_root is not None:
        command.extend(
            ["--fsrs6-adr-train-run-root", str(config.fsrs6_adr_train_run_root)]
        )
    if config.fsrs6_adr_policy_manifest is not None:
        command.extend(
            ["--fsrs6-adr-policy-manifest", str(config.fsrs6_adr_policy_manifest)]
        )
    if config.fsrs6_adr_lambda_values is not None:
        command.extend(
            ["--fsrs6-adr-lambda-values", _csv_token(config.fsrs6_adr_lambda_values)]
        )
    distill_policy_template = _distill_policy_template(config)
    if distill_policy_template is not None:
        command.extend(
            [
                "--oracle-stationary-finite-distill-policy-template",
                distill_policy_template,
            ]
        )
    if config.distill_cost_weights is not None:
        command.extend(
            [
                "--oracle-stationary-finite-distill-cost-weights",
                _csv_token(config.distill_cost_weights),
            ]
        )
    if config.no_plot:
        command.append("--no-plot")
    if config.no_progress:
        command.append("--no-progress")
    return command


def _read_csv(path: Path, *, user_id: int) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["user_id"] = user_id
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    has_user_id = any("user_id" in row for row in rows)
    fields: list[str] = ["user_id"] if has_user_id else []
    seen = {"user_id"} if has_user_id else set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows) if rows else ["user_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return float(raw)


def _summary_rows(regret_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in regret_rows:
        if row.get("baseline_scheduler") != "fsrs6":
            continue
        scheduler = str(row.get("scheduler", ""))
        if scheduler == "fsrs6":
            continue
        rows.append(
            {
                "user_id": int(str(row["user_id"])),
                "review_markov_transition": row.get("review_markov_transition"),
                "scheduler": scheduler,
                "baseline_scheduler": row["baseline_scheduler"],
                "same_target_time_saved_auc": row.get("same_target_time_saved_auc"),
                "baseline_time_auc": row.get("baseline_time_auc"),
                "relative_same_target_time_saved_auc_percent": row.get(
                    "relative_same_target_time_saved_auc_percent"
                ),
                "span_coverage_percent": row.get("span_coverage_percent"),
                "covered_target_count": row.get("covered_target_count"),
                "target_count": row.get("target_count"),
                "scheduler_frontier_count": row.get("scheduler_frontier_count"),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["scheduler"],
            str(item.get("review_markov_transition", "")),
            item["user_id"],
        ),
    )


def _mean(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _mean_summary_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (str(row["scheduler"]), str(row.get("review_markov_transition", "")))
            for row in summary_rows
        }
    )
    output: list[dict[str, Any]] = []
    for scheduler, review_markov_transition in keys:
        rows = [
            row
            for row in summary_rows
            if row["scheduler"] == scheduler
            and str(row.get("review_markov_transition", "")) == review_markov_transition
        ]
        time_saved = [
            value
            for row in rows
            if (value := _optional_float(row, "same_target_time_saved_auc")) is not None
        ]
        relative = [
            value
            for row in rows
            if (
                value := _optional_float(
                    row, "relative_same_target_time_saved_auc_percent"
                )
            )
            is not None
        ]
        coverage = [
            value
            for row in rows
            if (value := _optional_float(row, "span_coverage_percent")) is not None
        ]
        output.append(
            {
                "scheduler": scheduler,
                "review_markov_transition": rows[0].get("review_markov_transition")
                if rows
                else review_markov_transition,
                "user_count": len(rows),
                "covered_user_count": len(time_saved),
                "positive_user_count": sum(1 for value in time_saved if value > 0.0),
                "mean_same_target_time_saved_auc": _mean(time_saved),
                "mean_relative_same_target_time_saved_auc_percent": _mean(relative),
                "mean_span_coverage_percent": _mean(coverage),
                "min_span_coverage_percent": min(coverage) if coverage else None,
            }
        )
    return output


def _scheduler_label(scheduler: str) -> str:
    if scheduler == "fsrs6_adr":
        return "ADR"
    if scheduler == "fsrs6_adr_time":
        return "ADR time"
    if scheduler == "fsrs6_oracle_stationary_finite_distill":
        return "476-param distill"
    return scheduler


def _write_grouped_bar_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    values_by_scheduler_user: dict[str, dict[int, float]] = {}
    user_ids = sorted({int(str(row["user_id"])) for row in rows})
    schedulers = sorted({str(row["scheduler"]) for row in rows})
    for row in rows:
        value = _optional_float(row, value_key)
        if value is None:
            continue
        scheduler = str(row["scheduler"])
        user_id = int(str(row["user_id"]))
        values_by_scheduler_user.setdefault(scheduler, {})[user_id] = value
    if not user_ids or not values_by_scheduler_user:
        return

    import matplotlib.pyplot as plt

    width = min(0.8 / max(1, len(schedulers)), 0.36)
    x_positions = list(range(len(user_ids)))
    fig_width = max(8.0, 0.65 * len(user_ids) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    for scheduler_index, scheduler in enumerate(schedulers):
        offset = (scheduler_index - (len(schedulers) - 1) / 2.0) * width
        values = [
            values_by_scheduler_user.get(scheduler, {}).get(user_id, math.nan)
            for user_id in user_ids
        ]
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=_scheduler_label(scheduler),
        )
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xticks(x_positions, [str(user_id) for user_id in user_ids])
    ax.set_xlabel("User ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_plots(
    out_root: Path,
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_grouped_bar_plot(
        out_root / "same_target_time_saved_auc_by_user.png",
        summary_rows,
        value_key="same_target_time_saved_auc",
        ylabel="Deck minutes/day saved vs FSRS6",
        title="Same-Target Time Saved AUC By User",
    )
    _write_grouped_bar_plot(
        out_root / "relative_time_saved_by_user.png",
        summary_rows,
        value_key="relative_same_target_time_saved_auc_percent",
        ylabel="Relative time saved vs FSRS6 (%)",
        title="Relative Same-Target Time Saved By User",
    )
    _write_grouped_bar_plot(
        out_root / "span_coverage_by_user.png",
        summary_rows,
        value_key="span_coverage_percent",
        ylabel="Baseline memory span covered (%)",
        title="Common Span Coverage By User",
    )


def _write_aggregate_outputs(
    config: TradeoffRunConfig,
    *,
    result_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(config.out_root / "combined_results.csv", result_rows)
    _write_csv(config.out_root / "combined_regret_auc.csv", regret_rows)
    summary = _summary_rows(regret_rows)
    _write_csv(config.out_root / "summary.csv", summary)
    _write_csv(config.out_root / "mean_summary.csv", _mean_summary_rows(summary))
    _write_summary_plots(config.out_root, summary)


def _combine_outputs(config: TradeoffRunConfig, user_ids: Sequence[int]) -> None:
    result_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    for user_id in user_ids:
        user_dir = _user_dir(config, user_id)
        result_rows.extend(_read_csv(user_dir / "results.csv", user_id=user_id))
        regret_rows.extend(_read_csv(user_dir / "regret_auc.csv", user_id=user_id))
    _write_aggregate_outputs(config, result_rows=result_rows, regret_rows=regret_rows)


def _split_combined_outputs(config: TradeoffRunConfig, user_ids: Sequence[int]) -> None:
    result_rows = _read_csv_rows(config.out_root / "combined_results.csv")
    regret_rows = _read_csv_rows(config.out_root / "combined_regret_auc.csv")
    user_set = {int(user_id) for user_id in user_ids}
    for user_id in user_ids:
        user_dir = _user_dir(config, int(user_id))
        user_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            user_dir / "results.csv",
            [
                row
                for row in result_rows
                if int(str(row.get("user_id", "0") or "0")) == int(user_id)
            ],
        )
        _write_csv(
            user_dir / "regret_auc.csv",
            [
                row
                for row in regret_rows
                if int(str(row.get("user_id", "0") or "0")) == int(user_id)
            ],
        )
    unexpected = sorted(
        {
            int(str(row.get("user_id", "0") or "0"))
            for row in result_rows
            if int(str(row.get("user_id", "0") or "0")) not in user_set
        }
    )
    if unexpected:
        raise SystemExit(
            "Combined results contained unexpected user IDs: "
            + ",".join(str(user_id) for user_id in unexpected)
        )
    _write_aggregate_outputs(config, result_rows=result_rows, regret_rows=regret_rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    user_ids = _parse_user_subset(args.users, configured=config.user_ids)
    use_multiuser_batch = len(user_ids) > 1 and not args.no_multiuser_batch
    if use_multiuser_batch:
        command = _multiuser_tradeoff_command(config, user_ids)
        if args.dry_run:
            print(" ".join(command))
            return 0
        results_path = config.out_root / "combined_results.csv"
        regret_path = config.out_root / "combined_regret_auc.csv"
        if not args.force and results_path.exists() and regret_path.exists():
            print("Skipping multi-user batch: combined outputs already exist.")
        else:
            config.out_root.mkdir(parents=True, exist_ok=True)
            print(
                "Running multi-user batch "
                + ",".join(str(user_id) for user_id in user_ids),
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        _split_combined_outputs(config, user_ids)
        print(f"Wrote combined outputs under {config.out_root}")
        return 0

    for user_id in user_ids:
        user_dir = _user_dir(config, user_id)
        results_path = user_dir / "results.csv"
        regret_path = user_dir / "regret_auc.csv"
        command = _tradeoff_command(config, user_id)
        if args.dry_run:
            print(" ".join(command))
            continue
        if not args.force and results_path.exists() and regret_path.exists():
            print(f"Skipping user_{user_id}: outputs already exist.")
            continue
        user_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running user_{user_id}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not args.dry_run:
        _combine_outputs(config, user_ids)
        print(f"Wrote combined outputs under {config.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
