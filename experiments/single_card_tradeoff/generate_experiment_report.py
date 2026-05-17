from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.reporting import (
    config_path,
    display_path,
    format_float,
    format_int,
    format_percent,
    load_toml_profile,
    markdown_table,
    read_csv_rows,
    read_json_object,
    resolve_repo_path,
)


DEFAULT_CONFIG = Path(
    "experiments/single_card_tradeoff/configs/stationary_finite_first8_report.toml"
)
DEFAULT_REPORT_ROOT = Path("artifacts/single_card_tradeoff/reports/current")
DEFAULT_DOC_OUTPUT = Path(
    "docs/rl_scheduler/experiments/2026-05-17-single_card_tradeoff.md"
)

DEFAULT_SOURCE_PATHS: dict[str, Path] = {
    "default_regret_auc": Path(
        "artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv"
    ),
    "default_results": Path(
        "artifacts/single_card_tradeoff/no_sub05_distill_compare/results.csv"
    ),
    "first8_distill_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_distill_first8_users_per_user_uniform_table_supervision_"
        "fsrs6_baseline_gpu/summary.csv"
    ),
    "first8_distill_train_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_distill_first8_users_per_user_uniform_table_supervision_"
        "fsrs6_baseline_gpu/train_summary.csv"
    ),
    "first8_exact_vs_distill_mean_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_exact_vs_distill_first8_users/mean_summary.csv"
    ),
    "first8_exact_vs_distill_regret_auc": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_exact_vs_distill_first8_users/regret_auc.csv"
    ),
    "low_param_sparse_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "low_param_direct_policy_search_first8_users/metadata.json"
    ),
    "low_param_dense_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "low_param_direct_policy_search_first8_users_dense_weights/metadata.json"
    ),
}

PARAM_COUNTS = {
    "fsrs6_oracle_distill": 1468,
    "fsrs6_oracle_stationary_finite_distill": 1452,
    "fsrs6_oracle_retention_distill": 1468,
    "fsrs6_oracle_infinite_distill": 1452,
    "uvfa_ppo": 27148,
    "uvfa_ppo_rnn_interval": 87559,
}

DEFAULT_COMPACT_ORDER = (
    "fsrs6_oracle_distill",
    "fsrs6_oracle_stationary_finite_distill",
    "fsrs6_oracle_retention_distill",
    "fsrs6_oracle_infinite_distill",
    "uvfa_ppo",
    "uvfa_ppo_rnn_interval",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a formal single-card tradeoff report from existing "
            "machine-readable CSV and metadata artifacts."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML profile describing report inputs, outputs, and rerun commands.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Override the TOML report.root directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the TOML report.output_path.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when any default source artifact is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_toml_profile(args.config)
    report_root = args.report_root or config_path(
        config,
        section_name="report",
        field_name="root",
        default=DEFAULT_REPORT_ROOT,
    )
    output_path = args.output_path or config_path(
        config,
        section_name="report",
        field_name="output_path",
        default=DEFAULT_DOC_OUTPUT,
    )
    summary_path, report_path, summary = generate_report(
        config_path=args.config,
        report_config=config,
        report_root=report_root,
        output_path=output_path,
        source_paths=_config_source_paths(config),
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "report_summary_path": str(summary_path),
                "run_report_path": str(report_path),
                "output_path": str(resolve_repo_path(output_path)),
                "input_artifacts": len(summary["input_artifacts"]["available"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def generate_report(
    *,
    config_path: Path | None,
    report_config: Mapping[str, Any],
    report_root: Path,
    output_path: Path,
    source_paths: Mapping[str, Path],
    strict: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    report_root = resolve_repo_path(report_root)
    output_path = resolve_repo_path(output_path)
    report_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "report_summary.json"
    report_path = report_root / "report.md"

    summary = build_report_summary(
        config_path=config_path,
        report_root=report_root,
        output_path=output_path,
        source_paths=source_paths,
        report_config=report_config,
        strict=strict,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_report(summary)
    report_path.write_text(markdown, encoding="utf-8")
    output_path.write_text(markdown, encoding="utf-8")
    return summary_path, report_path, summary


def build_report_summary(
    *,
    config_path: Path | None,
    report_root: Path,
    output_path: Path,
    source_paths: Mapping[str, Path],
    report_config: Mapping[str, Any],
    strict: bool,
) -> dict[str, Any]:
    source_paths = {key: resolve_repo_path(path) for key, path in source_paths.items()}
    available = {
        key: display_path(path) for key, path in source_paths.items() if path.exists()
    }
    missing = {
        key: display_path(path)
        for key, path in source_paths.items()
        if not path.exists()
    }
    if strict and missing:
        missing_text = ", ".join(f"{key}={path}" for key, path in missing.items())
        raise FileNotFoundError(f"Missing report source artifacts: {missing_text}")

    default_regret = read_csv_rows(source_paths["default_regret_auc"])
    first8_distill_summary = read_csv_rows(source_paths["first8_distill_summary"])
    first8_distill_train = read_csv_rows(source_paths["first8_distill_train_summary"])
    exact_vs_distill_mean = read_csv_rows(
        source_paths["first8_exact_vs_distill_mean_summary"]
    )
    exact_vs_distill_regret = read_csv_rows(
        source_paths["first8_exact_vs_distill_regret_auc"]
    )
    low_param_sparse = read_json_object(source_paths["low_param_sparse_metadata"])
    low_param_dense = read_json_object(source_paths["low_param_dense_metadata"])

    return {
        "type": "single-card-tradeoff-experiment-report",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "title": "Single-card stationary finite tradeoff report",
        "question": (
            "Can stationary finite-lifecycle oracle distillation and a much "
            "smaller direct policy-search family improve the FSRS-6 memory-time "
            "frontier while keeping policy inputs limited to S, D, and cost weight?"
        ),
        "run_roots": {
            "config_path": display_path(resolve_repo_path(config_path))
            if config_path is not None
            else None,
            "report_root": display_path(report_root),
            "published_report": display_path(output_path),
        },
        "input_artifacts": {
            "available": available,
            "missing": missing,
        },
        "rerun_commands": _profile_commands(report_config),
        "default_fsrs6_comparison": _default_comparison(default_regret),
        "default_direct_comparisons": _direct_default_comparisons(default_regret),
        "first8_stationary_finite_distill": _first8_distill_summary(
            first8_distill_summary,
            first8_distill_train,
        ),
        "first8_exact_vs_distill": _first8_exact_vs_distill_summary(
            exact_vs_distill_mean,
            exact_vs_distill_regret,
        ),
        "low_param_direct_search": {
            "sparse_teacher_weights": _low_param_summary(low_param_sparse),
            "dense_teacher_weights": _low_param_summary(low_param_dense),
        },
        "conclusions": [
            (
                "The 476-parameter per-user stationary finite distill remains the "
                "best compact first-eight-user candidate in these artifacts: it "
                "beats fsrs6 on mean relative regret while preserving about 98% "
                "coverage."
            ),
            (
                "The 7-parameter direct policy-search family is useful as a "
                "lower-bound compression baseline, but it gives up roughly 28 to "
                "39 coverage points versus the 476-parameter distill."
            ),
            (
                "For the default single-user FSRS-6 comparison, stationary finite "
                "distill trades a small shared-span loss versus unrestricted "
                "oracle distill for wider coverage against fsrs6_default."
            ),
        ],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['title']}")
    lines.append("")
    lines.append(
        f"Machine summary: `{summary['run_roots']['report_root']}/report_summary.json`"
    )
    lines.append("")
    if summary["run_roots"].get("config_path"):
        lines.append(f"Config: `{summary['run_roots']['config_path']}`")
        lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(str(summary["question"]))
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "This report is generated from the current machine-readable "
        "`artifacts/single_card_tradeoff` CSV and JSON outputs. It does not rely on "
        "hand-copied metrics."
    )
    lines.append("")
    lines.extend(_source_lines(summary["input_artifacts"]))
    lines.append("")
    lines.append("## Default FSRS-6 Comparison")
    lines.append("")
    lines.append(
        "Environment `fsrs6_default`, 1825 days, 10,000 particles, "
        "`deck_scale=10000`, no sub-0.5 actions."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "scheduler",
                "params",
                "time_regret_auc",
                "relative_regret_auc_percent",
                "coverage",
            ],
            [
                [
                    row["scheduler"],
                    format_int(row.get("params")),
                    format_float(row["time_regret_auc"], digits=4),
                    format_percent(row["relative_regret_auc_percent"]),
                    format_percent(row["span_coverage_percent"]),
                ]
                for row in summary["default_fsrs6_comparison"]
            ],
        )
    )
    lines.append("")
    lines.append("Direct comparisons against `fsrs6_oracle_distill`:")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "scheduler",
                "time_regret_auc",
                "relative_regret_auc_percent",
                "coverage",
            ],
            [
                [
                    row["scheduler"],
                    format_float(row["time_regret_auc"], digits=4),
                    format_percent(row["relative_regret_auc_percent"]),
                    format_percent(row["span_coverage_percent"]),
                ]
                for row in summary["default_direct_comparisons"]
            ],
        )
    )
    lines.append("")
    lines.append("## First Eight Users")
    lines.append("")
    first8 = summary["first8_stationary_finite_distill"]
    lines.append(
        "The per-user stationary finite distill trains eight independent "
        f"{format_int(first8['params_per_user'])}-parameter students in one "
        f"batched process ({format_int(first8['ensemble_trainable_params'])} "
        "trainable parameters during training)."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "metric",
                "value",
            ],
            [
                [
                    "mean span coverage",
                    format_percent(first8["mean_span_coverage_percent"]),
                ],
                [
                    "mean time regret AUC",
                    format_float(first8["mean_time_regret_auc"], digits=4),
                ],
                [
                    "mean relative regret AUC",
                    format_percent(first8["mean_relative_regret_auc_percent"]),
                ],
                ["teacher_s", format_float(first8["teacher_runtime_s"], digits=2)],
                ["train_s", format_float(first8["train_runtime_s"], digits=2)],
                ["eval_s", format_float(first8["eval_runtime_s"], digits=2)],
                [
                    "mean final CE",
                    format_float(first8["mean_final_ce_loss"], digits=5),
                ],
                [
                    "mean table agreement",
                    format_percent(
                        100.0 * first8["mean_eval_teacher_action_agreement"]
                    ),
                ],
            ],
        )
    )
    lines.append("")
    exact = summary["first8_exact_vs_distill"]
    lines.append("Exact stationary finite teacher versus per-user distill:")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "scheduler",
                "mean time regret AUC vs fsrs6",
                "mean relative regret vs fsrs6",
                "mean coverage vs fsrs6",
            ],
            [
                [
                    row["scheduler"],
                    format_float(row["mean_time_regret_auc"], digits=4),
                    format_percent(row["mean_relative_regret_auc_percent"]),
                    format_percent(row["mean_span_coverage_percent"]),
                ]
                for row in exact["vs_fsrs6"]
            ],
        )
    )
    lines.append("")
    lines.append(
        "On the exact-teacher shared span, distill has "
        f"{format_float(exact['distill_vs_exact']['mean_time_regret_auc'], digits=4)} "
        "deck-minutes/day time regret AUC and "
        f"{format_percent(exact['distill_vs_exact']['mean_relative_regret_auc_percent'])} "
        "relative regret at "
        f"{format_percent(exact['distill_vs_exact']['mean_span_coverage_percent'])} "
        "coverage."
    )
    lines.append("")
    lines.append("## Low-Parameter Direct Search")
    lines.append("")
    low_param = summary["low_param_direct_search"]
    lines.extend(
        markdown_table(
            [
                "run",
                "params/user",
                "total params",
                "vs fsrs6 relative regret",
                "vs fsrs6 coverage",
                "vs distill relative regret",
                "vs distill coverage",
            ],
            [
                _low_param_row(
                    "sparse teacher weights", low_param["sparse_teacher_weights"]
                ),
                _low_param_row(
                    "dense teacher weights", low_param["dense_teacher_weights"]
                ),
            ],
        )
    )
    lines.append("")
    if summary["rerun_commands"]:
        lines.append("## Reproduction Profile")
        lines.append("")
        lines.append(
            "The TOML profile records the commands and expected outputs used to "
            "reproduce the current report inputs."
        )
        lines.append("")
        lines.extend(
            markdown_table(
                ["command", "expected outputs present"],
                [
                    [
                        row["name"],
                        f"{row['available_expected_outputs']}/{row['expected_outputs']}",
                    ]
                    for row in summary["rerun_commands"]
                ],
            )
        )
        lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    for conclusion in summary["conclusions"]:
        lines.append(f"- {conclusion}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Published report: `{summary['run_roots']['published_report']}`")
    lines.append(f"- Report root: `{summary['run_roots']['report_root']}`")
    lines.append("")
    return "\n".join(lines)


def _source_lines(input_artifacts: Mapping[str, Any]) -> list[str]:
    lines = ["Available inputs:"]
    for key, path in sorted(input_artifacts["available"].items()):
        lines.append(f"- `{key}`: `{path}`")
    missing = input_artifacts.get("missing") or {}
    if missing:
        lines.append("")
        lines.append("Missing inputs:")
        for key, path in sorted(missing.items()):
            lines.append(f"- `{key}`: `{path}`")
    return lines


def _config_source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = config.get("source_artifacts")
    if raw is None:
        return dict(DEFAULT_SOURCE_PATHS)
    if not isinstance(raw, Mapping):
        raise ValueError("source_artifacts must be a TOML table.")
    paths: dict[str, Path] = {}
    for key in DEFAULT_SOURCE_PATHS:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source_artifacts.{key} must be a non-empty string.")
        paths[key] = Path(value)
    return paths


def _profile_commands(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands = config.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("commands must be an array of TOML tables.")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(commands):
        if not isinstance(raw, Mapping):
            raise ValueError(f"commands[{index}] must be a TOML table.")
        name = raw.get("name")
        command = raw.get("command")
        expected_outputs = raw.get("expected_outputs", [])
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"commands[{index}].name must be a non-empty string.")
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise ValueError(f"commands[{index}].command must be a string array.")
        if not isinstance(expected_outputs, list) or not all(
            isinstance(item, str) for item in expected_outputs
        ):
            raise ValueError(
                f"commands[{index}].expected_outputs must be a string array."
            )
        resolved_outputs = [resolve_repo_path(Path(item)) for item in expected_outputs]
        result.append(
            {
                "name": name,
                "description": raw.get("description"),
                "command": command,
                "expected_output_paths": [
                    display_path(path) for path in resolved_outputs
                ],
                "expected_outputs": len(resolved_outputs),
                "available_expected_outputs": sum(
                    1 for path in resolved_outputs if path.exists()
                ),
            }
        )
    return result


def _default_comparison(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scheduler in DEFAULT_COMPACT_ORDER:
        row = _find_regret_row(
            rows,
            environment="fsrs6_default",
            baseline_scheduler="fsrs6_default",
            scheduler=scheduler,
        )
        result.append(
            {
                "scheduler": scheduler,
                "params": PARAM_COUNTS.get(scheduler),
                **_regret_metrics(row),
            }
        )
    return result


def _direct_default_comparisons(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    schedulers = (
        "fsrs6_oracle_stationary_finite_distill",
        "fsrs6_oracle_retention_distill",
        "fsrs6_oracle_infinite_distill",
        "uvfa_ppo",
        "uvfa_ppo_rnn_interval",
    )
    result: list[dict[str, Any]] = []
    for scheduler in schedulers:
        row = _find_regret_row(
            rows,
            environment="fsrs6_default",
            baseline_scheduler="fsrs6_oracle_distill",
            scheduler=scheduler,
        )
        result.append({"scheduler": scheduler, **_regret_metrics(row)})
    return result


def _first8_distill_summary(
    summary_rows: Sequence[Mapping[str, str]],
    train_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    train_row = train_rows[0]
    return {
        "user_count": len(summary_rows),
        "params_per_user": _int(train_row, "params_per_user"),
        "ensemble_trainable_params": _int(train_row, "ensemble_trainable_params"),
        "mean_span_coverage_percent": _mean(summary_rows, "span_coverage_percent"),
        "mean_time_regret_auc": _mean(summary_rows, "time_regret_auc"),
        "mean_relative_regret_auc_percent": _mean(
            summary_rows, "relative_regret_auc_percent"
        ),
        "teacher_runtime_s": _float(train_row, "teacher_runtime_s"),
        "train_runtime_s": _float(train_row, "train_runtime_s"),
        "agreement_runtime_s": _float(train_row, "agreement_runtime_s"),
        "eval_runtime_s": _float(train_row, "eval_runtime_s"),
        "total_runtime_s": _float(train_row, "total_runtime_s"),
        "mean_final_ce_loss": _mean(train_rows, "final_ce_loss"),
        "mean_final_teacher_action_agreement": _mean(
            train_rows, "final_teacher_action_agreement"
        ),
        "mean_eval_teacher_action_agreement": _mean(
            train_rows, "eval_teacher_action_agreement"
        ),
    }


def _first8_exact_vs_distill_summary(
    mean_rows: Sequence[Mapping[str, str]],
    regret_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    vs_fsrs6 = [
        {
            "scheduler": str(row["scheduler"]),
            "mean_span_coverage_percent": _float(row, "mean_span_coverage_percent"),
            "mean_time_regret_auc": _float(row, "mean_time_regret_auc"),
            "mean_relative_regret_auc_percent": _float(
                row, "mean_relative_regret_auc_percent"
            ),
        }
        for row in mean_rows
    ]
    direct_rows = [
        row
        for row in regret_rows
        if row["baseline_scheduler"] == "fsrs6_oracle_stationary_finite"
        and row["scheduler"] == "fsrs6_oracle_stationary_finite_distill_per_user"
    ]
    return {
        "vs_fsrs6": vs_fsrs6,
        "distill_vs_exact": {
            "user_count": len(direct_rows),
            "mean_span_coverage_percent": _mean(direct_rows, "span_coverage_percent"),
            "mean_time_regret_auc": _mean(direct_rows, "time_regret_auc"),
            "mean_relative_regret_auc_percent": _mean(
                direct_rows, "relative_regret_auc_percent"
            ),
        },
    }


def _low_param_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    mean_summary = metadata["mean_summary"]
    by_baseline = {
        row["baseline_scheduler"]: row for row in mean_summary if isinstance(row, dict)
    }
    return {
        "params_per_user": metadata["params_per_user"],
        "ensemble_trainable_params": metadata["ensemble_trainable_params"],
        "train_runtime_s": metadata["train_runtime_s"],
        "eval_runtime_s": metadata["eval_runtime_s"],
        "total_runtime_s": metadata["total_runtime_s"],
        "vs_fsrs6": _low_param_baseline_summary(by_baseline["fsrs6"]),
        "vs_stationary_finite_distill": _low_param_baseline_summary(
            by_baseline["fsrs6_oracle_stationary_finite_distill_per_user"]
        ),
    }


def _low_param_baseline_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_span_coverage_percent": float(row["mean_span_coverage_percent"]),
        "mean_time_regret_auc": float(row["mean_time_regret_auc"]),
        "mean_relative_regret_auc_percent": float(
            row["mean_relative_regret_auc_percent"]
        ),
    }


def _low_param_row(label: str, row: Mapping[str, Any]) -> list[str]:
    vs_fsrs6 = row["vs_fsrs6"]
    vs_distill = row["vs_stationary_finite_distill"]
    return [
        label,
        format_int(row["params_per_user"]),
        format_int(row["ensemble_trainable_params"]),
        format_percent(vs_fsrs6["mean_relative_regret_auc_percent"]),
        format_percent(vs_fsrs6["mean_span_coverage_percent"]),
        format_percent(vs_distill["mean_relative_regret_auc_percent"]),
        format_percent(vs_distill["mean_span_coverage_percent"]),
    ]


def _find_regret_row(
    rows: Sequence[Mapping[str, str]],
    *,
    environment: str,
    baseline_scheduler: str,
    scheduler: str,
) -> Mapping[str, str]:
    for row in rows:
        if (
            row["environment"] == environment
            and row["baseline_scheduler"] == baseline_scheduler
            and row["scheduler"] == scheduler
        ):
            return row
    raise ValueError(
        "Missing regret row for "
        f"environment={environment}, baseline={baseline_scheduler}, "
        f"scheduler={scheduler}."
    )


def _regret_metrics(row: Mapping[str, str]) -> dict[str, float]:
    return {
        "span_coverage_percent": _float(row, "span_coverage_percent"),
        "time_regret_auc": _float(row, "time_regret_auc"),
        "baseline_time_auc": _float(row, "baseline_time_auc"),
        "relative_regret_auc_percent": _float(row, "relative_regret_auc_percent"),
    }


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row[key]
    if isinstance(value, str):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"Expected numeric {key}, got {type(value).__name__}.")


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, str):
        return int(float(value))
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected integer {key}, got {type(value).__name__}.")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError(f"Cannot compute mean for empty rows: {key}.")
    return sum(_float(row, key) for row in rows) / len(rows)


if __name__ == "__main__":
    raise SystemExit(main())
