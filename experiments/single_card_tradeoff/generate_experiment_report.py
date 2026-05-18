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
DEFAULT_DOC_OUTPUT = Path("docs/single_card_tradeoff/experiments/2026-05-17-index.md")

REPORT_FILENAMES = {
    "index": "index.md",
    "default_no_sub05_tradeoff": "default_no_sub05_tradeoff.md",
    "first8_stationary_finite_distill": "first8_stationary_finite_distill.md",
    "first8_exact_vs_distill": "first8_exact_vs_distill.md",
    "low_param_direct_search": "low_param_direct_search.md",
}

DEFAULT_DOC_OUTPUTS = {
    "index": DEFAULT_DOC_OUTPUT,
    "default_no_sub05_tradeoff": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-default_no_sub05_tradeoff.md"
    ),
    "first8_stationary_finite_distill": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-17-first8_stationary_finite_distill.md"
    ),
    "first8_exact_vs_distill": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-first8_exact_vs_distill.md"
    ),
    "low_param_direct_search": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-low_param_direct_search.md"
    ),
}

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
    output_paths = _config_report_outputs(config, index_output_path=output_path)
    summary_path, report_path, summary = generate_report(
        config_path=args.config,
        report_config=config,
        report_root=report_root,
        output_paths=output_paths,
        source_paths=_config_source_paths(config),
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "report_summary_path": str(summary_path),
                "run_report_path": str(report_path),
                "output_path": str(resolve_repo_path(output_paths["index"])),
                "output_paths": {
                    key: str(resolve_repo_path(path))
                    for key, path in output_paths.items()
                },
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
    output_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
    strict: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    report_root = resolve_repo_path(report_root)
    resolved_output_paths = {
        key: resolve_repo_path(path) for key, path in output_paths.items()
    }
    report_root.mkdir(parents=True, exist_ok=True)
    for output_path in resolved_output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "report_summary.json"
    future_available_paths = {
        summary_path,
        *(report_root / filename for filename in REPORT_FILENAMES.values()),
        *resolved_output_paths.values(),
    }

    summary = build_report_summary(
        config_path=config_path,
        report_root=report_root,
        output_paths=resolved_output_paths,
        source_paths=source_paths,
        report_config=report_config,
        future_available_paths=future_available_paths,
        strict=strict,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reports = render_reports(summary)
    for key, markdown in reports.items():
        report_path = report_root / REPORT_FILENAMES[key]
        report_path.write_text(markdown, encoding="utf-8")
        resolved_output_paths[key].write_text(markdown, encoding="utf-8")
    return summary_path, report_root / REPORT_FILENAMES["index"], summary


def build_report_summary(
    *,
    config_path: Path | None,
    report_root: Path,
    output_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
    report_config: Mapping[str, Any],
    future_available_paths: set[Path],
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
            "published_report": display_path(output_paths["index"]),
            "published_reports": {
                key: display_path(path) for key, path in output_paths.items()
            },
            "report_artifacts": {
                key: display_path(report_root / REPORT_FILENAMES[key])
                for key in REPORT_FILENAMES
            },
        },
        "input_artifacts": {
            "available": available,
            "missing": missing,
        },
        "rerun_commands": _profile_commands(
            report_config,
            future_available_paths=future_available_paths,
        ),
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


def render_reports(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        "index": render_index_report(summary),
        "default_no_sub05_tradeoff": render_default_no_sub05_tradeoff_report(summary),
        "first8_stationary_finite_distill": (
            render_first8_stationary_finite_distill_report(summary)
        ),
        "first8_exact_vs_distill": render_first8_exact_vs_distill_report(summary),
        "low_param_direct_search": render_low_param_direct_search_report(summary),
    }


def render_index_report(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Single-card Tradeoff Experiment Reports")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "This index links the independent reports generated from the current "
            "single-card tradeoff artifacts."
        ),
    )
    lines.append("## Reports")
    lines.append("")
    published = summary["run_roots"]["published_reports"]
    stationary_default = next(
        row
        for row in summary["default_fsrs6_comparison"]
        if row["scheduler"] == "fsrs6_oracle_stationary_finite_distill"
    )
    first8 = summary["first8_stationary_finite_distill"]
    distill_vs_exact = summary["first8_exact_vs_distill"]["distill_vs_exact"]
    lines.extend(
        markdown_table(
            ["experiment", "published report", "main result"],
            [
                [
                    "default no-sub-0.5 tradeoff",
                    f"`{published['default_no_sub05_tradeoff']}`",
                    (
                        "`fsrs6_oracle_stationary_finite_distill` keeps "
                        f"{format_percent(stationary_default['span_coverage_percent'])} "
                        "coverage vs `fsrs6_default`."
                    ),
                ],
                [
                    "first-eight per-user distill",
                    f"`{published['first8_stationary_finite_distill']}`",
                    (
                        "Eight independent 476-parameter students average "
                        f"{format_percent(first8['mean_relative_regret_auc_percent'])} "
                        "relative regret vs `fsrs6`."
                    ),
                ],
                [
                    "first-eight exact vs distill",
                    f"`{published['first8_exact_vs_distill']}`",
                    (
                        "Direct distill-vs-exact relative regret is "
                        f"{format_percent(distill_vs_exact['mean_relative_regret_auc_percent'])} "
                        "over "
                        f"{format_percent(distill_vs_exact['mean_span_coverage_percent'])} "
                        "shared coverage."
                    ),
                ],
                [
                    "low-parameter direct search",
                    f"`{published['low_param_direct_search']}`",
                    (
                        "The 7-parameter family is a useful lower-capacity "
                        "baseline but loses substantial coverage."
                    ),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "All reports are generated from the current machine-readable "
        "`artifacts/single_card_tradeoff` CSV and JSON outputs."
    )
    lines.append("")
    lines.extend(_source_lines(summary["input_artifacts"]))
    lines.append("")
    _append_reproduction_profile(lines, summary, command_names=None)
    lines.append("## Conclusions")
    lines.append("")
    for conclusion in summary["conclusions"]:
        lines.append(f"- {conclusion}")
    lines.append("")
    return "\n".join(lines)


def render_default_no_sub05_tradeoff_report(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Default No-sub-0.5 Tradeoff")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "How do the clipped compact single-card policies compare against "
            "`fsrs6_default` when action and target retentions below 0.5 are "
            "removed?"
        ),
    )
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "Environment `fsrs6_default`, 1825 days, 10,000 particles, "
        "`deck_scale=10000`, no sub-0.5 actions."
    )
    lines.append("")
    lines.extend(
        _artifact_lines(summary, keys=("default_regret_auc", "default_results"))
    )
    lines.append("")
    lines.append("## Results")
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
    _append_reproduction_profile(
        lines,
        summary,
        command_names=("evaluate_default_no_sub05_tradeoff",),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "The stationary finite distill is slightly worse than unrestricted "
        "oracle distill on their shared span, but it covers more of the "
        "`fsrs6_default` frontier under the clipped action space."
    )
    lines.append("")
    _append_artifacts_footer(lines, summary, report_key="default_no_sub05_tradeoff")
    return "\n".join(lines)


def render_first8_stationary_finite_distill_report(
    summary: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# First-eight Per-user Stationary Finite Distill")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "Can independent 476-parameter stationary finite students trained in "
            "one batched process improve the first-eight-user FSRS-6 tradeoff?"
        ),
    )
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "Environment `fsrs6`, users 1 through 8, one independent student per "
        "user, uniform exact-table supervision over `(cost_weight, stability, "
        "difficulty)`, and `fsrs6` as the baseline."
    )
    lines.append("")
    lines.extend(
        _artifact_lines(
            summary,
            keys=("first8_distill_summary", "first8_distill_train_summary"),
        )
    )
    lines.append("")
    lines.append("## Results")
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
    _append_reproduction_profile(
        lines,
        summary,
        command_names=(
            "train_first8_stationary_finite_distill",
            "evaluate_first8_stationary_finite_distill",
        ),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "Uniform exact-table supervision fixed the high-cost interpolation failure "
        "without adding teacher cost weights. The current first-eight artifact "
        "keeps about 98% coverage and improves mean relative regret versus "
        "`fsrs6`."
    )
    lines.append("")
    _append_artifacts_footer(
        lines,
        summary,
        report_key="first8_stationary_finite_distill",
    )
    return "\n".join(lines)


def render_first8_exact_vs_distill_report(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# First-eight Exact Stationary Finite vs Distill")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "How does the evaluated tradeoff change when replacing exact "
            "stationary finite policy tables with per-user 476-parameter "
            "distills?"
        ),
    )
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "Environment `fsrs6`, users 1 through 8. The exact and distill policies "
        "are both evaluated against `fsrs6`; a direct regret row also compares "
        "the distill to the exact stationary finite teacher on their shared "
        "frontier span."
    )
    lines.append("")
    lines.extend(
        _artifact_lines(
            summary,
            keys=(
                "first8_exact_vs_distill_mean_summary",
                "first8_exact_vs_distill_regret_auc",
            ),
        )
    )
    lines.append("")
    lines.append("## Results")
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
    _append_reproduction_profile(
        lines,
        summary,
        command_names=("evaluate_first8_exact_vs_distill",),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "The per-user distill is not dominated in this sampled tradeoff "
        "evaluation: direct distill-vs-exact relative regret is negative on the "
        "shared span. The exact table remains the teacher and diagnostic target; "
        "the distill is the compact deployable approximation."
    )
    lines.append("")
    _append_artifacts_footer(lines, summary, report_key="first8_exact_vs_distill")
    return "\n".join(lines)


def render_low_param_direct_search_report(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Low-parameter Direct Policy Search")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "Can a 7-parameter direct desired-retention function optimized by "
            "evolutionary search replace the 476-parameter stationary finite "
            "distill?"
        ),
    )
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "Environment `fsrs6`, users 1 through 8. Both sparse and dense "
        "teacher-cost-weight direct-search runs are evaluated against `fsrs6` "
        "and against the per-user stationary finite distill."
    )
    lines.append("")
    lines.extend(
        _artifact_lines(
            summary,
            keys=("low_param_sparse_metadata", "low_param_dense_metadata"),
        )
    )
    lines.append("")
    lines.append("## Results")
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
    _append_reproduction_profile(
        lines,
        summary,
        command_names=(
            "train_low_param_direct_sparse",
            "train_low_param_direct_dense",
        ),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "The 7-parameter family is informative as a capacity floor, but it does "
        "not currently replace the 476-parameter distill: both sparse and dense "
        "runs lose substantial frontier coverage."
    )
    lines.append("")
    _append_artifacts_footer(lines, summary, report_key="low_param_direct_search")
    return "\n".join(lines)


def _append_report_preamble(
    lines: list[str],
    summary: Mapping[str, Any],
    *,
    question: str,
) -> None:
    lines.append(
        f"Machine summary: `{summary['run_roots']['report_root']}/report_summary.json`"
    )
    if summary["run_roots"].get("config_path"):
        lines.append(f"Config: `{summary['run_roots']['config_path']}`")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(question)
    lines.append("")


def _artifact_lines(
    summary: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> list[str]:
    available = summary["input_artifacts"]["available"]
    missing = summary["input_artifacts"].get("missing") or {}
    lines = ["Source artifacts:"]
    for key in keys:
        if key in available:
            lines.append(f"- `{key}`: `{available[key]}`")
        elif key in missing:
            lines.append(f"- `{key}`: missing `{missing[key]}`")
        else:
            lines.append(f"- `{key}`: not configured")
    return lines


def _append_reproduction_profile(
    lines: list[str],
    summary: Mapping[str, Any],
    *,
    command_names: Sequence[str] | None,
) -> None:
    commands = _select_commands(summary["rerun_commands"], command_names)
    if not commands:
        return
    lines.append("## Reproduction Profile")
    lines.append("")
    lines.append(
        "The TOML profile records the command and expected outputs used to "
        "reproduce this report input. CUDA reruns write "
        "`performance_summary.json` and `gpu_monitor/` memory samples under "
        "their configured output directories."
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
                for row in commands
            ],
        )
    )
    lines.append("")


def _select_commands(
    commands: Sequence[Mapping[str, Any]],
    command_names: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    if command_names is None:
        return list(commands)
    wanted = set(command_names)
    return [row for row in commands if row["name"] in wanted]


def _append_artifacts_footer(
    lines: list[str],
    summary: Mapping[str, Any],
    *,
    report_key: str,
) -> None:
    lines.append("## Artifacts")
    lines.append("")
    published = summary["run_roots"]["published_reports"][report_key]
    artifact = summary["run_roots"]["report_artifacts"][report_key]
    lines.append(f"- Published report: `{published}`")
    lines.append(f"- Artifact report: `{artifact}`")
    lines.append(f"- Report root: `{summary['run_roots']['report_root']}`")
    lines.append("")


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


def _config_report_outputs(
    config: Mapping[str, Any],
    *,
    index_output_path: Path,
) -> dict[str, Path]:
    outputs = dict(DEFAULT_DOC_OUTPUTS)
    outputs["index"] = index_output_path
    report = config.get("report")
    if report is None:
        return outputs
    if not isinstance(report, Mapping):
        raise ValueError("report must be a TOML table.")
    raw_outputs = report.get("outputs")
    if raw_outputs is None:
        return outputs
    if not isinstance(raw_outputs, Mapping):
        raise ValueError("report.outputs must be a TOML table.")
    for key in REPORT_FILENAMES:
        value = raw_outputs.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"report.outputs.{key} must be a non-empty string.")
        outputs[key] = Path(value)
    return outputs


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


def _profile_commands(
    config: Mapping[str, Any],
    *,
    future_available_paths: set[Path] | None = None,
) -> list[dict[str, Any]]:
    future_available_paths = future_available_paths or set()
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
                    1
                    for path in resolved_outputs
                    if path.exists() or path in future_available_paths
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
