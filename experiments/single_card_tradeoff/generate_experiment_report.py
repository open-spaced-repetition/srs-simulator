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
    "experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml"
)
DEFAULT_REPORT_ROOT = Path("artifacts/single_card_tradeoff/reports/current")
DEFAULT_DOC_OUTPUT = Path("docs/single_card_tradeoff/experiments/2026-05-17-index.md")

REPORT_FILENAMES = {
    "index": "index.md",
    "default_no_sub05_tradeoff": "default_no_sub05_tradeoff.md",
    "first8_stationary_finite_distill": "first8_stationary_finite_distill.md",
    "first8_exact_vs_distill": "first8_exact_vs_distill.md",
    "first8_stationary_finite_r4d1_e512": ("first8_stationary_finite_r4d1_e512.md"),
    "low_param_direct_search": "low_param_direct_search.md",
    "uvfa_ppo": "uvfa_ppo.md",
    "recurrent_interval_ppo": "recurrent_interval_ppo.md",
    "oracle_distill": "oracle_distill.md",
    "grid_oracle": "grid_oracle.md",
    "infinite_stationary_oracles": "infinite_stationary_oracles.md",
    "ppo_guide_ablation": "ppo_guide_ablation.md",
    "stationary_finite_policy_viz": "stationary_finite_policy_viz.md",
    "stationary_finite_compression": "stationary_finite_compression.md",
    "stationary_finite_epoch_extension": "stationary_finite_epoch_extension.md",
    "oracle_policy_outputs": "oracle_policy_outputs.md",
    "interval_oracle_distill": "interval_oracle_distill.md",
    "retention_distill": "retention_distill.md",
    "multiuser_eval_batching": "multiuser_eval_batching.md",
    "oracle_stationary_finite_cpu_gpu_benchmark": (
        "oracle_stationary_finite_cpu_gpu_benchmark.md"
    ),
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
    "first8_stationary_finite_r4d1_e512": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-18-first8_stationary_finite_r4d1_e512.md"
    ),
    "low_param_direct_search": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-low_param_direct_search.md"
    ),
    "uvfa_ppo": Path("docs/single_card_tradeoff/experiments/2026-05-17-uvfa_ppo.md"),
    "recurrent_interval_ppo": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-recurrent_interval_ppo.md"
    ),
    "oracle_distill": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-oracle_distill.md"
    ),
    "grid_oracle": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-grid_oracle.md"
    ),
    "infinite_stationary_oracles": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-17-infinite_stationary_oracles.md"
    ),
    "ppo_guide_ablation": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-ppo_guide_ablation.md"
    ),
    "stationary_finite_policy_viz": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-17-stationary_finite_policy_viz.md"
    ),
    "stationary_finite_compression": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-17-stationary_finite_compression.md"
    ),
    "stationary_finite_epoch_extension": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-18-stationary_finite_epoch_extension.md"
    ),
    "oracle_policy_outputs": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-oracle_policy_outputs.md"
    ),
    "interval_oracle_distill": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-interval_oracle_distill.md"
    ),
    "retention_distill": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-retention_distill.md"
    ),
    "multiuser_eval_batching": Path(
        "docs/single_card_tradeoff/experiments/2026-05-17-multiuser_eval_batching.md"
    ),
    "oracle_stationary_finite_cpu_gpu_benchmark": Path(
        "docs/single_card_tradeoff/experiments/"
        "2026-05-18-oracle_stationary_finite_cpu_gpu_benchmark.md"
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
    "first8_r4d1_e512_distill_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_distill_first8_users_per_user_r4d1_e512_"
        "uniform_table_supervision_fsrs6_baseline_gpu/summary.csv"
    ),
    "first8_r4d1_e512_distill_train_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_distill_first8_users_per_user_r4d1_e512_"
        "uniform_table_supervision_fsrs6_baseline_gpu/train_summary.csv"
    ),
    "first8_r4d1_e512_distill_gpu_monitor_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_distill_first8_users_per_user_r4d1_e512_"
        "uniform_table_supervision_fsrs6_baseline_gpu/gpu_monitor/summary.json"
    ),
    "first8_r4d1_e512_exact_vs_distill_mean_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_exact_vs_distill_first8_users_r4d1_e512/"
        "mean_summary.csv"
    ),
    "first8_r4d1_e512_exact_vs_distill_regret_auc": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_exact_vs_distill_first8_users_r4d1_e512/"
        "regret_auc.csv"
    ),
    "first8_r4d1_e512_exact_vs_distill_gpu_monitor_summary": Path(
        "artifacts/single_card_tradeoff/"
        "stationary_finite_exact_vs_distill_first8_users_r4d1_e512/"
        "gpu_monitor/summary.json"
    ),
    "low_param_sparse_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "low_param_direct_policy_search_first8_users/metadata.json"
    ),
    "low_param_dense_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "low_param_direct_policy_search_first8_users_dense_weights/metadata.json"
    ),
    "oracle_distill_results": Path(
        "artifacts/single_card_tradeoff/fsrs6_oracle_distill_results.csv"
    ),
    "oracle_distill_hparam_summary": Path(
        "artifacts/single_card_tradeoff/fsrs6_oracle_distill_hparam_summary.csv"
    ),
    "stationary_finite_distill_results": Path(
        "artifacts/single_card_tradeoff/"
        "fsrs6_oracle_stationary_finite_distill_results.csv"
    ),
    "infinite_distill_results": Path(
        "artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_results.csv"
    ),
    "uvfa_ppo_results": Path("artifacts/single_card_tradeoff/uvfa_ppo_results.csv"),
    "uvfa_ppo_hparam_summary": Path(
        "artifacts/single_card_tradeoff/uvfa_ppo_hparam_search_summary.csv"
    ),
    "uvfa_ppo_rnn_interval_results": Path(
        "artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_results.csv"
    ),
    "grid_oracle_regret_auc": Path(
        "artifacts/single_card_tradeoff/results_regret_auc.csv"
    ),
    "stationary_finite_compare_regret_auc": Path(
        "artifacts/single_card_tradeoff/stationary_finite_compare/regret_auc.csv"
    ),
    "infinite_distill_compare_regret_auc": Path(
        "artifacts/single_card_tradeoff/infinite_distill_compare/regret_auc.csv"
    ),
    "ppo_ablation_summary": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "ppo_oracle_warmup_ablation_summary.csv"
    ),
    "ppo_no_guide_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_no_guide_regret_auc.csv"
    ),
    "ppo_static_guide_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_static_guide_regret_auc.csv"
    ),
    "ppo_oracle_warmup_only_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_oracle_warmup_only_regret_auc.csv"
    ),
    "rnn_no_guide_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_rnn_interval_no_guide_regret_auc.csv"
    ),
    "rnn_static_guide_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_rnn_interval_static_guide_regret_auc.csv"
    ),
    "rnn_oracle_warmup_only_regret_auc": Path(
        "artifacts/single_card_tradeoff/ppo_ablation/"
        "tradeoff_uvfa_ppo_rnn_interval_oracle_warmup_only_regret_auc.csv"
    ),
    "stationary_finite_policy_action_summary": Path(
        "artifacts/single_card_tradeoff/stationary_finite_policy_viz/action_summary.csv"
    ),
    "stationary_finite_policy_distill_comparison": Path(
        "artifacts/single_card_tradeoff/stationary_finite_policy_viz/"
        "distill_exact_comparison.csv"
    ),
    "stationary_finite_policy_findings": Path(
        "artifacts/single_card_tradeoff/stationary_finite_policy_viz/findings.md"
    ),
    "stationary_finite_cost_weight_ablation": Path(
        "artifacts/single_card_tradeoff/stationary_finite_cost_weight_ablation/"
        "cost_weight_ablation_multiseed_summary.csv"
    ),
    "stationary_finite_model_size_ablation": Path(
        "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/"
        "model_size_ablation_summary.csv"
    ),
    "stationary_finite_model_size_sub216_ablation": Path(
        "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/"
        "sub216_summary.csv"
    ),
    "stationary_finite_model_size_sub216_e256_ablation": Path(
        "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/"
        "sub216_e256_summary.csv"
    ),
    "stationary_finite_model_size_sub216_e512_ablation": Path(
        "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/"
        "sub216_e512_all_summary.csv"
    ),
    "stationary_finite_model_size_gpu_monitor_summary": Path(
        "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/"
        "gpu_monitor/summary.json"
    ),
    "oracle_policy_outputs_rollout": Path(
        "artifacts/single_card_tradeoff/analysis/"
        "no_sub05_fsrs6_oracle_policy_outputs_rollout_1825_p2048.csv"
    ),
    "oracle_policy_outputs_table": Path(
        "artifacts/single_card_tradeoff/analysis/"
        "no_sub05_fsrs6_oracle_policy_outputs_table_1825.csv"
    ),
    "oracle_policy_outputs_rollout_detail": Path(
        "artifacts/single_card_tradeoff/analysis/"
        "no_sub05_fsrs6_oracle_policy_outputs_rollout_1825_p2048_detail.csv"
    ),
    "oracle_policy_outputs_table_detail": Path(
        "artifacts/single_card_tradeoff/analysis/"
        "no_sub05_fsrs6_oracle_policy_outputs_table_1825_detail.csv"
    ),
    "interval_distill_results": Path(
        "artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill.csv"
    ),
    "interval_compare_regret_auc": Path(
        "artifacts/single_card_tradeoff/oracle_interval_compare/regret_auc.csv"
    ),
    "interval_hparam_summary": Path(
        "artifacts/single_card_tradeoff/"
        "fsrs6_oracle_interval_distill_hparam_summary.csv"
    ),
    "retention_distill_results": Path(
        "artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_results.csv"
    ),
    "multiuser_eval_batch_smoke_all_train": Path(
        "artifacts/single_card_tradeoff/eval_batch_smoke_all/train_summary.csv"
    ),
    "multiuser_eval_batch_smoke_group1_train": Path(
        "artifacts/single_card_tradeoff/eval_batch_smoke_group1/train_summary.csv"
    ),
    "multiuser_eval_batch_smoke_post_patch_train": Path(
        "artifacts/single_card_tradeoff/eval_batch_smoke_post_patch/train_summary.csv"
    ),
    "oracle_stationary_finite_cpu_gpu_summary": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_cpu_gpu_benchmark/summary.csv"
    ),
    "oracle_stationary_finite_cpu_gpu_runs": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_cpu_gpu_benchmark/runs.csv"
    ),
    "oracle_stationary_finite_cpu_gpu_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_cpu_gpu_benchmark/metadata.json"
    ),
    "oracle_stationary_finite_multiuser_cpu_gpu_summary": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_multiuser_cpu_gpu_benchmark/summary.csv"
    ),
    "oracle_stationary_finite_multiuser_cpu_gpu_runs": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_multiuser_cpu_gpu_benchmark/runs.csv"
    ),
    "oracle_stationary_finite_multiuser_cpu_gpu_metadata": Path(
        "artifacts/single_card_tradeoff/"
        "oracle_stationary_finite_multiuser_cpu_gpu_benchmark/metadata.json"
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
    config = _load_report_config_tree(args.config)
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
    first8_r4d1_summary = read_csv_rows(
        source_paths["first8_r4d1_e512_distill_summary"]
    )
    first8_r4d1_train = read_csv_rows(
        source_paths["first8_r4d1_e512_distill_train_summary"]
    )
    first8_r4d1_train_gpu = read_json_object(
        source_paths["first8_r4d1_e512_distill_gpu_monitor_summary"]
    )
    first8_r4d1_exact_vs_distill_mean = read_csv_rows(
        source_paths["first8_r4d1_e512_exact_vs_distill_mean_summary"]
    )
    first8_r4d1_exact_vs_distill_regret = read_csv_rows(
        source_paths["first8_r4d1_e512_exact_vs_distill_regret_auc"]
    )
    first8_r4d1_exact_vs_distill_gpu = read_json_object(
        source_paths["first8_r4d1_e512_exact_vs_distill_gpu_monitor_summary"]
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
            "included_profiles": [
                display_path(path)
                for path in report_config.get("_included_profile_paths", [])
            ],
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
        "first8_stationary_finite_r4d1_e512": _first8_r4d1_e512_summary(
            baseline_summary=first8_distill_summary,
            baseline_train=first8_distill_train,
            summary_rows=first8_r4d1_summary,
            train_rows=first8_r4d1_train,
            exact_vs_distill_mean=first8_r4d1_exact_vs_distill_mean,
            exact_vs_distill_regret=first8_r4d1_exact_vs_distill_regret,
            train_gpu_monitor=first8_r4d1_train_gpu,
            exact_gpu_monitor=first8_r4d1_exact_vs_distill_gpu,
        ),
        "low_param_direct_search": {
            "sparse_teacher_weights": _low_param_summary(low_param_sparse),
            "dense_teacher_weights": _low_param_summary(low_param_dense),
        },
        "configured_reports": _configured_reports(
            source_paths=source_paths,
            default_regret=default_regret,
        ),
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
    reports = {
        "index": render_index_report(summary),
        "default_no_sub05_tradeoff": render_default_no_sub05_tradeoff_report(summary),
        "first8_stationary_finite_distill": (
            render_first8_stationary_finite_distill_report(summary)
        ),
        "first8_exact_vs_distill": render_first8_exact_vs_distill_report(summary),
        "first8_stationary_finite_r4d1_e512": (
            render_first8_stationary_finite_r4d1_e512_report(summary)
        ),
        "low_param_direct_search": render_low_param_direct_search_report(summary),
    }
    for report in summary["configured_reports"]:
        reports[report["key"]] = render_configured_report(summary, report)
    return reports


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
    first8_r4d1 = summary["first8_stationary_finite_r4d1_e512"]
    rows = [
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
            "first-eight residual:4:1 e512",
            f"`{published['first8_stationary_finite_r4d1_e512']}`",
            (
                "The 132-parameter per-user student reaches "
                f"{format_percent(first8_r4d1['mean_relative_regret_auc_percent'])} "
                "relative regret at "
                f"{format_percent(first8_r4d1['mean_span_coverage_percent'])} "
                "coverage vs `fsrs6`."
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
    ]
    rows.extend(
        [
            [
                str(report["title"]),
                f"`{published[report['key']]}`",
                str(report["index_summary"]),
            ]
            for report in summary["configured_reports"]
        ]
    )
    lines.extend(
        markdown_table(
            ["experiment", "published report", "main result"],
            rows,
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


def render_configured_report(
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# {report['title']}")
    lines.append("")
    _append_report_preamble(lines, summary, question=str(report["question"]))
    lines.append("## Evidence")
    lines.append("")
    for paragraph in report["evidence"]:
        lines.append(str(paragraph))
        lines.append("")
    source_lines = _configured_source_lines(report)
    if source_lines:
        lines.extend(source_lines)
        lines.append("")
    if report.get("notes"):
        lines.append("Notes:")
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    result_paragraphs = report.get("result_paragraphs") or []
    tables = report.get("tables") or []
    if result_paragraphs or tables:
        lines.append("## Results")
        lines.append("")
        for paragraph in result_paragraphs:
            lines.append(str(paragraph))
            lines.append("")
        for table in tables:
            if table.get("title"):
                lines.append(f"### {table['title']}")
                lines.append("")
            lines.extend(markdown_table(table["headers"], table["rows"]))
            lines.append("")

    _append_reproduction_profile(
        lines,
        summary,
        command_names=report.get("command_names") or (),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(str(report["conclusion"]))
    lines.append("")
    _append_artifacts_footer(lines, summary, report_key=str(report["key"]))
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


def render_first8_stationary_finite_r4d1_e512_report(
    summary: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# First-eight Residual:4:1 Epoch-512 Validation")
    lines.append("")
    _append_report_preamble(
        lines,
        summary,
        question=(
            "Does the 132-parameter `residual:4:1` stationary finite student "
            "trained for 512 epochs transfer to the first eight FSRS-6 users?"
        ),
    )
    report = summary["first8_stationary_finite_r4d1_e512"]
    baseline = report["baseline_476"]
    exact = report["exact_vs_distill"]
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "Environment `fsrs6`, users 1 through 8, one independent student per "
        "user, uniform exact-table supervision, 512 epochs, and the same sparse "
        "teacher weights, clipped action grid, evaluation weights, and eval "
        "particles as the current 476-parameter first-eight baseline."
    )
    lines.append("")
    lines.extend(
        _artifact_lines(
            summary,
            keys=(
                "first8_r4d1_e512_distill_summary",
                "first8_r4d1_e512_distill_train_summary",
                "first8_r4d1_e512_distill_gpu_monitor_summary",
                "first8_r4d1_e512_exact_vs_distill_mean_summary",
                "first8_r4d1_e512_exact_vs_distill_regret_auc",
                "first8_r4d1_e512_exact_vs_distill_gpu_monitor_summary",
            ),
        )
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "The validation trains eight independent 132-parameter students "
        f"({format_int(report['ensemble_trainable_params'])} total trainable "
        "parameters during batched training)."
    )
    lines.append("")
    lines.append("### Mean vs fsrs6")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "model",
                "params/user",
                "epochs",
                "mean coverage",
                "mean time regret AUC",
                "mean relative regret",
                "mean agreement",
                "mean CE",
            ],
            [
                _first8_model_summary_row("residual:8:2", baseline),
                _first8_model_summary_row("residual:4:1", report),
                [
                    "delta r4d1 - r8d2",
                    _signed_int_delta(
                        report["params_per_user"] - baseline["params_per_user"]
                    ),
                    _signed_int_delta(report["epochs"] - baseline["epochs"]),
                    _signed_percent(report["mean_span_coverage_delta_vs_476"]),
                    _signed_float(report["mean_time_regret_delta_vs_476"], digits=4),
                    _signed_percent(report["mean_relative_regret_auc_delta_vs_476"]),
                    _signed_percent(
                        100.0
                        * report["mean_eval_teacher_action_agreement_delta_vs_476"]
                    ),
                    _signed_float(report["mean_final_ce_loss_delta_vs_476"], digits=5),
                ],
            ],
        )
    )
    lines.append("")
    lines.append("### Per-user deltas")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "user",
                "r4d1 coverage",
                "r4d1 relative regret",
                "coverage delta vs 476",
                "relative regret delta vs 476",
            ],
            [
                [
                    row["user"],
                    format_percent(row["span_coverage_percent"]),
                    format_percent(row["relative_regret_auc_percent"]),
                    _signed_percent(row["span_coverage_delta_vs_476"]),
                    _signed_percent(row["relative_regret_auc_delta_vs_476"]),
                ]
                for row in report["per_user_rows"]
            ],
        )
    )
    lines.append("")
    lines.append("### Exact-vs-distill check")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "scheduler",
                "mean coverage vs fsrs6",
                "mean time regret AUC vs fsrs6",
                "mean relative regret vs fsrs6",
            ],
            [
                [
                    row["scheduler"],
                    format_percent(row["mean_span_coverage_percent"]),
                    format_float(row["mean_time_regret_auc"], digits=4),
                    format_percent(row["mean_relative_regret_auc_percent"]),
                ]
                for row in exact["vs_fsrs6"]
            ],
        )
    )
    lines.append("")
    lines.append(
        "On the exact-teacher shared span, `residual:4:1` has "
        f"{format_float(exact['distill_vs_exact']['mean_time_regret_auc'], digits=4)} "
        "time regret AUC and "
        f"{format_percent(exact['distill_vs_exact']['mean_relative_regret_auc_percent'])} "
        "relative regret at "
        f"{format_percent(exact['distill_vs_exact']['mean_span_coverage_percent'])} "
        "coverage."
    )
    lines.append("")
    lines.append("### GPU monitor")
    lines.append("")
    lines.extend(
        markdown_table(
            ["stage", "shared spill", "peak shared memory", "peak FB memory"],
            [
                _gpu_monitor_row("train + fsrs6 eval", report["train_gpu_monitor"]),
                _gpu_monitor_row("exact-vs-distill eval", report["exact_gpu_monitor"]),
            ],
        )
    )
    lines.append("")
    _append_reproduction_profile(
        lines,
        summary,
        command_names=(
            "train_first8_stationary_finite_r4d1_e512",
            "evaluate_first8_exact_vs_distill_r4d1_e512",
        ),
    )
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "`residual:4:1` at 512 epochs does not validate as a drop-in "
        "replacement for the first-eight per-user default. It cuts parameters "
        f"from {format_int(baseline['params_per_user'])} to "
        f"{format_int(report['params_per_user'])} per user, but mean coverage "
        "changes by "
        f"{_signed_float(report['mean_span_coverage_delta_vs_476'])} points "
        "and mean relative regret changes by "
        f"{_signed_float(report['mean_relative_regret_auc_delta_vs_476'])} "
        "points versus the 476-parameter baseline. The largest coverage loss "
        "is user 5, where the change is -11.25 points. Treat the 132-parameter "
        "result as promising for the single default-user sweep, but not yet "
        "robust across users."
    )
    lines.append("")
    _append_artifacts_footer(
        lines,
        summary,
        report_key="first8_stationary_finite_r4d1_e512",
    )
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


def _configured_source_lines(report: Mapping[str, Any]) -> list[str]:
    sources = report.get("source_artifacts") or []
    if not sources:
        return []
    lines = ["Source artifacts:"]
    for source in sources:
        lines.append(f"- `{source['key']}`: `{source['path']}`")
    return lines


def _configured_reports(
    *,
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    oracle_distill_hparams = read_csv_rows(
        source_paths["oracle_distill_hparam_summary"]
    )
    uvfa_hparams = read_csv_rows(source_paths["uvfa_ppo_hparam_summary"])
    grid_regret = read_csv_rows(source_paths["grid_oracle_regret_auc"])
    stationary_compare = read_csv_rows(
        source_paths["stationary_finite_compare_regret_auc"]
    )
    infinite_compare = read_csv_rows(
        source_paths["infinite_distill_compare_regret_auc"]
    )
    ppo_summary = read_csv_rows(source_paths["ppo_ablation_summary"])
    action_summary = read_csv_rows(
        source_paths["stationary_finite_policy_action_summary"]
    )
    distill_policy_comparison = read_csv_rows(
        source_paths["stationary_finite_policy_distill_comparison"]
    )
    cost_weight_ablation = read_csv_rows(
        source_paths["stationary_finite_cost_weight_ablation"]
    )
    model_size_ablation = read_csv_rows(
        source_paths["stationary_finite_model_size_ablation"]
    )
    model_size_sub216_ablation = read_csv_rows(
        source_paths["stationary_finite_model_size_sub216_ablation"]
    )
    model_size_sub216_e256_ablation = read_csv_rows(
        source_paths["stationary_finite_model_size_sub216_e256_ablation"]
    )
    model_size_sub216_e512_ablation = read_csv_rows(
        source_paths["stationary_finite_model_size_sub216_e512_ablation"]
    )
    model_size_gpu_monitor_summary = read_json_object(
        source_paths["stationary_finite_model_size_gpu_monitor_summary"]
    )
    oracle_rollout_outputs = read_csv_rows(
        source_paths["oracle_policy_outputs_rollout"]
    )
    oracle_table_outputs = read_csv_rows(source_paths["oracle_policy_outputs_table"])
    interval_results = read_csv_rows(source_paths["interval_distill_results"])
    interval_regret = read_csv_rows(source_paths["interval_compare_regret_auc"])
    interval_hparams = read_csv_rows(source_paths["interval_hparam_summary"])
    retention_results = read_csv_rows(source_paths["retention_distill_results"])
    smoke_all = read_csv_rows(source_paths["multiuser_eval_batch_smoke_all_train"])
    smoke_group1 = read_csv_rows(
        source_paths["multiuser_eval_batch_smoke_group1_train"]
    )
    smoke_post_patch = read_csv_rows(
        source_paths["multiuser_eval_batch_smoke_post_patch_train"]
    )
    cpu_gpu_summary = read_csv_rows(
        source_paths["oracle_stationary_finite_cpu_gpu_summary"]
    )
    cpu_gpu_metadata = read_json_object(
        source_paths["oracle_stationary_finite_cpu_gpu_metadata"]
    )
    multiuser_cpu_gpu_summary = read_csv_rows(
        source_paths["oracle_stationary_finite_multiuser_cpu_gpu_summary"]
    )
    multiuser_cpu_gpu_metadata = read_json_object(
        source_paths["oracle_stationary_finite_multiuser_cpu_gpu_metadata"]
    )

    return [
        _uvfa_ppo_report(source_paths, default_regret, uvfa_hparams),
        _recurrent_interval_ppo_report(
            source_paths,
            default_regret,
            ppo_summary,
        ),
        _oracle_distill_report(source_paths, default_regret, oracle_distill_hparams),
        _grid_oracle_report(source_paths, grid_regret),
        _infinite_stationary_oracles_report(
            source_paths,
            stationary_compare,
            infinite_compare,
        ),
        _ppo_guide_ablation_report(source_paths, default_regret, ppo_summary),
        _stationary_finite_policy_viz_report(
            source_paths,
            action_summary,
            distill_policy_comparison,
        ),
        _stationary_finite_compression_report(
            source_paths,
            cost_weight_ablation,
            model_size_ablation,
            model_size_sub216_ablation,
        ),
        _stationary_finite_epoch_extension_report(
            source_paths,
            model_size_ablation,
            model_size_sub216_ablation,
            model_size_sub216_e256_ablation,
            model_size_sub216_e512_ablation,
            model_size_gpu_monitor_summary,
        ),
        _oracle_policy_outputs_report(
            source_paths,
            oracle_rollout_outputs,
            oracle_table_outputs,
        ),
        _interval_oracle_distill_report(
            source_paths,
            interval_results,
            interval_regret,
            interval_hparams,
        ),
        _retention_distill_report(source_paths, default_regret, retention_results),
        _multiuser_eval_batching_report(
            source_paths,
            smoke_all,
            smoke_group1,
            smoke_post_patch,
        ),
        _oracle_stationary_finite_cpu_gpu_benchmark_report(
            source_paths,
            cpu_gpu_summary,
            cpu_gpu_metadata,
            multiuser_cpu_gpu_summary,
            multiuser_cpu_gpu_metadata,
        ),
    ]


def _uvfa_ppo_report(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    hparams: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    current = _regret_summary(
        default_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="uvfa_ppo",
    )
    best = hparams[0]
    return {
        "key": "uvfa_ppo",
        "title": "UVFA PPO",
        "question": (
            "How does the goal-conditioned discrete-action PPO policy compare "
            "with the clipped `fsrs6_default` frontier, and how sensitive is it "
            "to model size?"
        ),
        "index_summary": (
            "`uvfa_ppo` reaches "
            f"{format_percent(current['relative_regret_auc_percent'])} relative "
            f"regret at {format_percent(current['span_coverage_percent'])} "
            "coverage, using 27,148 parameters."
        ),
        "evidence": [
            (
                "The deployment comparison uses the current no-sub-0.5 "
                "`fsrs6_default` tradeoff artifact. The model-scale table comes "
                "from the rerun hparam search CSV, not README prose."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "default_regret_auc",
            "uvfa_ppo_hparam_summary",
            "uvfa_ppo_results",
        ),
        "result_paragraphs": [
            (
                "The best hparam-search candidate by mean scalar objective is "
                f"`{best['candidate']}` with "
                f"{format_int(_int(best, 'param_count'))} parameters and mean "
                f"scalar objective {format_float(best['mean_scalar_objective'], digits=4)}."
            ),
        ],
        "tables": [
            {
                "title": "Current clipped tradeoff",
                "headers": [
                    "scheduler",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row(
                        "uvfa_ppo",
                        current,
                        params=PARAM_COUNTS["uvfa_ppo"],
                    )
                ],
            },
            {
                "title": "Model-scale search",
                "headers": [
                    "candidate",
                    "network",
                    "params",
                    "mean scalar",
                    "delta vs best",
                    "train_s",
                ],
                "rows": _hparam_rows(hparams, limit=6),
            },
        ],
        "command_names": (
            "evaluate_default_no_sub05_tradeoff",
            "search_uvfa_ppo_hparams",
        ),
        "conclusion": (
            "The default PPO checkpoint is competitive on coverage and AUC, "
            "but it is a much larger policy than the compact oracle-distill "
            "baselines."
        ),
    }


def _recurrent_interval_ppo_report(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    ppo_summary: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    current = _regret_summary(
        default_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="uvfa_ppo_rnn_interval",
    )
    default_train = _find_ppo_summary_row(
        ppo_summary,
        method="uvfa_ppo_rnn_interval",
        ablation="default_oracle_guide",
    )
    return {
        "key": "recurrent_interval_ppo",
        "title": "Recurrent Interval PPO",
        "question": (
            "Does the recurrent continuous-interval PPO policy improve the "
            "single-card memory-time frontier after the clipped action-space "
            "change?"
        ),
        "index_summary": (
            "`uvfa_ppo_rnn_interval` reaches "
            f"{format_percent(current['relative_regret_auc_percent'])} relative "
            f"regret at {format_percent(current['span_coverage_percent'])} "
            "coverage, but uses 87,559 parameters."
        ),
        "evidence": [
            (
                "The main row is from the current no-sub-0.5 default comparison. "
                "The guide-ablation rows are read from the structured PPO "
                "ablation CSVs."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "default_regret_auc",
            "uvfa_ppo_rnn_interval_results",
            "ppo_ablation_summary",
            "rnn_no_guide_regret_auc",
            "rnn_static_guide_regret_auc",
            "rnn_oracle_warmup_only_regret_auc",
        ),
        "result_paragraphs": [
            (
                "The default recurrent run used "
                f"{format_int(_int(default_train, 'train_transitions'))} training "
                f"transitions and took {format_float(default_train['train_runtime_s'])}s."
            ),
        ],
        "tables": [
            {
                "title": "Current clipped tradeoff",
                "headers": [
                    "scheduler",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row(
                        "uvfa_ppo_rnn_interval",
                        current,
                        params=PARAM_COUNTS["uvfa_ppo_rnn_interval"],
                    )
                ],
            },
            {
                "title": "Recurrent guide ablations",
                "headers": [
                    "variant",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": _ppo_ablation_rows(
                    source_paths,
                    default_regret,
                    specs=(
                        (
                            "oracle guide",
                            "uvfa_ppo_rnn_interval",
                            "default",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "no guide",
                            "uvfa_ppo_rnn_interval",
                            "rnn_no_guide_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "static guide",
                            "uvfa_ppo_rnn_interval",
                            "rnn_static_guide_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "oracle warmup only",
                            "uvfa_ppo_rnn_interval",
                            "rnn_oracle_warmup_only_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                    ),
                ),
            },
        ],
        "command_names": ("evaluate_default_no_sub05_tradeoff",),
        "conclusion": (
            "The recurrent interval policy is useful as a continuous-action "
            "baseline, but the current artifact does not dominate the compact "
            "stationary finite distill on coverage per parameter."
        ),
    }


def _oracle_distill_report(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    hparams: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    current = _regret_summary(
        default_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle_distill",
    )
    compact = next(row for row in hparams if row["candidate"] == "res16d2")
    return {
        "key": "oracle_distill",
        "title": "Discrete Oracle Distillation",
        "question": (
            "How strong is the unrestricted finite-horizon FSRS-6 oracle "
            "distillation baseline under the clipped action space?"
        ),
        "index_summary": (
            "`fsrs6_oracle_distill` is the strongest compact default baseline: "
            f"{format_percent(current['relative_regret_auc_percent'])} relative "
            f"regret at {format_percent(current['span_coverage_percent'])} coverage."
        ),
        "evidence": [
            (
                "The tradeoff row is the current clipped default comparison. "
                "The model-size facts come from the rerun oracle-distill "
                "hparam-search summary."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "default_regret_auc",
            "oracle_distill_results",
            "oracle_distill_hparam_summary",
        ),
        "result_paragraphs": [
            (
                "The current compact default is `res16d2` with "
                f"{format_int(_int(compact, 'param_count'))} parameters, final "
                f"CE {format_float(compact['train_final_loss'], digits=4)}, and "
                f"train teacher agreement "
                f"{format_percent(100.0 * _float(compact, 'train_final_action_agreement'))}."
            ),
        ],
        "tables": [
            {
                "title": "Current clipped tradeoff",
                "headers": [
                    "scheduler",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row(
                        "fsrs6_oracle_distill",
                        current,
                        params=PARAM_COUNTS["fsrs6_oracle_distill"],
                    )
                ],
            },
            {
                "title": "Model-scale search",
                "headers": [
                    "candidate",
                    "network",
                    "params",
                    "mean scalar",
                    "delta vs best",
                    "train_s",
                ],
                "rows": _hparam_rows(hparams, limit=8),
            },
        ],
        "command_names": (
            "evaluate_default_no_sub05_tradeoff",
            "search_oracle_distill_hparams",
        ),
        "conclusion": (
            "The unrestricted finite-oracle distill remains the compact baseline "
            "to beat on the default FSRS-6 single-card frontier."
        ),
    }


def _grid_oracle_report(
    source_paths: Mapping[str, Path],
    regret_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    finite = _regret_summary(
        regret_rows,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle",
    )
    stationary_finite = _regret_summary(
        regret_rows,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle_stationary_finite",
    )
    return {
        "key": "grid_oracle",
        "title": "Grid Oracle",
        "question": (
            "How do exact finite-horizon grid policies compare with the "
            "stationary finite policy class on the clipped default frontier?"
        ),
        "index_summary": (
            "Exact finite grid oracle reaches "
            f"{format_percent(finite['span_coverage_percent'])} coverage; "
            "stationary finite exact reaches "
            f"{format_percent(stationary_finite['span_coverage_percent'])}."
        ),
        "evidence": [
            (
                "This report uses the structured regret-AUC output from the "
                "exact oracle comparison run."
            ),
        ],
        "source_artifacts": _source_refs(source_paths, "grid_oracle_regret_auc"),
        "tables": [
            {
                "title": "Exact policy classes vs fsrs6_default",
                "headers": [
                    "scheduler",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row_no_params("fsrs6_oracle", finite),
                    _regret_table_row_no_params(
                        "fsrs6_oracle_stationary_finite",
                        stationary_finite,
                    ),
                ],
            },
        ],
        "command_names": ("evaluate_grid_oracle_compare",),
        "conclusion": (
            "The stationary finite exact policy gives up a small amount of "
            "time-regret performance and coverage versus the unrestricted "
            "finite-horizon table, but removes the remaining-time policy input."
        ),
    }


def _infinite_stationary_oracles_report(
    source_paths: Mapping[str, Path],
    stationary_compare: Sequence[Mapping[str, str]],
    infinite_compare: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    rows = [
        _regret_summary(
            stationary_compare,
            baseline_scheduler="fsrs6_default",
            scheduler=scheduler,
        )
        for scheduler in (
            "fsrs6_oracle_infinite",
            "fsrs6_oracle_infinite_distill",
            "fsrs6_oracle_stationary_finite",
        )
    ]
    infinite_direct = _regret_summary(
        infinite_compare,
        baseline_scheduler="fsrs6_oracle_distill",
        scheduler="fsrs6_oracle_infinite_distill",
    )
    return {
        "key": "infinite_stationary_oracles",
        "title": "Infinite And Stationary Oracles",
        "question": (
            "Does the average-reward infinite oracle transfer well to a finite "
            "new-card lifecycle?"
        ),
        "index_summary": (
            "Infinite distill has low default-baseline AUC but only "
            f"{format_percent(rows[1]['span_coverage_percent'])} coverage."
        ),
        "evidence": [
            (
                "The comparison uses finite-lifecycle tradeoff artifacts for "
                "the infinite exact/distill and stationary finite exact policies."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "stationary_finite_compare_regret_auc",
            "infinite_distill_compare_regret_auc",
            "infinite_distill_results",
        ),
        "result_paragraphs": [
            (
                "Directly against `fsrs6_oracle_distill`, infinite distill has "
                f"{format_percent(infinite_direct['relative_regret_auc_percent'])} "
                "relative regret over only "
                f"{format_percent(infinite_direct['span_coverage_percent'])} "
                "coverage."
            ),
        ],
        "tables": [
            {
                "title": "Finite-lifecycle evaluation",
                "headers": [
                    "scheduler",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row_no_params(scheduler, row)
                    for scheduler, row in zip(
                        (
                            "fsrs6_oracle_infinite",
                            "fsrs6_oracle_infinite_distill",
                            "fsrs6_oracle_stationary_finite",
                        ),
                        rows,
                        strict=True,
                    )
                ],
            }
        ],
        "command_names": ("evaluate_infinite_stationary_oracles",),
        "conclusion": (
            "The average-reward infinite objective is not aligned with the "
            "1825-day new-card lifecycle in the current evaluation; its useful "
            "frontier span is narrow after the action floor."
        ),
    }


def _ppo_guide_ablation_report(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    ppo_summary: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    default_discrete = _find_ppo_summary_row(
        ppo_summary,
        method="uvfa_ppo",
        ablation="default_oracle_guide",
    )
    default_recurrent = _find_ppo_summary_row(
        ppo_summary,
        method="uvfa_ppo_rnn_interval",
        ablation="default_oracle_guide",
    )
    return {
        "key": "ppo_guide_ablation",
        "title": "PPO Guide Ablation",
        "question": (
            "Which guide setup is responsible for the PPO policies' clipped "
            "single-card frontier performance?"
        ),
        "index_summary": (
            "The no-guide discrete PPO still beats `fsrs6_default`, while RNN "
            "interval PPO relies more heavily on the oracle guide."
        ),
        "evidence": [
            (
                "The table combines current no-sub-0.5 default rows for the "
                "oracle-guided defaults with the six structured no-sub-0.5 "
                "ablation regret CSVs."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "default_regret_auc",
            "ppo_ablation_summary",
            "ppo_no_guide_regret_auc",
            "ppo_static_guide_regret_auc",
            "ppo_oracle_warmup_only_regret_auc",
            "rnn_no_guide_regret_auc",
            "rnn_static_guide_regret_auc",
            "rnn_oracle_warmup_only_regret_auc",
        ),
        "result_paragraphs": [
            (
                "The default discrete PPO summary reports "
                f"{format_int(_int(default_discrete, 'train_transitions'))} "
                "training transitions; the default recurrent summary reports "
                f"{format_int(_int(default_recurrent, 'train_transitions'))}."
            ),
        ],
        "tables": [
            {
                "title": "Clipped regret-AUC ablations",
                "headers": [
                    "variant",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": _ppo_ablation_rows(
                    source_paths,
                    default_regret,
                    specs=(
                        ("ppo oracle guide", "uvfa_ppo", "default", "uvfa_ppo"),
                        (
                            "ppo no guide",
                            "uvfa_ppo",
                            "ppo_no_guide_regret_auc",
                            "uvfa_ppo",
                        ),
                        (
                            "ppo static guide",
                            "uvfa_ppo",
                            "ppo_static_guide_regret_auc",
                            "uvfa_ppo",
                        ),
                        (
                            "ppo oracle warmup only",
                            "uvfa_ppo",
                            "ppo_oracle_warmup_only_regret_auc",
                            "uvfa_ppo",
                        ),
                        (
                            "rnn oracle guide",
                            "uvfa_ppo_rnn_interval",
                            "default",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "rnn no guide",
                            "uvfa_ppo_rnn_interval",
                            "rnn_no_guide_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "rnn static guide",
                            "uvfa_ppo_rnn_interval",
                            "rnn_static_guide_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                        (
                            "rnn oracle warmup only",
                            "uvfa_ppo_rnn_interval",
                            "rnn_oracle_warmup_only_regret_auc",
                            "uvfa_ppo_rnn_interval",
                        ),
                    ),
                ),
            }
        ],
        "command_names": (),
        "conclusion": (
            "The oracle guide remains the safest PPO recipe in the current "
            "artifact set. Guide choice changes both regret and frontier span, "
            "so coverage must be reported with AUC."
        ),
    }


def _stationary_finite_policy_viz_report(
    source_paths: Mapping[str, Path],
    action_summary: Sequence[Mapping[str, str]],
    distill_comparison: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    action_rows = _stationary_action_summary_rows(action_summary)
    distill_rows = [
        [
            _weight_label(row),
            format_percent(100.0 * _float(row, "exact_match_cell_share")),
            format_percent(100.0 * _float(row, "distill_lower_cell_share")),
            format_percent(100.0 * _float(row, "distill_higher_cell_share")),
            format_float(row["mean_abs_retention_diff"], digits=4),
            format_float(row["mean_retention_diff"], digits=4),
        ]
        for row in distill_comparison
    ]
    return {
        "key": "stationary_finite_policy_viz",
        "title": "Stationary Finite Policy Visualization",
        "question": (
            "What action patterns does the exact stationary finite policy table "
            "learn after removing action retentions below 0.5?"
        ),
        "index_summary": (
            "The exact table shifts from modal 0.98 at `w=0` to modal 0.50 at "
            "`w=1024`, with mixed behavior at intermediate weights."
        ),
        "evidence": [
            (
                "The visualization report is generated from `action_summary.csv` "
                "and the exact-vs-distill table comparison CSV."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "stationary_finite_policy_action_summary",
            "stationary_finite_policy_distill_comparison",
            "stationary_finite_policy_findings",
        ),
        "tables": [
            {
                "title": "Exact table action summary",
                "headers": [
                    "weight",
                    "modal retention",
                    "modal share",
                    "mean action retention",
                    "normalized entropy",
                    "iterations",
                ],
                "rows": action_rows,
            },
            {
                "title": "Distill vs exact table",
                "headers": [
                    "weight",
                    "exact match",
                    "distill lower",
                    "distill higher",
                    "mean abs retention diff",
                    "mean retention diff",
                ],
                "rows": distill_rows,
            },
        ],
        "command_names": ("visualize_stationary_finite_policy",),
        "conclusion": (
            "The exact policy remains state-sensitive after clipping. The "
            "distill mostly tracks the table, but exact-cell agreement is "
            "lowest around the mixed `w=64` region."
        ),
    }


def _stationary_finite_model_size_table_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["variant"],
        row["arch_label"],
        format_int(_int(row, "parameter_count")),
        format_int(_int(row, "epochs")),
        format_percent(100.0 * _float(row, "eval_teacher_action_agreement")),
        _mean_std_percent(
            row,
            "relative_regret_auc_percent_mean",
            "relative_regret_auc_percent_std",
        ),
        _mean_std_percent(
            row,
            "span_coverage_percent_mean",
            "span_coverage_percent_std",
        ),
    ]


def _stationary_finite_epoch_extension_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    by_variant = {row["variant"]: row for row in rows}
    ordered_rows: list[Mapping[str, str]] = []
    for label in (
        "r5d1",
        "r4d1",
        "r3d1",
        "mlp8",
        "mlp6",
        "mlp4",
        "linear",
        "quadratic",
    ):
        for epochs in (128, 256, 512):
            row = by_variant.get(f"sf_train5_{label}_e{epochs}")
            if row is not None:
                ordered_rows.append(row)
    return ordered_rows


def _find_variant(
    rows: Sequence[Mapping[str, str]],
    variant: str,
) -> Mapping[str, str]:
    for row in rows:
        if row["variant"] == variant:
            return row
    raise ValueError(f"Missing summary row for {variant}.")


def _stationary_finite_compression_report(
    source_paths: Mapping[str, Path],
    cost_weight_rows: Sequence[Mapping[str, str]],
    model_size_rows: Sequence[Mapping[str, str]],
    model_size_sub216_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    practical_floor = next(
        row for row in model_size_rows if row["variant"] == "sf_train5_r8d1_e128"
    )
    best_sub216 = max(
        model_size_sub216_rows,
        key=lambda row: _float(row, "span_coverage_percent_mean"),
    )
    best_regret_sub216 = min(
        model_size_sub216_rows,
        key=lambda row: _float(row, "relative_regret_auc_percent_mean"),
    )
    residual_floor = next(
        row for row in model_size_rows if row["variant"] == "sf_train5_r6d1_e128"
    )
    return {
        "key": "stationary_finite_compression",
        "title": "Stationary Finite Compression",
        "question": (
            "How far can the stationary finite distill be compressed while "
            "preserving relative regret and span coverage?"
        ),
        "index_summary": (
            "The aligned 128-epoch rerun keeps a strong "
            f"{format_int(_int(practical_floor, 'parameter_count'))}-parameter "
            "student; the 216-parameter `residual:6:1` remains the smallest "
            "128-epoch row that keeps broad span coverage."
        ),
        "evidence": [
            (
                "The report uses the multi-seed sparse-cost-weight ablation and "
                "the aligned model-size ablation summaries from the current "
                "clipped action-space run."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "stationary_finite_cost_weight_ablation",
            "stationary_finite_model_size_ablation",
            "stationary_finite_model_size_sub216_ablation",
        ),
        "tables": [
            {
                "title": "Teacher cost-weight ablation",
                "headers": [
                    "variant",
                    "weights",
                    "params",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    [
                        row["variant"],
                        row["train_cost_weights"],
                        format_int(_int(row, "parameter_count")),
                        _mean_std_percent(
                            row,
                            "relative_regret_auc_percent_mean",
                            "relative_regret_auc_percent_std",
                        ),
                        _mean_std_percent(
                            row,
                            "span_coverage_percent_mean",
                            "span_coverage_percent_std",
                        ),
                    ]
                    for row in cost_weight_rows
                ],
            },
            {
                "title": "Model-size ablation",
                "headers": [
                    "variant",
                    "arch",
                    "params",
                    "epochs",
                    "agreement",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _stationary_finite_model_size_table_row(row)
                    for row in model_size_rows
                ],
            },
            {
                "title": "Sub-216 and structured sweep",
                "headers": [
                    "variant",
                    "family",
                    "arch",
                    "params",
                    "epochs",
                    "agreement",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    [
                        row["variant"],
                        row.get("family", ""),
                        row["arch_label"],
                        format_int(_int(row, "parameter_count")),
                        format_int(_int(row, "epochs")),
                        format_percent(
                            100.0 * _float(row, "eval_teacher_action_agreement")
                        ),
                        _mean_std_percent(
                            row,
                            "relative_regret_auc_percent_mean",
                            "relative_regret_auc_percent_std",
                        ),
                        _mean_std_percent(
                            row,
                            "span_coverage_percent_mean",
                            "span_coverage_percent_std",
                        ),
                    ]
                    for row in model_size_sub216_rows
                ],
            },
        ],
        "command_names": (
            "rerun_stationary_finite_model_size_ablation",
            "rerun_stationary_finite_sub216_model_size_ablation",
        ),
        "conclusion": (
            "After aligning epochs and evaluation seeds, the 316-parameter "
            "`residual:8:1` student recovers frontier span and remains competitive "
            "with larger students. The 216-parameter `residual:6:1` student also "
            "keeps span coverage, but with weaker relative regret in this rerun. "
            "In the quick sub-216 sweep, the best-regret row is "
            f"`{best_regret_sub216['arch_label']}` at "
            f"{format_int(_int(best_regret_sub216, 'parameter_count'))} parameters, "
            f"{_mean_std_percent(best_regret_sub216, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret, and "
            f"{_mean_std_percent(best_regret_sub216, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage. The best coverage row is "
            f"`{best_sub216['arch_label']}` at "
            f"{format_int(_int(best_sub216, 'parameter_count'))} parameters, "
            f"{_mean_std_percent(best_sub216, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret, and "
            f"{_mean_std_percent(best_sub216, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage. Both are weaker than "
            f"{_mean_std_percent(residual_floor, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret and "
            f"{_mean_std_percent(residual_floor, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage for `residual:6:1`. The separate stationary finite epoch "
            "extension report tests whether longer training changes the "
            "sub-216 conclusion."
        ),
    }


def _stationary_finite_epoch_extension_report(
    source_paths: Mapping[str, Path],
    model_size_rows: Sequence[Mapping[str, str]],
    model_size_sub216_rows: Sequence[Mapping[str, str]],
    model_size_sub216_e256_rows: Sequence[Mapping[str, str]],
    model_size_sub216_e512_rows: Sequence[Mapping[str, str]],
    gpu_monitor_summary: Mapping[str, Any],
) -> dict[str, Any]:
    residual_floor = _find_variant(model_size_rows, "sf_train5_r6d1_e128")
    r5d1_e256 = _find_variant(model_size_sub216_e256_rows, "sf_train5_r5d1_e256")
    r4d1_e512 = _find_variant(model_size_sub216_e512_rows, "sf_train5_r4d1_e512")
    mlp8_e512 = _find_variant(model_size_sub216_e512_rows, "sf_train5_mlp8_e512")
    quadratic_e512 = _find_variant(
        model_size_sub216_e512_rows,
        "sf_train5_quadratic_e512",
    )
    r3d1_e512 = _find_variant(model_size_sub216_e512_rows, "sf_train5_r3d1_e512")
    best_long_epoch = min(
        [*model_size_sub216_e256_rows, *model_size_sub216_e512_rows],
        key=lambda row: _float(row, "relative_regret_auc_percent_mean"),
    )
    epoch_extension_rows = _stationary_finite_epoch_extension_rows(
        [
            *model_size_sub216_rows,
            *model_size_sub216_e256_rows,
            *model_size_sub216_e512_rows,
        ]
    )
    return {
        "key": "stationary_finite_epoch_extension",
        "title": "Stationary Finite Epoch Extension",
        "question": (
            "Can additional distillation epochs recover the relative regret and "
            "span coverage of low-parameter stationary finite students?"
        ),
        "index_summary": (
            f"`residual:4:1` reaches {_mean_std_percent(r4d1_e512, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret and "
            f"{_mean_std_percent(r4d1_e512, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage at 512 epochs, but recovery is architecture-dependent."
        ),
        "evidence": [
            (
                "The experiment reruns representative sub-216 architectures for "
                "256 epochs and the full sub-216 set for 512 epochs."
            ),
            (
                "All non-epoch variables stay aligned with the current default "
                "stationary finite distill recipe: five sparse teacher weights, "
                "the clipped 11-action grid, `uniform_table` supervision, "
                "64 steps per epoch, 10,000 evaluation particles, and eval "
                "seeds `42,43,44`."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "stationary_finite_model_size_ablation",
            "stationary_finite_model_size_sub216_ablation",
            "stationary_finite_model_size_sub216_e256_ablation",
            "stationary_finite_model_size_sub216_e512_ablation",
            "stationary_finite_model_size_gpu_monitor_summary",
        ),
        "tables": [
            {
                "title": "Epoch-extension rows",
                "headers": [
                    "variant",
                    "arch",
                    "params",
                    "epochs",
                    "agreement",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _stationary_finite_model_size_table_row(row)
                    for row in epoch_extension_rows
                ],
            },
            {
                "title": "Key comparators",
                "headers": [
                    "variant",
                    "arch",
                    "params",
                    "epochs",
                    "agreement",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _stationary_finite_model_size_table_row(row)
                    for row in (
                        residual_floor,
                        r5d1_e256,
                        r4d1_e512,
                        mlp8_e512,
                        r3d1_e512,
                        quadratic_e512,
                    )
                ],
            },
            {
                "title": "GPU monitor",
                "headers": ["metric", "value"],
                "rows": [
                    [
                        "shared memory spill",
                        _optional_bool_label(
                            gpu_monitor_summary.get("shared_memory_spill_detected")
                        ),
                    ],
                    [
                        "peak shared memory",
                        f"{format_float(_mib(_optional_float(gpu_monitor_summary, 'shared_memory_peak_single_adapter_bytes')), digits=1)} MiB",
                    ],
                    [
                        "peak FB memory",
                        f"{format_float(gpu_monitor_summary.get('nvidia_smi_peak_memory_used_mib'), digits=1)} MiB",
                    ],
                ],
            },
        ],
        "command_names": (
            "rerun_stationary_finite_sub216_e256_epoch_extension",
            "rerun_stationary_finite_sub216_e512_epoch_extension",
        ),
        "conclusion": (
            "Increasing epochs can recover low-parameter regret and coverage, "
            "but not uniformly. `residual:5:1` reaches "
            f"{_mean_std_percent(r5d1_e256, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret and "
            f"{_mean_std_percent(r5d1_e256, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage at 256 epochs, then loses regret at 512 epochs. "
            "`residual:4:1` needs 512 epochs to reach "
            f"{_mean_std_percent(r4d1_e512, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret and "
            f"{_mean_std_percent(r4d1_e512, 'span_coverage_percent_mean', 'span_coverage_percent_std')} "
            "coverage, making it the smallest observed candidate that recovers "
            "both metrics in this single-train-seed sweep. The best long-epoch "
            "regret row is "
            f"`{best_long_epoch['arch_label']}` at "
            f"{format_int(_int(best_long_epoch, 'parameter_count'))} parameters, "
            f"{_mean_std_percent(best_long_epoch, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret, but `mlp:8` is only marginally smaller than the "
            "216-parameter `residual:6:1` baseline. Capacity still matters: "
            f"`residual:3:1` remains broken at {_mean_std_percent(r3d1_e512, 'relative_regret_auc_percent_mean', 'relative_regret_auc_percent_std')} "
            "relative regret, and `quadratic` loses coverage at 512 epochs. "
            "The 132-parameter row needs more train seeds and per-user "
            "validation before it should replace the 128-epoch defaults."
        ),
    }


def _oracle_policy_outputs_report(
    source_paths: Mapping[str, Path],
    rollout_outputs: Sequence[Mapping[str, str]],
    table_outputs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "key": "oracle_policy_outputs",
        "title": "Oracle Policy Outputs",
        "question": (
            "Which desired-retention actions does the finite-horizon oracle "
            "choose under rollout-weighted and table-weighted state sampling?"
        ),
        "index_summary": (
            "The rerun clipped output analysis shows rollout weighting favors "
            "high-retention actions at low cost and `0.50` only at the highest "
            "cost."
        ),
        "evidence": [
            (
                "The old output-distribution artifacts used actions below 0.5, "
                "so this report uses the rerun `no_sub05_*` structured CSVs."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "oracle_policy_outputs_rollout",
            "oracle_policy_outputs_table",
            "oracle_policy_outputs_rollout_detail",
            "oracle_policy_outputs_table_detail",
        ),
        "tables": [
            {
                "title": "Rollout-weighted modal actions",
                "headers": ["weight", "modal retention", "share", "decisions"],
                "rows": _policy_output_rows(rollout_outputs),
            },
            {
                "title": "Table-weighted modal actions",
                "headers": ["weight", "modal retention", "share", "decisions"],
                "rows": _policy_output_rows(table_outputs),
            },
        ],
        "command_names": (
            "analyze_oracle_policy_outputs_rollout",
            "analyze_oracle_policy_outputs_table",
        ),
        "conclusion": (
            "Rollout weighting and table weighting expose different parts of the "
            "finite oracle, but both show the cost-driven shift toward cheaper "
            "actions after the clipped action-space rerun."
        ),
    }


def _interval_oracle_distill_report(
    source_paths: Mapping[str, Path],
    interval_results: Sequence[Mapping[str, str]],
    interval_regret: Sequence[Mapping[str, str]],
    interval_hparams: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    exact = _regret_summary(
        interval_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle_interval",
    )
    distill = _regret_summary(
        interval_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle_interval_distill",
    )
    train_row = interval_results[0]
    return {
        "key": "interval_oracle_distill",
        "title": "Interval Oracle Distillation",
        "question": (
            "Can an integer-interval teacher and log-interval student improve "
            "the default FSRS-6 single-card frontier?"
        ),
        "index_summary": (
            "Interval distill reaches "
            f"{format_percent(distill['relative_regret_auc_percent'])} relative "
            f"regret at {format_percent(distill['span_coverage_percent'])} "
            "coverage in the rerun comparison."
        ),
        "evidence": [
            (
                "The interval distill training, tradeoff comparison, and "
                "model-size search were rerun to produce structured artifacts "
                "for this report."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "interval_distill_results",
            "interval_compare_regret_auc",
            "interval_hparam_summary",
        ),
        "result_paragraphs": [
            (
                "The rerun interval distill final loss is "
                f"{format_float(train_row['train_final_loss'], digits=5)}, with "
                "eval log-interval MAE "
                f"{format_float(train_row['eval_log_interval_mae'], digits=4)}."
            ),
        ],
        "tables": [
            {
                "title": "Interval policies vs fsrs6_default",
                "headers": [
                    "scheduler",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row_no_params("fsrs6_oracle_interval", exact),
                    _regret_table_row_no_params(
                        "fsrs6_oracle_interval_distill",
                        distill,
                    ),
                ],
            },
            {
                "title": "Interval model-scale search",
                "headers": [
                    "candidate",
                    "network",
                    "params",
                    "mean scalar",
                    "delta vs best",
                    "train_s",
                ],
                "rows": _hparam_rows(interval_hparams, limit=6),
            },
        ],
        "command_names": (
            "train_interval_distill",
            "evaluate_interval_oracle_compare",
            "search_interval_distill_hparams",
        ),
        "conclusion": (
            "The interval distill is strong in the default comparison, while the "
            "exact interval oracle row covers only the few cost weights solved "
            "in this comparison."
        ),
    }


def _retention_distill_report(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    retention_results: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    current = _regret_summary(
        default_regret,
        baseline_scheduler="fsrs6_default",
        scheduler="fsrs6_oracle_retention_distill",
    )
    train_row = retention_results[0]
    return {
        "key": "retention_distill",
        "title": "Continuous Desired-Retention Distillation",
        "question": (
            "How does the continuous desired-retention distill compare with "
            "the clipped discrete-oracle distill baselines?"
        ),
        "index_summary": (
            "`fsrs6_oracle_retention_distill` reaches "
            f"{format_percent(current['relative_regret_auc_percent'])} relative "
            f"regret at {format_percent(current['span_coverage_percent'])} coverage."
        ),
        "evidence": [
            (
                "The deployment row comes from the current no-sub-0.5 default "
                "comparison; training/evaluation loss fields come from the "
                "structured retention-distill results CSV."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "default_regret_auc",
            "retention_distill_results",
        ),
        "result_paragraphs": [
            (
                "The structured result reports final loss "
                f"{format_float(train_row['train_final_loss'], digits=4)}, "
                "eval retention MAE "
                f"{format_float(train_row['eval_retention_mae'], digits=4)}, and "
                "eval log-interval MAE "
                f"{format_float(train_row['eval_log_interval_mae'], digits=4)}."
            ),
        ],
        "tables": [
            {
                "title": "Current clipped tradeoff",
                "headers": [
                    "scheduler",
                    "params",
                    "time_regret_auc",
                    "relative_regret",
                    "coverage",
                ],
                "rows": [
                    _regret_table_row(
                        "fsrs6_oracle_retention_distill",
                        current,
                        params=PARAM_COUNTS["fsrs6_oracle_retention_distill"],
                    )
                ],
            }
        ],
        "command_names": ("evaluate_default_no_sub05_tradeoff",),
        "conclusion": (
            "The retention-output student is compact and viable, but in the "
            "current clipped comparison it trails the discrete oracle distill "
            "and stationary finite distill on coverage."
        ),
    }


def _multiuser_eval_batching_report(
    source_paths: Mapping[str, Path],
    smoke_all: Sequence[Mapping[str, str]],
    smoke_group1: Sequence[Mapping[str, str]],
    smoke_post_patch: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "key": "multiuser_eval_batching",
        "title": "Multiuser Evaluation Batching",
        "question": (
            "Does the multi-user stationary finite distill path have structured "
            "smoke evidence for single-process batched training/evaluation?"
        ),
        "index_summary": (
            "The available structured artifact is a tiny two-user smoke check; "
            "it validates the batched path but is not a full performance study."
        ),
        "evidence": [
            (
                "No full sequential-versus-batched benchmark artifact is "
                "available in the current tree. Per the report rule, this "
                "section is limited to the existing structured smoke CSVs and "
                "does not restate unstructured README timing claims."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "multiuser_eval_batch_smoke_all_train",
            "multiuser_eval_batch_smoke_group1_train",
            "multiuser_eval_batch_smoke_post_patch_train",
        ),
        "tables": [
            {
                "title": "Smoke runtime summaries",
                "headers": [
                    "artifact",
                    "users",
                    "teacher_s",
                    "train_s",
                    "eval_s",
                    "params/user",
                ],
                "rows": [
                    _smoke_runtime_row("all", smoke_all),
                    _smoke_runtime_row("group1", smoke_group1),
                    _smoke_runtime_row("post_patch", smoke_post_patch),
                ],
            }
        ],
        "command_names": (),
        "conclusion": (
            "The code path has structured smoke coverage. A full timing claim "
            "should be made only after writing a dedicated benchmark artifact."
        ),
    }


def _oracle_stationary_finite_cpu_gpu_benchmark_report(
    source_paths: Mapping[str, Path],
    summary_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
    multiuser_summary_rows: Sequence[Mapping[str, str]],
    multiuser_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    cpu = _device_summary(summary_rows, "cpu")
    cuda = _device_summary(summary_rows, "cuda")
    cuda_speedup = _optional_float(cuda, "cpu_relative_speedup")
    multiuser_cpu = _device_summary(multiuser_summary_rows, "cpu")
    multiuser_cuda = _device_summary(multiuser_summary_rows, "cuda")
    multiuser_cuda_speedup = _optional_float(multiuser_cuda, "cpu_relative_speedup")
    settings = metadata.get("settings")
    if not isinstance(settings, Mapping):
        settings = {}
    multiuser_settings = multiuser_metadata.get("settings")
    if not isinstance(multiuser_settings, Mapping):
        multiuser_settings = {}
    repeat_count = int(_optional_float(cpu, "repeat_count") or 0)
    multiuser_repeat_count = int(_optional_float(multiuser_cpu, "repeat_count") or 0)
    return {
        "key": "oracle_stationary_finite_cpu_gpu_benchmark",
        "title": "Oracle Stationary Finite CPU/GPU Benchmark",
        "question": (
            "How do CPU and CUDA runtimes compare for the default "
            "`oracle_stationary_finite_distill` workloads?"
        ),
        "index_summary": (
            f"Single-user: {_cpu_gpu_speed_fragment(cuda_speedup)}; "
            f"multi-user: {_cpu_gpu_speed_fragment(multiuser_cuda_speedup)}."
        ),
        "evidence": [
            (
                "The benchmark runs `oracle_stationary_finite_distill.py` with "
                "the formal default single-card workload: `fsrs6_default`, 1825 "
                "days, default stationary finite teacher weights, 128 epochs, "
                "64 steps per epoch, and 10,000 evaluation particles."
            ),
            (
                "The multi-user benchmark runs "
                "`oracle_stationary_finite_distill_multiuser.py --per-user-models` "
                "for the first eight benchmark users with the same default "
                "stationary finite distill recipe and button-usage costs."
            ),
            (
                "Each workload/device pair is run once, so the report gives "
                "observed timings without a variance estimate. CUDA memory and "
                "spill fields come from the benchmark GPU monitor artifacts."
            ),
        ],
        "source_artifacts": _source_refs(
            source_paths,
            "oracle_stationary_finite_cpu_gpu_summary",
            "oracle_stationary_finite_cpu_gpu_runs",
            "oracle_stationary_finite_cpu_gpu_metadata",
            "oracle_stationary_finite_multiuser_cpu_gpu_summary",
            "oracle_stationary_finite_multiuser_cpu_gpu_runs",
            "oracle_stationary_finite_multiuser_cpu_gpu_metadata",
        ),
        "notes": [
            (
                f"Configured repeats per device: {format_int(repeat_count)}; "
                f"CUDA available: {metadata.get('cuda_available')}."
            ),
            (
                "Benchmark settings: "
                f"epochs={settings.get('epochs')}, "
                f"steps_per_epoch={settings.get('steps_per_epoch')}, "
                f"eval_particles={settings.get('eval_particles')}."
            ),
            (
                "Multi-user benchmark settings: "
                f"users={multiuser_settings.get('user_ids')}, "
                f"repeats={format_int(multiuser_repeat_count)}, "
                f"eval_particles={multiuser_settings.get('eval_particles')}."
            ),
        ],
        "tables": [
            {
                "title": "Single-user device timing",
                "headers": [
                    "device",
                    "wall_s",
                    "speedup_vs_cpu",
                    "train_s",
                    "eval_s",
                    "repeats",
                ],
                "rows": [
                    _cpu_gpu_timing_row(cpu),
                    _cpu_gpu_timing_row(cuda),
                ],
            },
            {
                "title": "Single-user quality guard",
                "headers": [
                    "device",
                    "params",
                    "final CE",
                    "train agreement",
                    "eval agreement",
                ],
                "rows": [
                    _cpu_gpu_quality_row(cpu),
                    _cpu_gpu_quality_row(cuda),
                ],
            },
            {
                "title": "Single-user CUDA memory",
                "headers": [
                    "device",
                    "samples",
                    "dedicated MiB",
                    "shared peak MiB",
                    "spill",
                ],
                "rows": [_cpu_gpu_memory_row(cuda)],
            },
            {
                "title": "Multi-user device timing",
                "headers": [
                    "device",
                    "wall_s",
                    "speedup_vs_cpu",
                    "teacher_s",
                    "train_s",
                    "eval_s",
                ],
                "rows": [
                    _multiuser_cpu_gpu_timing_row(multiuser_cpu),
                    _multiuser_cpu_gpu_timing_row(multiuser_cuda),
                ],
            },
            {
                "title": "Multi-user quality guard",
                "headers": [
                    "device",
                    "users",
                    "params/user",
                    "ensemble params",
                    "mean CE",
                    "eval agreement",
                ],
                "rows": [
                    _multiuser_cpu_gpu_quality_row(multiuser_cpu),
                    _multiuser_cpu_gpu_quality_row(multiuser_cuda),
                ],
            },
            {
                "title": "Multi-user CUDA memory",
                "headers": [
                    "device",
                    "samples",
                    "dedicated MiB",
                    "shared peak MiB",
                    "spill",
                ],
                "rows": [_cpu_gpu_memory_row(multiuser_cuda)],
            },
        ],
        "command_names": (
            "benchmark_oracle_stationary_finite_cpu_gpu",
            "benchmark_oracle_stationary_finite_multiuser_cpu_gpu",
        ),
        "conclusion": (
            f"Single-user: {_cpu_gpu_speed_sentence(cuda_speedup)} "
            f"Multi-user: {_cpu_gpu_speed_sentence(multiuser_cuda_speedup)} "
            "Treat both as point estimates until the benchmark is rerun with "
            "multiple repeats."
        ),
    }


def _source_refs(source_paths: Mapping[str, Path], *keys: str) -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "path": display_path(source_paths[key]),
        }
        for key in keys
    ]


def _device_summary(
    rows: Sequence[Mapping[str, str]],
    device: str,
) -> Mapping[str, str]:
    for row in rows:
        if row["device"] == device:
            return row
    raise ValueError(f"Missing CPU/GPU benchmark summary row for {device}.")


def _cpu_gpu_timing_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["device"],
        format_float(row["wall_runtime_s_mean"], digits=2),
        f"{format_float(row['cpu_relative_speedup'], digits=2)}x",
        format_float(row["train_runtime_s_mean"], digits=2),
        format_float(row["eval_runtime_s_mean"], digits=2),
        format_int(_optional_float(row, "repeat_count")),
    ]


def _cpu_gpu_quality_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["device"],
        format_int(_optional_float(row, "parameter_count")),
        format_float(row["final_ce_loss_mean"], digits=5),
        format_percent(100.0 * _float(row, "train_teacher_action_agreement_mean")),
        format_percent(100.0 * _float(row, "eval_teacher_action_agreement_mean")),
    ]


def _multiuser_cpu_gpu_timing_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["device"],
        format_float(row["wall_runtime_s_mean"], digits=2),
        f"{format_float(row['cpu_relative_speedup'], digits=2)}x",
        format_float(row["teacher_runtime_s_mean"], digits=2),
        format_float(row["train_runtime_s_mean"], digits=2),
        format_float(row["eval_runtime_s_mean"], digits=2),
    ]


def _multiuser_cpu_gpu_quality_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["device"],
        format_int(_optional_float(row, "user_count")),
        format_int(_optional_float(row, "params_per_user")),
        format_int(_optional_float(row, "ensemble_trainable_params")),
        format_float(row["mean_final_ce_loss"], digits=5),
        format_percent(100.0 * _float(row, "mean_eval_teacher_action_agreement")),
    ]


def _cpu_gpu_memory_row(row: Mapping[str, str]) -> list[str]:
    return [
        row["device"],
        format_int(_optional_float(row, "gpu_monitor_sample_count_max")),
        format_float(
            _optional_float(row, "gpu_monitor_nvidia_smi_peak_memory_used_mib_max"),
            digits=1,
        ),
        format_float(
            _mib(
                _optional_float(
                    row,
                    "gpu_monitor_shared_memory_peak_single_adapter_bytes_max",
                )
            ),
            digits=1,
        ),
        _optional_bool_label(row.get("gpu_monitor_shared_memory_spill_detected")),
    ]


def _cpu_gpu_speed_sentence(speedup: float | None) -> str:
    if speedup is None:
        return "CPU/CUDA wall-clock speedup is unavailable for this artifact."
    if speedup >= 1.0:
        return (
            "CUDA completes the default stationary finite distill workload "
            f"{format_float(speedup, digits=2)}x faster than CPU in the single "
            "observed run."
        )
    return (
        "CUDA is slower than CPU for the default stationary finite distill "
        f"workload in the single observed run: CPU is "
        f"{format_float(1.0 / speedup, digits=2)}x faster."
    )


def _cpu_gpu_speed_fragment(speedup: float | None) -> str:
    if speedup is None:
        return "CPU/CUDA speedup unavailable"
    if speedup >= 1.0:
        return f"CUDA {format_float(speedup, digits=2)}x faster than CPU"
    return f"CPU {format_float(1.0 / speedup, digits=2)}x faster than CUDA"


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _mib(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1024.0 / 1024.0


def _optional_bool_label(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return "yes" if str(value).lower() == "true" else "no"


def _signed_float(value: float, *, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _signed_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _signed_int_delta(value: int) -> str:
    return f"{value:+,}"


def _first8_model_summary_row(label: str, row: Mapping[str, Any]) -> list[str]:
    return [
        label,
        format_int(row["params_per_user"]),
        format_int(row["epochs"]),
        format_percent(row["mean_span_coverage_percent"]),
        format_float(row["mean_time_regret_auc"], digits=4),
        format_percent(row["mean_relative_regret_auc_percent"]),
        format_percent(100.0 * float(row["mean_eval_teacher_action_agreement"])),
        format_float(row["mean_final_ce_loss"], digits=5),
    ]


def _gpu_monitor_row(label: str, row: Mapping[str, Any]) -> list[str]:
    return [
        label,
        _optional_bool_label(row.get("shared_memory_spill_detected")),
        f"{format_float(_mib(_optional_float(row, 'shared_memory_peak_single_adapter_bytes')), digits=1)} MiB",
        f"{format_float(row.get('nvidia_smi_peak_memory_used_mib'), digits=1)} MiB",
    ]


def _hparam_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    limit: int,
) -> list[list[str]]:
    return [
        [
            row["candidate"],
            f"{row['network']}:{row['hidden_size']}:{row['network_depth']}",
            format_int(_int(row, "param_count")),
            format_float(row["mean_scalar_objective"], digits=4),
            format_float(row["mean_delta_vs_best_candidate"], digits=4),
            format_float(row["train_runtime_s"], digits=2),
        ]
        for row in rows[:limit]
    ]


def _regret_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    baseline_scheduler: str,
    scheduler: str,
    environment: str = "fsrs6_default",
) -> dict[str, float]:
    return _regret_metrics(
        _find_regret_row(
            rows,
            environment=environment,
            baseline_scheduler=baseline_scheduler,
            scheduler=scheduler,
        )
    )


def _regret_table_row(
    scheduler: str,
    metrics: Mapping[str, Any],
    *,
    params: int | None,
) -> list[str]:
    return [
        scheduler,
        format_int(params),
        format_float(metrics["time_regret_auc"], digits=4),
        format_percent(metrics["relative_regret_auc_percent"]),
        format_percent(metrics["span_coverage_percent"]),
    ]


def _regret_table_row_no_params(
    scheduler: str,
    metrics: Mapping[str, Any],
) -> list[str]:
    return [
        scheduler,
        format_float(metrics["time_regret_auc"], digits=4),
        format_percent(metrics["relative_regret_auc_percent"]),
        format_percent(metrics["span_coverage_percent"]),
    ]


def _find_ppo_summary_row(
    rows: Sequence[Mapping[str, str]],
    *,
    method: str,
    ablation: str,
) -> Mapping[str, str]:
    for row in rows:
        if row["method"] == method and row["ablation"] == ablation:
            return row
    raise ValueError(f"Missing PPO summary row for {method}/{ablation}.")


def _ppo_ablation_rows(
    source_paths: Mapping[str, Path],
    default_regret: Sequence[Mapping[str, str]],
    *,
    specs: Sequence[tuple[str, str, str, str]],
) -> list[list[str]]:
    result: list[list[str]] = []
    for label, method, source_key, scheduler in specs:
        if source_key == "default":
            metrics = _regret_summary(
                default_regret,
                baseline_scheduler="fsrs6_default",
                scheduler=scheduler,
            )
        else:
            metrics = _regret_summary(
                read_csv_rows(source_paths[source_key]),
                baseline_scheduler="fsrs6_default",
                scheduler=scheduler,
            )
        result.append(
            [
                label,
                format_int(PARAM_COUNTS[method]),
                format_float(metrics["time_regret_auc"], digits=4),
                format_percent(metrics["relative_regret_auc_percent"]),
                format_percent(metrics["span_coverage_percent"]),
            ]
        )
    return result


def _stationary_action_summary_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[list[str]]:
    by_weight: dict[float, Mapping[str, str]] = {}
    for row in rows:
        weight = _float(row, "goal_cost_weight")
        by_weight.setdefault(weight, row)
    return [
        [
            _weight_label(row),
            format_float(row["modal_action_retention"], digits=2),
            format_percent(100.0 * _float(row, "modal_cell_share")),
            format_float(row["mean_action_retention"], digits=4),
            format_float(row["normalized_action_entropy"], digits=3),
            format_int(_int(row, "policy_iterations")),
        ]
        for _, row in sorted(by_weight.items())
    ]


def _policy_output_rows(rows: Sequence[Mapping[str, str]]) -> list[list[str]]:
    by_weight: dict[float, list[Mapping[str, str]]] = {}
    for row in rows:
        by_weight.setdefault(_float(row, "goal_cost_weight"), []).append(row)
    result: list[list[str]] = []
    for _, group in sorted(by_weight.items()):
        top = max(group, key=lambda row: _float(row, "decision_share"))
        result.append(
            [
                _weight_label(top),
                format_float(top["action_retention"], digits=2),
                format_percent(100.0 * _float(top, "decision_share")),
                format_int(_int(top, "total_decisions")),
            ]
        )
    return result


def _smoke_runtime_row(
    label: str,
    rows: Sequence[Mapping[str, str]],
) -> list[str]:
    row = rows[0]
    return [
        label,
        format_int(len(rows)),
        format_float(row["teacher_runtime_s"], digits=3),
        format_float(row["train_runtime_s"], digits=3),
        format_float(row["eval_runtime_s"], digits=3),
        format_int(_int(row, "params_per_user")),
    ]


def _mean_std_percent(
    row: Mapping[str, Any],
    mean_key: str,
    std_key: str,
) -> str:
    mean = _float(row, mean_key)
    std = _float(row, std_key)
    if std == 0.0:
        return format_percent(mean)
    return f"{format_percent(mean)} +/- {format_percent(std)}"


def _weight_label(row: Mapping[str, Any]) -> str:
    weight = _float(row, "goal_cost_weight")
    if weight.is_integer():
        return format_int(int(weight))
    return format_float(weight, digits=2)


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
    paths = dict(DEFAULT_SOURCE_PATHS)
    for key, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source_artifacts.{key} must be a non-empty string.")
        paths[key] = Path(value)
    return paths


def _load_report_config_tree(path: Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    config = load_toml_profile(resolved)
    return _expand_report_config(config, config_path=resolved, seen={resolved})


def _expand_report_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    seen: set[Path],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(config)
    included_profiles: list[Path] = []
    for profile_path in _child_profile_paths(config, config_path=config_path):
        resolved = resolve_repo_path(profile_path)
        if resolved in seen:
            raise ValueError(
                f"Recursive report profile include: {display_path(resolved)}"
            )
        child = load_toml_profile(resolved)
        expanded_child = _expand_report_config(
            child,
            config_path=resolved,
            seen={*seen, resolved},
        )
        included_profiles.append(resolved)
        included_profiles.extend(expanded_child.get("_included_profile_paths", []))
        _merge_report_profile(merged, expanded_child)
    if included_profiles:
        merged["_included_profile_paths"] = included_profiles
    return merged


def _child_profile_paths(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> list[Path]:
    report = config.get("report")
    if report is None:
        return []
    if not isinstance(report, Mapping):
        raise ValueError("report must be a TOML table.")
    raw = report.get("profile_paths", [])
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("report.profile_paths must be a string array.")
    base = config_path.parent
    paths: list[Path] = []
    for item in raw:
        child_path = Path(item)
        if not child_path.is_absolute():
            child_path = base / child_path
        paths.append(child_path)
    return paths


def _merge_report_profile(
    merged: dict[str, Any],
    child: Mapping[str, Any],
) -> None:
    child_report = child.get("report")
    if isinstance(child_report, Mapping):
        merged_report = dict(merged.get("report") or {})
        for key, value in child_report.items():
            if key == "outputs":
                if not isinstance(value, Mapping):
                    raise ValueError("report.outputs must be a TOML table.")
                outputs = dict(merged_report.get("outputs") or {})
                outputs.update(value)
                merged_report["outputs"] = outputs
            elif key == "profile_paths":
                continue
            elif key not in merged_report:
                merged_report[key] = value
        merged["report"] = merged_report

    child_sources = child.get("source_artifacts")
    if child_sources is not None:
        if not isinstance(child_sources, Mapping):
            raise ValueError("source_artifacts must be a TOML table.")
        sources = dict(merged.get("source_artifacts") or {})
        sources.update(child_sources)
        merged["source_artifacts"] = sources

    child_commands = child.get("commands", [])
    if child_commands:
        if not isinstance(child_commands, list):
            raise ValueError("commands must be an array of TOML tables.")
        commands = list(merged.get("commands") or [])
        commands.extend(child_commands)
        merged["commands"] = commands


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


def _first8_r4d1_e512_summary(
    *,
    baseline_summary: Sequence[Mapping[str, str]],
    baseline_train: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
    train_rows: Sequence[Mapping[str, str]],
    exact_vs_distill_mean: Sequence[Mapping[str, str]],
    exact_vs_distill_regret: Sequence[Mapping[str, str]],
    train_gpu_monitor: Mapping[str, Any],
    exact_gpu_monitor: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _first8_distill_summary(baseline_summary, baseline_train)
    current = _first8_distill_summary(summary_rows, train_rows)
    baseline_by_env = {row["environment"]: row for row in baseline_summary}
    per_user_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        baseline_row = baseline_by_env[row["environment"]]
        per_user_rows.append(
            {
                "user": row["environment"].replace("fsrs6_user_", ""),
                "span_coverage_percent": _float(row, "span_coverage_percent"),
                "time_regret_auc": _float(row, "time_regret_auc"),
                "relative_regret_auc_percent": _float(
                    row,
                    "relative_regret_auc_percent",
                ),
                "span_coverage_delta_vs_476": _float(
                    row,
                    "span_coverage_percent",
                )
                - _float(baseline_row, "span_coverage_percent"),
                "time_regret_delta_vs_476": _float(row, "time_regret_auc")
                - _float(baseline_row, "time_regret_auc"),
                "relative_regret_auc_delta_vs_476": _float(
                    row,
                    "relative_regret_auc_percent",
                )
                - _float(baseline_row, "relative_regret_auc_percent"),
            }
        )

    current.update(
        {
            "baseline_476": baseline,
            "epochs": _int(train_rows[0], "epochs"),
            "mean_span_coverage_delta_vs_476": (
                current["mean_span_coverage_percent"]
                - baseline["mean_span_coverage_percent"]
            ),
            "mean_time_regret_delta_vs_476": (
                current["mean_time_regret_auc"] - baseline["mean_time_regret_auc"]
            ),
            "mean_relative_regret_auc_delta_vs_476": (
                current["mean_relative_regret_auc_percent"]
                - baseline["mean_relative_regret_auc_percent"]
            ),
            "mean_final_ce_loss_delta_vs_476": (
                current["mean_final_ce_loss"] - baseline["mean_final_ce_loss"]
            ),
            "mean_eval_teacher_action_agreement_delta_vs_476": (
                current["mean_eval_teacher_action_agreement"]
                - baseline["mean_eval_teacher_action_agreement"]
            ),
            "per_user_rows": per_user_rows,
            "exact_vs_distill": _first8_exact_vs_distill_summary(
                exact_vs_distill_mean,
                exact_vs_distill_regret,
            ),
            "train_gpu_monitor": dict(train_gpu_monitor),
            "exact_gpu_monitor": dict(exact_gpu_monitor),
        }
    )
    baseline["epochs"] = _int(baseline_train[0], "epochs")
    return current


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
