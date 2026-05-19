from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


USER_RE = re.compile(r"user_(\d+)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a formal RL-scheduler experiment report from JSON artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-run-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--comparison-label", default="Comparison")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    summary_path, report_path, summary = generate_report(
        run_root=args.run_root,
        comparison_run_root=args.comparison_run_root,
        output_path=args.output_path,
        title=args.title,
        question=args.question,
        candidate_label=args.candidate_label,
        comparison_label=args.comparison_label,
    )
    print(
        json.dumps(
            {
                "report_summary_path": str(summary_path),
                "run_report_path": str(report_path),
                "output_path": str(args.output_path),
                "passed": True,
                "title": summary["title"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def generate_report(
    *,
    run_root: Path,
    comparison_run_root: Path,
    output_path: Path,
    title: str | None = None,
    question: str | None = None,
    candidate_label: str = "Candidate",
    comparison_label: str = "Comparison",
) -> tuple[Path, Path, dict[str, Any]]:
    run_root = run_root.resolve()
    comparison_run_root = comparison_run_root.resolve()
    output_path = output_path.resolve()
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_summary_path = report_dir / "report_summary.json"
    run_report_path = report_dir / "report.md"

    summary = build_report_summary(
        run_root=run_root,
        comparison_run_root=comparison_run_root,
        report_summary_path=report_summary_path,
        output_path=output_path,
        title=title,
        question=question,
        candidate_label=candidate_label,
        comparison_label=comparison_label,
    )
    report_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_report_from_summary(
        report_summary_path, _read_json(report_summary_path)
    )
    run_report_path.write_text(markdown, encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return report_summary_path, run_report_path, summary


def build_report_summary(
    *,
    run_root: Path,
    comparison_run_root: Path,
    report_summary_path: Path,
    output_path: Path,
    title: str | None,
    question: str | None,
    candidate_label: str,
    comparison_label: str,
) -> dict[str, Any]:
    candidate_analysis_path = _analysis_summary_path(run_root)
    comparison_analysis_path = _analysis_summary_path(comparison_run_root)
    candidate_analysis = _read_json(candidate_analysis_path)
    comparison_analysis = _read_json(comparison_analysis_path)
    candidate_scheduler = _primary_scheduler(candidate_analysis)
    comparison_scheduler = _primary_scheduler(comparison_analysis)
    if title is None:
        title = f"{candidate_label} vs {comparison_label} experiment report"

    return {
        "type": "rl-scheduler-experiment-report",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "title": title,
        "question": question,
        "labels": {
            "candidate": candidate_label,
            "comparison": comparison_label,
        },
        "run_roots": {
            "candidate": _display_path(run_root),
            "comparison": _display_path(comparison_run_root),
        },
        "output_paths": {
            "report_summary": _display_path(report_summary_path),
            "run_report": _display_path(report_summary_path.parent / "report.md"),
            "requested_output": _display_path(output_path),
        },
        "source_summary_paths": {
            "candidate_analysis_summary": _display_path(candidate_analysis_path),
            "comparison_analysis_summary": _display_path(comparison_analysis_path),
        },
        "run_metadata": {
            "candidate": _run_metadata(
                run_root=run_root,
                analysis=candidate_analysis,
                scheduler=candidate_scheduler,
            ),
            "comparison": _run_metadata(
                run_root=comparison_run_root,
                analysis=comparison_analysis,
                scheduler=comparison_scheduler,
            ),
        },
        "provenance": {
            "candidate": _provenance(run_root),
            "comparison": _provenance(comparison_run_root),
        },
        "stages": {
            "candidate": _stage_summaries(run_root),
            "comparison": _stage_summaries(comparison_run_root),
        },
        "performance": {
            "candidate": _performance_summaries(run_root),
            "comparison": _performance_summaries(comparison_run_root),
        },
        "gpu_monitor": {
            "candidate": _gpu_monitor_summaries(run_root),
            "comparison": _gpu_monitor_summaries(comparison_run_root),
        },
        "training_hv_gains": {
            "candidate": _training_hv_gains(run_root),
            "comparison": _training_hv_gains(comparison_run_root),
        },
        "diagnostics": {
            "policy_point_diagnostics": _policy_point_diagnostics(
                candidate_analysis=candidate_analysis,
                comparison_analysis=comparison_analysis,
                candidate_scheduler=candidate_scheduler,
                comparison_scheduler=comparison_scheduler,
            )
        },
        "external_pareto": {
            "candidate_scheduler": candidate_scheduler,
            "comparison_scheduler": comparison_scheduler,
            "environments": _external_pareto_environments(
                candidate_analysis=candidate_analysis,
                comparison_analysis=comparison_analysis,
                candidate_scheduler=candidate_scheduler,
                comparison_scheduler=comparison_scheduler,
            ),
        },
        "per_user_hv_deltas": {
            "environments": _per_user_hv_environments(
                candidate_analysis=candidate_analysis,
                comparison_analysis=comparison_analysis,
                candidate_scheduler=candidate_scheduler,
                comparison_scheduler=comparison_scheduler,
            )
        },
    }


def render_report_from_summary(
    report_summary_path: Path, summary: dict[str, Any]
) -> str:
    lines: list[str] = []
    lines.append(f"# {summary['title']}")
    lines.append("")
    lines.append(f"Machine summary: `{summary['output_paths']['report_summary']}`")
    lines.append("")
    if summary.get("question"):
        lines.append("## Question")
        lines.append("")
        lines.append(str(summary["question"]))
        lines.append("")

    lines.append("## Runs")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "run",
                "scheduler",
                "config",
                "portfolio budget",
                "baseline DR manifest",
            ],
            _run_rows(summary),
        )
    )
    lines.append("")
    lines.append("Analysis summaries:")
    lines.append("")
    lines.append(
        f"- {summary['labels']['candidate']}: "
        f"`{summary['source_summary_paths']['candidate_analysis_summary']}`"
    )
    lines.append(
        f"- {summary['labels']['comparison']}: "
        f"`{summary['source_summary_paths']['comparison_analysis_summary']}`"
    )
    lines.append("")

    lines.append("## Stage Status")
    lines.append("")
    lines.append(
        markdown_table(
            ["run", "stage", "passed", "failures"],
            _stage_status_rows(summary, "candidate")
            + _stage_status_rows(summary, "comparison"),
        )
    )
    lines.append("")

    lines.append("## Performance")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "run",
                "stage",
                "device",
                "elapsed seconds",
                "user-days/s",
                "candidate-days/s",
            ],
            _performance_rows(summary, "candidate")
            + _performance_rows(summary, "comparison"),
        )
    )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(
        markdown_table(
            ["run", "git commit", "dirty", "Python", "PyTorch", "CUDA", "device"],
            _provenance_rows(summary),
        )
    )
    lines.append("")

    lines.append("## GPU Monitor")
    lines.append("")
    if (
        not summary["gpu_monitor"]["candidate"]
        and not summary["gpu_monitor"]["comparison"]
    ):
        lines.append(
            "No automatic GPU monitor artifacts are present for these historical "
            "runs. The table below reports machine-readable Torch CUDA peak "
            "memory from `performance_summary.json`; shared-memory spill cannot "
            "be judged from these artifacts."
        )
        lines.append("")
        lines.append(
            markdown_table(
                [
                    "run",
                    "stage",
                    "device",
                    "peak allocated MiB",
                    "peak reserved MiB",
                    "shared-memory spill",
                ],
                _legacy_gpu_rows(summary, "candidate")
                + _legacy_gpu_rows(summary, "comparison"),
            )
        )
        lines.append("")
    else:
        lines.append(
            markdown_table(
                [
                    "run",
                    "stage",
                    "summary",
                    "shared peak MiB",
                    "summed peak MiB",
                    "spill",
                    "nvidia-smi peak MiB",
                ],
                _gpu_monitor_rows(summary, "candidate")
                + _gpu_monitor_rows(summary, "comparison"),
            )
        )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.extend(_conclusion_lines(summary))
    lines.append("")

    lines.append("## External Pareto Results")
    lines.append("")
    lines.append(
        "Scheduler-only hypervolume values are sums of per-user HV delta against "
        "the same staged FSRS6 baseline manifest. Positive HV delta and "
        "same-budget memory lift are better. Positive same-target time saved is better. "
        "The two baseline-relative AUC columns use user-simple averages from "
        "the analysis summary."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "environment",
                "scheduler",
                "HV delta sum",
                "HV delta / baseline HV",
                "frontier points",
                "same-budget memory lift AUC",
                "same-budget memory lift / baseline",
                "budget coverage",
                "same-target time saved AUC",
                "same-target time saved / baseline",
                "target coverage",
            ],
            _external_pareto_rows(summary),
        )
    )
    lines.append("")

    lines.append("## Per-User HV Delta")
    lines.append("")
    lines.append(_per_user_hv_sentence(summary))
    lines.append("")
    lines.append(markdown_table(_per_user_headers(summary), _per_user_rows(summary)))
    lines.append("")

    lines.append("## Diagnostics")
    lines.append("")
    lines.append(
        "Unweighted policy-point averages describe where sampled policies lie; "
        "they are diagnostics only and do not replace external Pareto evidence."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                "environment",
                "scheduler",
                "policy-point avg memorized",
                "policy-point avg time",
                "policy-point avg efficiency",
                "policy-point avg reviews",
            ],
            _policy_point_rows(summary),
        )
    )
    lines.append("")

    lines.append("Train-overfit final HV gain by user:")
    lines.append("")
    lines.append(
        markdown_table(
            ["run", "user", "final training HV gain"],
            _training_hv_user_rows(summary, "candidate")
            + _training_hv_user_rows(summary, "comparison"),
        )
    )
    lines.append("")

    lines.append("## Training HV")
    lines.append("")
    lines.append(
        markdown_table(
            ["run", "users", "final training HV gain sum"],
            [
                [
                    summary["labels"]["candidate"],
                    str(summary["training_hv_gains"]["candidate"]["users"]),
                    fmt_number(summary["training_hv_gains"]["candidate"]["sum"]),
                ],
                [
                    summary["labels"]["comparison"],
                    str(summary["training_hv_gains"]["comparison"]["users"]),
                    fmt_number(summary["training_hv_gains"]["comparison"]["sum"]),
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## Artifact Paths")
    lines.append("")
    lines.append(f"- Report summary: `{summary['output_paths']['report_summary']}`")
    lines.append(f"- Run-local report: `{summary['output_paths']['run_report']}`")
    lines.append(f"- Published report: `{summary['output_paths']['requested_output']}`")
    lines.append("")
    return "\n".join(lines)


def _analysis_summary_path(run_root: Path) -> Path:
    path = (
        run_root / "analyze-pareto" / "analyze_pareto_outputs" / "analysis_summary.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing machine-readable analyze-pareto summary: {path}"
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return loaded


def _primary_scheduler(analysis: dict[str, Any]) -> str:
    schedulers = [
        scheduler
        for scheduler in analysis["filters"]["schedulers"]
        if scheduler != "fsrs6"
    ]
    if not schedulers:
        raise ValueError("analysis summary does not contain a non-fsrs6 scheduler.")
    return str(schedulers[0])


def _run_metadata(
    *,
    run_root: Path,
    analysis: dict[str, Any],
    scheduler: str,
) -> dict[str, Any]:
    resolved = _resolved_config(run_root)
    run_record = _read_optional_json(run_root / "analyze-pareto" / "run_record.json")
    training = _mapping(resolved.get("training"))
    portfolio = _mapping(training.get("portfolio"))
    baseline_dr_selection = _mapping(resolved.get("baseline_dr_selection"))
    users = _mapping(resolved.get("users"))
    analyze_filters = _mapping(analysis.get("filters"))
    command = _mapping(run_record.get("command")).get("command")
    return {
        "run_id": run_root.name,
        "run_root": _display_path(run_root),
        "scheduler": scheduler,
        "config_path": _display_path_value(
            run_record.get("config_path") or resolved.get("config_path")
        ),
        "formal_command": command if isinstance(command, list) else [],
        "seed": resolved.get("seed"),
        "users": list(users.get("train", []))
        if isinstance(users.get("train"), list)
        else [],
        "environments": list(analyze_filters.get("envs", []))
        if isinstance(analyze_filters.get("envs"), list)
        else [],
        "portfolio_budget": dict(portfolio),
        "portfolio_budget_text": _portfolio_budget_text(portfolio),
        "baseline_dr_manifest": _display_path_value(
            baseline_dr_selection.get("manifest")
        ),
        "baseline_dr_target_count": baseline_dr_selection.get("target_count"),
    }


def _resolved_config(run_root: Path) -> dict[str, Any]:
    for rel in (
        "analyze-pareto/resolved_config.json",
        "train-overfit/resolved_config.json",
        "sweep/resolved_config.json",
    ):
        path = run_root / rel
        if path.exists():
            return _read_json(path)
    return {}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _portfolio_budget_text(portfolio: dict[str, Any]) -> str:
    keys = ("population_size", "offspring_size", "generations", "portfolio_size")
    labels = {
        "population_size": "population",
        "offspring_size": "offspring",
        "generations": "generations",
        "portfolio_size": "portfolio",
    }
    parts = [
        f"{labels[key]}={portfolio[key]}"
        for key in keys
        if portfolio.get(key) is not None
    ]
    return ", ".join(parts) if parts else "-"


def _provenance(run_root: Path) -> dict[str, Any]:
    analyze_stage_summary = run_root / "analyze-pareto" / "analyze_pareto_summary.json"
    if analyze_stage_summary.exists():
        summary = _read_json(analyze_stage_summary)
        return dict(summary.get("environment", {}))
    return {}


def _stage_summaries(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/gate_summary.json")):
        data = _read_json(path)
        rows.append(
            {
                "stage": path.parent.name,
                "passed": bool(data.get("passed")),
                "failures": list(data.get("failures", [])),
                "path": _display_path(path),
            }
        )
    return rows


def _performance_summaries(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/performance_summary.json")):
        data = _read_json(path)
        runtime = data.get("runtime_metrics", {})
        rows.append(
            {
                "stage": path.parent.name,
                "path": _display_path(path),
                "passed": data.get("passed"),
                "device": data.get("device"),
                "elapsed_seconds": _number(runtime.get("elapsed_seconds")),
                "user_days_per_second": _number(runtime.get("user_days_per_second")),
                "candidate_days_per_second": _number(
                    runtime.get("candidate_days_per_second")
                ),
                "gpu_metrics": dict(data.get("gpu_metrics", {})),
            }
        )
    return rows


def _gpu_monitor_summaries(run_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for perf in _performance_summaries(run_root):
        gpu = perf["gpu_metrics"]
        summary_path = gpu.get("gpu_monitor_summary_path")
        if not summary_path:
            continue
        path = Path(str(summary_path))
        monitor = _read_json(path) if path.exists() else {}
        summaries.append(
            {
                "stage": perf["stage"],
                "summary_path": _display_path(path),
                "shared_memory_peak_single_adapter_bytes": monitor.get(
                    "shared_memory_peak_single_adapter_bytes"
                ),
                "shared_memory_peak_summed_bytes": monitor.get(
                    "shared_memory_peak_summed_bytes"
                ),
                "shared_memory_spill_detected": monitor.get(
                    "shared_memory_spill_detected"
                ),
                "nvidia_smi_peak_memory_used_mib": monitor.get(
                    "nvidia_smi_peak_memory_used_mib"
                ),
                "notes": list(monitor.get("notes", [])),
            }
        )
    return summaries


def _training_hv_gains(run_root: Path) -> dict[str, Any]:
    training_summary_path = run_root / "train-overfit" / "training_summary.json"
    if not training_summary_path.exists():
        return {"sum": None, "users": 0, "by_user": []}
    training_summary = _read_json(training_summary_path)
    by_user: list[dict[str, Any]] = []
    for raw_path in training_summary.get("training_progress_paths", []):
        path = Path(str(raw_path))
        if not path.exists():
            continue
        user_id: int | None = None
        final_gain: float | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if user_id is None and isinstance(record.get("user_id"), int):
                    user_id = int(record["user_id"])
                if record.get("event") == "sms_emoa_generation":
                    final_gain = _number(record.get("hypervolume_improvement"))
        if user_id is None:
            user_id = _user_id_from_path(path)
        if user_id is not None and final_gain is not None:
            by_user.append({"user_id": user_id, "final_hv_gain": final_gain})
    by_user.sort(key=lambda row: row["user_id"])
    return {
        "sum": sum(row["final_hv_gain"] for row in by_user) if by_user else None,
        "users": len(by_user),
        "by_user": by_user,
    }


def _external_pareto_environments(
    *,
    candidate_analysis: dict[str, Any],
    comparison_analysis: dict[str, Any],
    candidate_scheduler: str,
    comparison_scheduler: str,
) -> list[dict[str, Any]]:
    environments = sorted(
        set(candidate_analysis["environments"])
        & set(comparison_analysis["environments"])
    )
    rows: list[dict[str, Any]] = []
    for env in environments:
        candidate = _env_scheduler_metrics(candidate_analysis, env, candidate_scheduler)
        comparison = _env_scheduler_metrics(
            comparison_analysis, env, comparison_scheduler
        )
        rows.append(
            {
                "environment": env,
                "candidate": candidate,
                "comparison": comparison,
                "delta": _pareto_delta(candidate, comparison),
            }
        )
    return rows


def _per_user_hv_environments(
    *,
    candidate_analysis: dict[str, Any],
    comparison_analysis: dict[str, Any],
    candidate_scheduler: str,
    comparison_scheduler: str,
) -> list[dict[str, Any]]:
    environments = sorted(
        set(candidate_analysis["environments"])
        & set(comparison_analysis["environments"])
    )
    output: list[dict[str, Any]] = []
    for env in environments:
        candidate_rows = {
            int(row["user_id"]): row
            for row in candidate_analysis["environments"][env]["per_user_hypervolume"][
                candidate_scheduler
            ]
        }
        comparison_rows = {
            int(row["user_id"]): row
            for row in comparison_analysis["environments"][env]["per_user_hypervolume"][
                comparison_scheduler
            ]
        }
        users = sorted(set(candidate_rows) & set(comparison_rows))
        output.append(
            {
                "environment": env,
                "rows": [
                    {
                        "user_id": user_id,
                        "candidate_hv_delta": candidate_rows[user_id]["hv_delta"],
                        "comparison_hv_delta": comparison_rows[user_id]["hv_delta"],
                        "delta": _subtract(
                            candidate_rows[user_id]["hv_delta"],
                            comparison_rows[user_id]["hv_delta"],
                        ),
                    }
                    for user_id in users
                ],
            }
        )
    return output


def _policy_point_diagnostics(
    *,
    candidate_analysis: dict[str, Any],
    comparison_analysis: dict[str, Any],
    candidate_scheduler: str,
    comparison_scheduler: str,
) -> list[dict[str, Any]]:
    environments = sorted(
        set(candidate_analysis["environments"])
        & set(comparison_analysis["environments"])
    )
    rows: list[dict[str, Any]] = []
    for env in environments:
        for label, analysis, scheduler in (
            ("candidate", candidate_analysis, candidate_scheduler),
            ("comparison", comparison_analysis, comparison_scheduler),
        ):
            diagnostics = analysis["environments"][env].get(
                "policy_point_diagnostics", []
            )
            row = _find_scheduler_row(diagnostics, scheduler)
            if row:
                rows.append(
                    {
                        "side": label,
                        "environment": env,
                        "scheduler": scheduler,
                        "policy_point_avg_memorized": row.get(
                            "policy_point_avg_memorized"
                        ),
                        "policy_point_avg_time": row.get("policy_point_avg_time"),
                        "policy_point_avg_efficiency": row.get(
                            "policy_point_avg_efficiency"
                        ),
                        "policy_point_avg_reviews": row.get("policy_point_avg_reviews"),
                    }
                )
    return rows


def _env_scheduler_metrics(
    analysis: dict[str, Any],
    env: str,
    scheduler: str,
) -> dict[str, Any]:
    env_summary = analysis["environments"][env]
    hv = _find_scheduler_row(env_summary["primary_hypervolume_summary"], scheduler)
    budget = _find_scheduler_row(
        _summary_rows(
            env_summary,
            "same_budget_memory_lift_auc",
            "budget_memory_gain_auc",
        ),
        scheduler,
    )
    time_saved = _find_scheduler_row(
        _summary_rows(
            env_summary,
            "same_target_time_saved_auc",
            "memory_target_regret_auc",
        ),
        scheduler,
    )
    return {
        "scheduler": scheduler,
        "hv_delta_sum": hv.get("hv_delta_sum"),
        "hv_delta_baseline_ratio_percent": hv.get("hv_delta_baseline_ratio_percent"),
        "frontier_points": hv.get("scheduler_frontier_points"),
        "same_budget_memory_lift_auc": _metric_value(
            budget,
            "same_budget_memory_lift_auc_mean",
            "memory_gain_auc_mean",
        ),
        "baseline_memory_auc": budget.get("baseline_memory_auc_mean"),
        "relative_same_budget_memory_lift_auc_percent": _existing_relative_percent(
            _metric_value(
                budget,
                "relative_same_budget_memory_lift_auc_percent",
                "relative_gain_auc_percent",
            )
        ),
        "covered_budget_count": budget.get("covered_budget_count"),
        "budget_count": budget.get("budget_count"),
        "budget_span_coverage_percent": budget.get("span_coverage_percent"),
        "same_target_time_saved_auc": _metric_value(
            time_saved,
            "same_target_time_saved_auc_mean",
            "time_regret_auc_mean",
            legacy_sign=-1.0,
        ),
        "baseline_time_auc": time_saved.get("baseline_time_auc_mean"),
        "relative_same_target_time_saved_auc_percent": _existing_relative_percent(
            _metric_value(
                time_saved,
                "relative_same_target_time_saved_auc_percent",
                "relative_regret_auc_percent",
                legacy_sign=-1.0,
            )
        ),
        "covered_target_count": time_saved.get("covered_target_count"),
        "target_count": time_saved.get("target_count"),
        "target_span_coverage_percent": time_saved.get("span_coverage_percent"),
    }


def _find_scheduler_row(rows: list[dict[str, Any]], scheduler: str) -> dict[str, Any]:
    for row in rows:
        if row.get("scheduler") == scheduler:
            return row
    return {}


def _summary_rows(
    env_summary: dict[str, Any],
    key: str,
    legacy_key: str,
) -> list[dict[str, Any]]:
    return env_summary.get(key) or env_summary.get(legacy_key, [])


def _metric_value(
    row: dict[str, Any],
    key: str,
    legacy_key: str,
    *,
    legacy_sign: float = 1.0,
) -> Any:
    if key in row:
        return row[key]
    value = row.get(legacy_key)
    if isinstance(value, int | float):
        return legacy_sign * float(value)
    return value


def _pareto_delta(
    candidate: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    return {
        "scheduler": f"{candidate.get('scheduler')} - {comparison.get('scheduler')}",
        "hv_delta_sum": _subtract(
            candidate.get("hv_delta_sum"), comparison.get("hv_delta_sum")
        ),
        "hv_delta_baseline_ratio_percent": _subtract(
            candidate.get("hv_delta_baseline_ratio_percent"),
            comparison.get("hv_delta_baseline_ratio_percent"),
        ),
        "frontier_points": _subtract(
            candidate.get("frontier_points"), comparison.get("frontier_points")
        ),
        "same_budget_memory_lift_auc": _subtract(
            candidate.get("same_budget_memory_lift_auc"),
            comparison.get("same_budget_memory_lift_auc"),
        ),
        "relative_same_budget_memory_lift_auc_percent": _subtract(
            candidate.get("relative_same_budget_memory_lift_auc_percent"),
            comparison.get("relative_same_budget_memory_lift_auc_percent"),
        ),
        "covered_budget_count": _subtract(
            candidate.get("covered_budget_count"),
            comparison.get("covered_budget_count"),
        ),
        "budget_count": candidate.get("budget_count"),
        "budget_span_coverage_percent": _subtract(
            candidate.get("budget_span_coverage_percent"),
            comparison.get("budget_span_coverage_percent"),
        ),
        "same_target_time_saved_auc": _subtract(
            candidate.get("same_target_time_saved_auc"),
            comparison.get("same_target_time_saved_auc"),
        ),
        "relative_same_target_time_saved_auc_percent": _subtract(
            candidate.get("relative_same_target_time_saved_auc_percent"),
            comparison.get("relative_same_target_time_saved_auc_percent"),
        ),
        "covered_target_count": _subtract(
            candidate.get("covered_target_count"),
            comparison.get("covered_target_count"),
        ),
        "target_count": candidate.get("target_count"),
        "target_span_coverage_percent": _subtract(
            candidate.get("target_span_coverage_percent"),
            comparison.get("target_span_coverage_percent"),
        ),
    }


def _run_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for side in ("candidate", "comparison"):
        metadata = summary["run_metadata"][side]
        rows.append(
            [
                f"`{metadata['run_id']}`",
                f"`{metadata['scheduler']}`",
                f"`{metadata['config_path']}`",
                str(metadata["portfolio_budget_text"]),
                f"`{metadata['baseline_dr_manifest']}`",
            ]
        )
    return rows


def _provenance_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for side in ("candidate", "comparison"):
        label = summary["labels"][side]
        provenance = summary["provenance"][side]
        device = _primary_device_name(summary["performance"][side])
        rows.append(
            [
                label,
                str(provenance.get("git_commit", "-")),
                str(provenance.get("dirty", "-")).lower(),
                str(provenance.get("python_version", "-")),
                str(provenance.get("torch_version", "-")),
                str(provenance.get("cuda_version", "-")),
                device,
            ]
        )
    return rows


def _primary_device_name(performance_rows: list[dict[str, Any]]) -> str:
    for row in performance_rows:
        gpu = _mapping(row.get("gpu_metrics"))
        name = gpu.get("device_name")
        if isinstance(name, str) and name:
            return name
    return "-"


def _stage_status_rows(summary: dict[str, Any], side: str) -> list[list[str]]:
    label = summary["labels"][side]
    return [
        [
            label,
            row["stage"],
            "yes" if row["passed"] else "no",
            ", ".join(str(item) for item in row["failures"]) or "-",
        ]
        for row in summary["stages"][side]
    ]


def _performance_rows(summary: dict[str, Any], side: str) -> list[list[str]]:
    label = summary["labels"][side]
    return [
        [
            label,
            row["stage"],
            str(row["device"]),
            fmt_number(row["elapsed_seconds"], digits=1),
            fmt_number(row["user_days_per_second"], digits=1),
            fmt_number(row["candidate_days_per_second"], digits=1),
        ]
        for row in summary["performance"][side]
    ]


def _gpu_monitor_rows(summary: dict[str, Any], side: str) -> list[list[str]]:
    label = summary["labels"][side]
    rows: list[list[str]] = []
    for row in summary["gpu_monitor"][side]:
        rows.append(
            [
                label,
                row["stage"],
                f"`{row['summary_path']}`",
                fmt_mib(row["shared_memory_peak_single_adapter_bytes"]),
                fmt_mib(row["shared_memory_peak_summed_bytes"]),
                str(row["shared_memory_spill_detected"]),
                fmt_number(row["nvidia_smi_peak_memory_used_mib"], digits=1),
            ]
        )
    if rows:
        return rows
    return [[label, "-", "-", "-", "-", "-", "-"]]


def _legacy_gpu_rows(summary: dict[str, Any], side: str) -> list[list[str]]:
    label = summary["labels"][side]
    rows: list[list[str]] = []
    for row in summary["performance"][side]:
        gpu = _mapping(row.get("gpu_metrics"))
        rows.append(
            [
                label,
                str(row["stage"]),
                str(row["device"]),
                fmt_mib(gpu.get("peak_allocated_memory_bytes")),
                fmt_mib(gpu.get("peak_reserved_memory_bytes")),
                "unavailable",
            ]
        )
    return rows or [[label, "-", "-", "-", "-", "unavailable"]]


def _conclusion_lines(summary: dict[str, Any]) -> list[str]:
    candidate_label = summary["labels"]["candidate"]
    comparison_label = summary["labels"]["comparison"]
    candidate_scheduler = summary["external_pareto"]["candidate_scheduler"]
    environments = summary["external_pareto"]["environments"]
    deltas = [env["delta"] for env in environments]
    all_hv_negative = bool(deltas) and all(
        (_number(delta.get("hv_delta_sum")) or 0.0) < 0.0 for delta in deltas
    )
    if all_hv_negative:
        interpretation = (
            f"With the matched portfolio budget, {candidate_label} remains behind "
            f"{comparison_label} on HV; same-budget memory lift and same-target "
            "time saved deltas are:"
        )
        diagnostic_note = (
            "Training HV gains and lower-time sampled policy points do not survive "
            "external Pareto evaluation."
        )
    else:
        interpretation = (
            "Candidate-minus-comparison deltas on the primary external Pareto "
            "metrics are:"
        )
        diagnostic_note = (
            "Training HV and sampled policy-point diagnostics should be interpreted "
            "against the external Pareto metrics."
        )
    lines = [
        f"Do not promote `{candidate_scheduler}`."
        if all_hv_negative
        else f"Promotion decision for `{candidate_scheduler}` is inconclusive.",
        "",
        interpretation,
        "",
    ]
    for env in environments:
        delta = env["delta"]
        lines.append(
            "- "
            f"{env['environment']}: "
            f"{fmt_number(delta.get('hv_delta_sum'), digits=0)} HV, "
            f"{fmt_number(delta.get('same_budget_memory_lift_auc'), digits=1)} "
            "same-budget memory lift AUC, "
            f"{fmt_number(delta.get('same_target_time_saved_auc'), digits=2)} "
            "same-target time saved AUC versus comparison."
        )
    lines.extend(
        [
            "",
            diagnostic_note,
        ]
    )
    return lines


def _external_pareto_rows(summary: dict[str, Any]) -> list[list[str]]:
    output: list[list[str]] = []
    for env in summary["external_pareto"]["environments"]:
        for key, label in (
            ("candidate", summary["labels"]["candidate"]),
            ("comparison", summary["labels"]["comparison"]),
            (
                "delta",
                f"{summary['labels']['candidate']} - {summary['labels']['comparison']}",
            ),
        ):
            row = env[key]
            coverage_formatter = (
                _coverage_delta_text if key == "delta" else _coverage_text
            )
            output.append(
                [
                    env["environment"],
                    label,
                    fmt_number(row.get("hv_delta_sum"), digits=0),
                    fmt_percent(row.get("hv_delta_baseline_ratio_percent")),
                    fmt_number(row.get("frontier_points"), digits=0),
                    fmt_number(row.get("same_budget_memory_lift_auc"), digits=1),
                    fmt_percent(
                        row.get("relative_same_budget_memory_lift_auc_percent")
                    ),
                    coverage_formatter(
                        row.get("covered_budget_count"),
                        row.get("budget_count"),
                        row.get("budget_span_coverage_percent"),
                    ),
                    fmt_number(row.get("same_target_time_saved_auc"), digits=2),
                    fmt_percent(row.get("relative_same_target_time_saved_auc_percent")),
                    coverage_formatter(
                        row.get("covered_target_count"),
                        row.get("target_count"),
                        row.get("target_span_coverage_percent"),
                    ),
                ]
            )
    return output


def _policy_point_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    labels = summary["labels"]
    for row in summary["diagnostics"]["policy_point_diagnostics"]:
        rows.append(
            [
                str(row["environment"]),
                labels[str(row["side"])],
                fmt_number(row.get("policy_point_avg_memorized"), digits=1),
                fmt_number(row.get("policy_point_avg_time"), digits=2),
                fmt_number(row.get("policy_point_avg_efficiency"), digits=2),
                fmt_number(row.get("policy_point_avg_reviews"), digits=2),
            ]
        )
    return rows


def _per_user_headers(summary: dict[str, Any]) -> list[str]:
    headers = ["user"]
    for env in summary["per_user_hv_deltas"]["environments"]:
        headers.extend(
            [
                f"{env['environment']} {summary['labels']['candidate']} HV delta",
                f"{env['environment']} {summary['labels']['comparison']} HV delta",
                f"{env['environment']} delta",
            ]
        )
    return headers


def _per_user_hv_sentence(summary: dict[str, Any]) -> str:
    rows = [
        row
        for env in summary["per_user_hv_deltas"]["environments"]
        for row in env["rows"]
    ]
    negative_count = sum(1 for row in rows if (_number(row.get("delta")) or 0.0) < 0.0)
    total = len(rows)
    if total > 0 and negative_count == total:
        return (
            f"{summary['labels']['candidate']} is behind "
            f"{summary['labels']['comparison']} for every user in every formal "
            "environment."
        )
    return (
        f"Candidate-minus-comparison per-user HV delta is negative for "
        f"{negative_count}/{total} environment-user rows."
    )


def _per_user_rows(summary: dict[str, Any]) -> list[list[str]]:
    envs = summary["per_user_hv_deltas"]["environments"]
    user_ids = sorted({int(row["user_id"]) for env in envs for row in env["rows"]})
    rows_by_env = {
        env["environment"]: {int(row["user_id"]): row for row in env["rows"]}
        for env in envs
    }
    output: list[list[str]] = []
    for user_id in user_ids:
        row = [str(user_id)]
        for env in envs:
            values = rows_by_env[env["environment"]].get(user_id, {})
            row.extend(
                [
                    fmt_number(values.get("candidate_hv_delta"), digits=0),
                    fmt_number(values.get("comparison_hv_delta"), digits=0),
                    fmt_number(values.get("delta"), digits=0),
                ]
            )
        output.append(row)
    return output


def _training_hv_user_rows(summary: dict[str, Any], side: str) -> list[list[str]]:
    label = summary["labels"][side]
    return [
        [
            label,
            str(row["user_id"]),
            fmt_number(row.get("final_hv_gain"), digits=0),
        ]
        for row in summary["training_hv_gains"][side]["by_user"]
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_number(value: Any, *, digits: int = 0) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:,.{digits}f}"


def fmt_percent(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:+.3f}%"


def fmt_mib(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number / (1024 * 1024):,.1f}"


def _coverage_text(covered: Any, total: Any, span_percent: Any) -> str:
    covered_number = _number(covered)
    total_number = _number(total)
    span_number = _number(span_percent)
    if covered_number is None or total_number is None:
        return "-"
    if span_number is None:
        return f"{covered_number:.0f}/{total_number:.0f}"
    return f"{covered_number:.0f}/{total_number:.0f}, {span_number:.3f}% span"


def _coverage_delta_text(covered: Any, total: Any, span_percent: Any) -> str:
    covered_number = _number(covered)
    span_number = _number(span_percent)
    if covered_number is None:
        return "-"
    if span_number is None:
        return f"{covered_number:+.0f}"
    return f"{covered_number:+.0f}, {span_number:+.3f} pp span"


def _display_path_value(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "-"
    return _display_path(Path(value))


def _display_path(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value)


def _subtract(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _existing_relative_percent(existing: Any) -> Any:
    if _number(existing) is None:
        return None
    return existing


def _user_id_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = USER_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
