from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ROOT = Path("artifacts/rl_scheduler/fsrs6_cost_adr_init_sensitivity_users_1_8")
DEFAULT_REPORT = Path(
    "docs/rl_scheduler/experiments/"
    "2026-05-29-fsrs6_cost_adr_init_sensitivity_users_1_8.md"
)
DEFAULT_BASELINE_CONDITION = "first8_mean"
GENERATION_CHECKPOINTS = (0, 1, 2, 5, 10, 15, 19)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    condition: str
    seed: int
    run_id: str
    run_root: Path
    fsrs6_hv_delta: float
    fsrs6_hv_ratio_percent: float
    fsrs6_relative_time_save_auc_percent: float
    fsrs6_target_span_coverage_percent: float
    fsrs6_relative_memory_lift_auc_percent: float
    fsrs6_budget_span_coverage_percent: float
    fsrs6_frontier_points: int
    training_final_hv_delta: float
    training_generation_hv_delta: dict[int, float]
    all_summary_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "run_id": self.run_id,
            "run_root": _relative(self.run_root),
            "fsrs6_hv_delta": self.fsrs6_hv_delta,
            "fsrs6_hv_ratio_percent": self.fsrs6_hv_ratio_percent,
            "fsrs6_relative_time_save_auc_percent": (
                self.fsrs6_relative_time_save_auc_percent
            ),
            "fsrs6_target_span_coverage_percent": (
                self.fsrs6_target_span_coverage_percent
            ),
            "fsrs6_relative_memory_lift_auc_percent": (
                self.fsrs6_relative_memory_lift_auc_percent
            ),
            "fsrs6_budget_span_coverage_percent": (
                self.fsrs6_budget_span_coverage_percent
            ),
            "fsrs6_frontier_points": self.fsrs6_frontier_points,
            "training_final_hv_delta": self.training_final_hv_delta,
            "training_generation_hv_delta": {
                str(key): value
                for key, value in self.training_generation_hv_delta.items()
            },
            "all_summary_passed": self.all_summary_passed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the FSRS6 Cost-ADR initialization-sensitivity runs.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--baseline-condition",
        default=DEFAULT_BASELINE_CONDITION,
        help="Condition used as the paired-difference baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_path(args.root)
    manifest_path = root / "init_sensitivity_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    runs = [_load_run_metrics(item) for item in manifest["runs"]]
    if not runs:
        raise SystemExit("Manifest contains no runs.")

    condition_summaries = _condition_summaries(runs)
    paired_differences = _paired_differences(
        runs,
        baseline_condition=args.baseline_condition,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "root": _relative(root),
        "manifest": _relative(manifest_path),
        "baseline_condition": args.baseline_condition,
        "runs": [run.to_dict() for run in runs],
        "condition_summaries": condition_summaries,
        "paired_differences": paired_differences,
    }
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "init_sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path = _repo_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            manifest=manifest,
            runs=runs,
            condition_summaries=condition_summaries,
            paired_differences=paired_differences,
            summary_path=summary_path,
            baseline_condition=args.baseline_condition,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary: {_relative(summary_path)}")
    print(f"Wrote report: {_relative(report_path)}")
    return 0


def _load_run_metrics(raw: dict[str, Any]) -> RunMetrics:
    run_root = _repo_path(Path(raw["run_root"]))
    analysis_path = (
        run_root / "analyze-pareto" / "analyze_pareto_outputs" / "analysis_summary.json"
    )
    all_summary_path = run_root / "all" / "all_summary.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis summary: {analysis_path}")
    analysis = _read_json(analysis_path)
    env = analysis["environments"]["fsrs6"]
    hv = env["primary_hypervolume_summary"][0]
    time_auc = _scheduler_entry(env["same_target_time_saved_auc"])
    memory_auc = _scheduler_entry(env["same_budget_memory_lift_auc"])
    generation_hv = _training_generation_hv(run_root)
    final_generation = max(generation_hv)
    return RunMetrics(
        condition=str(raw["condition"]),
        seed=int(raw["seed"]),
        run_id=str(raw["run_id"]),
        run_root=run_root,
        fsrs6_hv_delta=float(hv["hv_delta_sum"]),
        fsrs6_hv_ratio_percent=float(hv["hv_delta_baseline_ratio_percent"]),
        fsrs6_relative_time_save_auc_percent=float(
            time_auc["relative_same_target_time_saved_auc_percent"]
        ),
        fsrs6_target_span_coverage_percent=float(time_auc["span_coverage_percent"]),
        fsrs6_relative_memory_lift_auc_percent=float(
            memory_auc["relative_same_budget_memory_lift_auc_percent"]
        ),
        fsrs6_budget_span_coverage_percent=float(memory_auc["span_coverage_percent"]),
        fsrs6_frontier_points=int(hv["scheduler_frontier_points"]),
        training_final_hv_delta=generation_hv[final_generation],
        training_generation_hv_delta={
            generation: generation_hv[generation]
            for generation in GENERATION_CHECKPOINTS
            if generation in generation_hv
        },
        all_summary_passed=bool(_read_json(all_summary_path).get("passed"))
        if all_summary_path.exists()
        else False,
    )


def _training_generation_hv(run_root: Path) -> dict[int, float]:
    by_generation: dict[int, float] = defaultdict(float)
    progress_paths = sorted(
        (run_root / "train-overfit" / "train_outputs").glob(
            "user_*/training_progress.jsonl"
        )
    )
    if not progress_paths:
        raise FileNotFoundError(f"No training progress files under {run_root}")
    for path in progress_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "cmaes_generation":
                continue
            generation = int(event["generation"])
            by_generation[generation] += float(event["best_hypervolume_delta"])
    if not by_generation:
        raise ValueError(f"No cmaes_generation records under {run_root}")
    return dict(sorted(by_generation.items()))


def _condition_summaries(runs: list[RunMetrics]) -> dict[str, Any]:
    by_condition: dict[str, list[RunMetrics]] = defaultdict(list)
    for run in runs:
        by_condition[run.condition].append(run)
    summaries: dict[str, Any] = {}
    for condition, condition_runs in sorted(by_condition.items()):
        summaries[condition] = {
            "run_count": len(condition_runs),
            "seeds": sorted(run.seed for run in condition_runs),
            "all_passed": all(run.all_summary_passed for run in condition_runs),
            "fsrs6_hv_delta": _stats(run.fsrs6_hv_delta for run in condition_runs),
            "fsrs6_relative_time_save_auc_percent": _stats(
                run.fsrs6_relative_time_save_auc_percent for run in condition_runs
            ),
            "fsrs6_target_span_coverage_percent": _stats(
                run.fsrs6_target_span_coverage_percent for run in condition_runs
            ),
            "fsrs6_relative_memory_lift_auc_percent": _stats(
                run.fsrs6_relative_memory_lift_auc_percent for run in condition_runs
            ),
            "fsrs6_budget_span_coverage_percent": _stats(
                run.fsrs6_budget_span_coverage_percent for run in condition_runs
            ),
            "training_final_hv_delta": _stats(
                run.training_final_hv_delta for run in condition_runs
            ),
            "training_generation_0_hv_delta": _stats(
                run.training_generation_hv_delta.get(0, math.nan)
                for run in condition_runs
            ),
        }
    return summaries


def _paired_differences(
    runs: list[RunMetrics],
    *,
    baseline_condition: str,
) -> dict[str, Any]:
    by_key = {(run.condition, run.seed): run for run in runs}
    seeds = sorted({run.seed for run in runs if run.condition == baseline_condition})
    output: dict[str, Any] = {}
    for condition in sorted({run.condition for run in runs}):
        if condition == baseline_condition:
            continue
        diffs: dict[str, list[float]] = defaultdict(list)
        used_seeds: list[int] = []
        for seed in seeds:
            baseline = by_key.get((baseline_condition, seed))
            candidate = by_key.get((condition, seed))
            if baseline is None or candidate is None:
                continue
            used_seeds.append(seed)
            diffs["fsrs6_hv_delta"].append(
                candidate.fsrs6_hv_delta - baseline.fsrs6_hv_delta
            )
            diffs["fsrs6_relative_time_save_auc_percent"].append(
                candidate.fsrs6_relative_time_save_auc_percent
                - baseline.fsrs6_relative_time_save_auc_percent
            )
            diffs["fsrs6_target_span_coverage_percent"].append(
                candidate.fsrs6_target_span_coverage_percent
                - baseline.fsrs6_target_span_coverage_percent
            )
            diffs["training_final_hv_delta"].append(
                candidate.training_final_hv_delta - baseline.training_final_hv_delta
            )
            diffs["training_generation_0_hv_delta"].append(
                candidate.training_generation_hv_delta.get(0, math.nan)
                - baseline.training_generation_hv_delta.get(0, math.nan)
            )
        output[condition] = {
            "baseline_condition": baseline_condition,
            "seeds": used_seeds,
            **{key: _stats(values) for key, values in sorted(diffs.items())},
        }
    return output


def _stats(values: Any) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _render_report(
    *,
    manifest: dict[str, Any],
    runs: list[RunMetrics],
    condition_summaries: dict[str, Any],
    paired_differences: dict[str, Any],
    summary_path: Path,
    baseline_condition: str,
) -> str:
    lines = [
        "# FSRS6 Cost-ADR Initialization Sensitivity",
        "",
        "Date: 2026-05-29",
        "",
        "## Question",
        "",
        "Measure whether the current compressed Cost-ADR policy search is "
        "sensitive to the initialization point when every other experimental "
        "setting is held fixed and optimizer/simulator seeds are repeated as "
        "matched pairs.",
        "",
        "## Design",
        "",
        "All runs use users 1-8, FSRS6 environment, Markov off, batched engine, "
        "the 15-parameter `fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1` "
        "formula, desired-retention head, coefficient bounds `[-64, 64]`, "
        "retention bounds `[0.30, 0.995]`, `coefficient_preconditioning = "
        "none`, pop16/gen20 CMA-ES, `sigma0 = 1.0`, the same 16 Cost-ADR "
        "weights, and the same 16 fixed FSRS6 baseline DR manifest.",
        "",
        "Simulator/baseline seed is fixed at "
        f"{manifest.get('simulation_seed', 42)}. Optimizer seeds are matched "
        f"across initialization conditions: {manifest['optimizer_seeds']}.",
        "",
        "| condition | initialization point |",
        "| --- | --- |",
    ]
    condition_descriptions = {
        item["name"]: item["description"] for item in manifest["conditions"]
    }
    for condition in sorted(condition_summaries):
        lines.append(f"| `{condition}` | {condition_descriptions[condition]} |")
    lines.extend(
        [
            "",
            "The reproducible runner is:",
            "",
            "```bash",
            "uv run python experiments/rl_scheduler/run_fsrs6_cost_adr_init_sensitivity.py",
            "```",
            "",
            "The analysis summary JSON is:",
            "",
            f"`{_relative(summary_path)}`",
            "",
            "## Aggregate Results",
            "",
            "| condition | runs | FSRS6 HV delta mean +- std | rel time-save AUC mean +- std | target coverage mean +- std | train final HV mean +- std | gen0 train HV mean +- std |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition, summary in sorted(condition_summaries.items()):
        lines.append(
            "| "
            f"`{condition}` | {summary['run_count']} | "
            f"{_fmt_stats(summary['fsrs6_hv_delta'], digits=0)} | "
            f"{_fmt_stats(summary['fsrs6_relative_time_save_auc_percent'])}% | "
            f"{_fmt_stats(summary['fsrs6_target_span_coverage_percent'])}% | "
            f"{_fmt_stats(summary['training_final_hv_delta'], digits=0)} | "
            f"{_fmt_stats(summary['training_generation_0_hv_delta'], digits=0)} |"
        )
    lines.extend(
        [
            "",
            f"## Paired Deltas Versus `{baseline_condition}`",
            "",
            "| condition | seeds | HV delta diff mean +- std | rel time-save diff mean +- std | target coverage diff mean +- std | train final HV diff mean +- std | gen0 train HV diff mean +- std |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition, summary in sorted(paired_differences.items()):
        lines.append(
            "| "
            f"`{condition}` | {len(summary['seeds'])} | "
            f"{_fmt_stats(summary['fsrs6_hv_delta'], digits=0)} | "
            f"{_fmt_stats(summary['fsrs6_relative_time_save_auc_percent'])} pp | "
            f"{_fmt_stats(summary['fsrs6_target_span_coverage_percent'])} pp | "
            f"{_fmt_stats(summary['training_final_hv_delta'], digits=0)} | "
            f"{_fmt_stats(summary['training_generation_0_hv_delta'], digits=0)} |"
        )
    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| condition | seed | FSRS6 HV delta | rel time-save AUC | target coverage | budget coverage | train final HV | gen0 train HV | passed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run in sorted(runs, key=lambda item: (item.condition, item.seed)):
        lines.append(
            "| "
            f"`{run.condition}` | {run.seed} | "
            f"{run.fsrs6_hv_delta:,.0f} | "
            f"{run.fsrs6_relative_time_save_auc_percent:.3f}% | "
            f"{run.fsrs6_target_span_coverage_percent:.3f}% | "
            f"{run.fsrs6_budget_span_coverage_percent:.3f}% | "
            f"{run.training_final_hv_delta:,.0f} | "
            f"{run.training_generation_hv_delta.get(0, math.nan):,.0f} | "
            f"{'yes' if run.all_summary_passed else 'no'} |"
        )
    lines.extend(
        _interpretation(condition_summaries, paired_differences, baseline_condition)
    )
    return "\n".join(lines) + "\n"


def _interpretation(
    condition_summaries: dict[str, Any],
    paired_differences: dict[str, Any],
    baseline_condition: str,
) -> list[str]:
    lines = ["", "## Interpretation", ""]
    baseline = condition_summaries.get(baseline_condition)
    if baseline is None:
        return lines + [f"Baseline condition `{baseline_condition}` was not found."]
    best_condition = max(
        condition_summaries,
        key=lambda key: condition_summaries[key]["fsrs6_hv_delta"]["mean"],
    )
    worst_condition = min(
        condition_summaries,
        key=lambda key: condition_summaries[key]["fsrs6_hv_delta"]["mean"],
    )
    hv_spread = (
        condition_summaries[best_condition]["fsrs6_hv_delta"]["mean"]
        - condition_summaries[worst_condition]["fsrs6_hv_delta"]["mean"]
    )
    lines.append(
        "Initialization is materially relevant if the matched-seed spread is "
        "large relative to seed-to-seed noise. In this run, the best aggregate "
        f"condition is `{best_condition}` and the worst is `{worst_condition}`, "
        f"with a mean FSRS6 HV spread of {hv_spread:,.0f}."
    )
    for condition, diffs in sorted(paired_differences.items()):
        hv = diffs["fsrs6_hv_delta"]["mean"]
        time_save = diffs["fsrs6_relative_time_save_auc_percent"]["mean"]
        lines.append(
            f"Against `{baseline_condition}`, `{condition}` changes mean HV by "
            f"{hv:,.0f} and relative time-save AUC by {time_save:+.3f} pp."
        )
    lines.append(
        "Because all conditions use identical users, cost weights, objective, "
        "budget, bounds, preconditioning mode, and matched seeds, these deltas "
        "are attributable to the initialization point plus normal matched-seed "
        "optimizer noise."
    )
    return lines


def _scheduler_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in entries:
        if entry.get("scheduler") == "fsrs6_cost_adr":
            return entry
    raise KeyError("No fsrs6_cost_adr scheduler entry found.")


def _fmt_stats(stats: dict[str, float], *, digits: int = 3) -> str:
    mean = stats["mean"]
    std = stats["std"]
    return f"{mean:,.{digits}f} +- {std:,.{digits}f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(path: Path) -> Path:
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
