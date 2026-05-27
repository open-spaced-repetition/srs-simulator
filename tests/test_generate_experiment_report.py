# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.generate_experiment_report import (
    _display_path,
    generate_report,
)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _analysis_summary(
    *,
    scheduler: str,
    hv_delta: float,
    user_delta: float,
    include_relative_metrics: bool = True,
    legacy_metric_names: bool = False,
) -> dict[str, object]:
    budget_row: dict[str, object] = {
        "scheduler": scheduler,
        "baseline_memory_auc_mean": 100.0,
        "covered_budget_count": 2,
        "budget_count": 3,
        "span_coverage_percent": 50.0,
    }
    time_saved_row: dict[str, object] = {
        "scheduler": scheduler,
        "baseline_time_auc_mean": 10.0,
        "covered_target_count": 2,
        "target_count": 3,
        "span_coverage_percent": 60.0,
    }
    budget_key = (
        "budget_memory_gain_auc"
        if legacy_metric_names
        else "same_budget_memory_lift_auc"
    )
    time_key = (
        "memory_target_regret_auc"
        if legacy_metric_names
        else "same_target_time_saved_auc"
    )
    if legacy_metric_names:
        budget_row["memory_gain_auc_mean"] = hv_delta / 10.0
        time_saved_row["time_regret_auc_mean"] = -hv_delta / 1000.0
    else:
        budget_row["same_budget_memory_lift_auc_mean"] = hv_delta / 10.0
        time_saved_row["same_target_time_saved_auc_mean"] = hv_delta / 1000.0
    if include_relative_metrics:
        if legacy_metric_names:
            budget_row["relative_gain_auc_percent"] = hv_delta / 10.0
            time_saved_row["relative_regret_auc_percent"] = -hv_delta / 100.0
        else:
            budget_row["relative_same_budget_memory_lift_auc_percent"] = hv_delta / 10.0
            time_saved_row["relative_same_target_time_saved_auc_percent"] = (
                hv_delta / 100.0
            )

    return {
        "type": "scheduler-comparison-analysis",
        "filters": {"schedulers": ["fsrs6", scheduler]},
        "environments": {
            "fsrs6": {
                "primary_hypervolume_summary": [
                    {
                        "scheduler": scheduler,
                        "hv_delta_sum": hv_delta,
                        "hv_delta_baseline_ratio_percent": hv_delta / 100.0,
                        "scheduler_frontier_points": 2,
                    }
                ],
                budget_key: [budget_row],
                time_key: [time_saved_row],
                "per_user_hypervolume": {
                    scheduler: [
                        {
                            "user_id": 1,
                            "baseline_hv": 10.0,
                            "target_hv": 10.0 + user_delta,
                            "hv_delta": user_delta,
                            "target_frontier_count": 1,
                        }
                    ]
                },
                "policy_point_diagnostics": [
                    {
                        "scheduler": scheduler,
                        "policy_point_avg_memorized": 100.0 + hv_delta,
                        "policy_point_avg_time": 10.0,
                        "policy_point_avg_efficiency": 5.0,
                        "policy_point_avg_reviews": 20.0,
                    }
                ],
            }
        },
    }


def _write_run(
    root: Path,
    *,
    scheduler: str,
    hv_delta: float,
    user_delta: float,
    include_gpu_monitor: bool = True,
    include_relative_metrics: bool = True,
    legacy_metric_names: bool = False,
    use_policy_search_budget: bool = False,
) -> None:
    if use_policy_search_budget:
        training_config: dict[str, object] = {
            "optimizer": {
                "name": "cma_es",
                "population_size": 16,
                "generations": 20,
                "sigma0": 1.0,
            },
            "policy_search": {"cost_weights": [0.0, 1.0, 2.0]},
            "portfolio": {},
        }
    else:
        training_config = {
            "portfolio": {
                "population_size": 16,
                "offspring_size": 16,
                "generations": 20,
                "portfolio_size": 16,
            }
        }
    _write_json(
        root / "analyze-pareto" / "analyze_pareto_outputs" / "analysis_summary.json",
        _analysis_summary(
            scheduler=scheduler,
            hv_delta=hv_delta,
            user_delta=user_delta,
            include_relative_metrics=include_relative_metrics,
            legacy_metric_names=legacy_metric_names,
        ),
    )
    _write_json(
        root / "analyze-pareto" / "analyze_pareto_summary.json",
        {
            "environment": {
                "git_commit": "abc",
                "dirty": False,
                "python_version": "3.13.11",
                "torch_version": "2.9.1+cu126",
                "cuda_version": "12.6",
            }
        },
    )
    _write_json(
        root / "analyze-pareto" / "run_record.json",
        {
            "config_path": "experiments/rl_scheduler/configs/test.toml",
            "command": {
                "command": [
                    "uv",
                    "run",
                    "python",
                    "experiments/rl_scheduler/run_experiment.py",
                ]
            },
        },
    )
    _write_json(
        root / "analyze-pareto" / "resolved_config.json",
        {
            "config_path": "experiments/rl_scheduler/configs/test.toml",
            "seed": 42,
            "users": {"train": [1]},
            "baseline_dr_selection": {
                "manifest": "artifacts/rl_scheduler/baseline_dr_selection/test.json",
                "target_count": 16,
            },
            "training": training_config,
        },
    )
    _write_json(root / "train-overfit" / "gate_summary.json", {"passed": True})
    _write_json(root / "sweep" / "gate_summary.json", {"passed": True})
    gpu_metrics: dict[str, object] = {
        "device_name": "NVIDIA Test GPU",
        "peak_allocated_memory_bytes": 256 * 1024 * 1024,
        "peak_reserved_memory_bytes": 512 * 1024 * 1024,
    }
    if include_gpu_monitor:
        gpu_metrics["gpu_monitor_summary_path"] = str(
            root / "train-overfit" / "gpu_monitor" / "summary.json"
        )
    _write_json(
        root / "train-overfit" / "performance_summary.json",
        {
            "passed": True,
            "device": "cuda",
            "runtime_metrics": {
                "elapsed_seconds": 10.0,
                "user_days_per_second": 20.0,
                "candidate_days_per_second": 30.0,
            },
            "gpu_metrics": gpu_metrics,
        },
    )
    if include_gpu_monitor:
        _write_json(
            root / "train-overfit" / "gpu_monitor" / "summary.json",
            {
                "shared_memory_peak_single_adapter_bytes": 512 * 1024 * 1024,
                "shared_memory_peak_summed_bytes": 600 * 1024 * 1024,
                "shared_memory_spill_detected": False,
                "nvidia_smi_peak_memory_used_mib": 1024.0,
                "notes": [],
            },
        )
    progress = (
        root / "train-overfit" / "train_outputs" / "user_1" / "training_progress.jsonl"
    )
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(
        json.dumps({"event": "started", "user_id": 1})
        + "\n"
        + json.dumps(
            {
                "event": "sms_emoa_generation",
                "hypervolume_improvement": hv_delta / 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "train-overfit" / "training_summary.json",
        {"training_progress_paths": [str(progress)]},
    )


class GenerateExperimentReportTests(unittest.TestCase):
    def test_generates_summary_and_markdown_from_machine_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate, scheduler="candidate_sched", hv_delta=20.0, user_delta=3.0
            )
            _write_run(
                comparison, scheduler="comparison_sched", hv_delta=12.0, user_delta=1.0
            )

            summary_path, report_path, summary = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
                title="Candidate report",
                question="Does candidate improve external Pareto?",
                candidate_label="Candidate",
                comparison_label="Comparison",
            )

            markdown = report_path.read_text(encoding="utf-8")
            summary_exists = summary_path.exists()

        self.assertTrue(summary_exists)
        self.assertIn(f"Machine summary: `{summary_path}`", markdown)
        self.assertIn("## Provenance", markdown)
        self.assertIn("## Conclusion", markdown)
        self.assertIn("Promote Candidate.", markdown)
        self.assertIn("## Diagnostics", markdown)
        self.assertIn("policy-point avg memorized", markdown)
        self.assertIn("same-budget memory lift / baseline", markdown)
        self.assertIn("same-target time saved / baseline", markdown)
        self.assertIn("+2.000%", markdown)
        self.assertIn("+0.200%", markdown)
        self.assertIn(
            "population=16, offspring=16, generations=20, portfolio=16", markdown
        )
        env = summary["external_pareto"]["environments"][0]
        self.assertEqual(env["delta"]["hv_delta_sum"], 8.0)
        self.assertEqual(
            env["candidate"]["relative_same_budget_memory_lift_auc_percent"],
            2.0,
        )
        self.assertEqual(
            env["candidate"]["relative_same_target_time_saved_auc_percent"],
            0.2,
        )
        self.assertAlmostEqual(
            env["delta"]["relative_same_budget_memory_lift_auc_percent"],
            0.8,
        )
        self.assertAlmostEqual(
            env["delta"]["relative_same_target_time_saved_auc_percent"],
            0.08,
        )
        self.assertEqual(
            summary["run_metadata"]["candidate"]["baseline_dr_manifest"],
            "artifacts/rl_scheduler/baseline_dr_selection/test.json",
        )
        self.assertEqual(
            summary["diagnostics"]["policy_point_diagnostics"][0][
                "policy_point_avg_memorized"
            ],
            120.0,
        )
        self.assertEqual(
            summary["per_user_hv_deltas"]["environments"][0]["rows"][0]["delta"],
            2.0,
        )
        self.assertIn("Candidate - Comparison", markdown)

    def test_policy_search_budget_is_reported_when_portfolio_budget_is_absent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate,
                scheduler="candidate_sched",
                hv_delta=20.0,
                user_delta=3.0,
                use_policy_search_budget=True,
            )
            _write_run(
                comparison, scheduler="comparison_sched", hv_delta=12.0, user_delta=1.0
            )

            _, report_path, _ = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
                candidate_label="Candidate",
                comparison_label="Comparison",
            )

            markdown = report_path.read_text(encoding="utf-8")

        self.assertIn("training budget", markdown)
        self.assertIn(
            "optimizer=cma_es, population=16, generations=20, sigma0=1.0, "
            "cost_weights=3",
            markdown,
        )

    def test_cmaes_training_progress_reports_final_best_hv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate, scheduler="candidate_sched", hv_delta=20.0, user_delta=3.0
            )
            _write_run(
                comparison, scheduler="comparison_sched", hv_delta=12.0, user_delta=1.0
            )
            progress = (
                candidate
                / "train-overfit"
                / "train_outputs"
                / "user_1"
                / "training_progress.jsonl"
            )
            progress.write_text(
                json.dumps({"event": "started", "user_id": 1})
                + "\n"
                + json.dumps(
                    {
                        "event": "cmaes_generation",
                        "best_hypervolume_delta": 7.0,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "cmaes_completed",
                        "best_hypervolume_delta": 9.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            _, report_path, summary = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
                candidate_label="Candidate",
                comparison_label="Comparison",
            )

            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(summary["training_hv_gains"]["candidate"]["users"], 1)
        self.assertEqual(summary["training_hv_gains"]["candidate"]["sum"], 9.0)
        self.assertIn("| Candidate | 1 | 9 |", markdown)

    def test_historical_runs_without_monitor_report_torch_cuda_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate,
                scheduler="candidate_sched",
                hv_delta=-20.0,
                user_delta=-3.0,
                include_gpu_monitor=False,
            )
            _write_run(
                comparison,
                scheduler="comparison_sched",
                hv_delta=12.0,
                user_delta=1.0,
                include_gpu_monitor=False,
            )

            _, report_path, _ = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
                candidate_label="Candidate",
                comparison_label="Comparison",
            )

            markdown = report_path.read_text(encoding="utf-8")

        self.assertIn("No automatic GPU monitor artifacts", markdown)
        self.assertIn("shared-memory spill cannot be judged", markdown)
        self.assertIn("256.0", markdown)
        self.assertIn("Do not promote Candidate.", markdown)

    def test_legacy_auc_metric_names_are_converted_to_time_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate,
                scheduler="candidate_sched",
                hv_delta=20.0,
                user_delta=3.0,
                legacy_metric_names=True,
            )
            _write_run(
                comparison,
                scheduler="comparison_sched",
                hv_delta=12.0,
                user_delta=1.0,
                legacy_metric_names=True,
            )

            _, _, summary = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
            )

        env = summary["external_pareto"]["environments"][0]
        self.assertEqual(env["candidate"]["same_target_time_saved_auc"], 0.02)
        self.assertEqual(
            env["candidate"]["relative_same_target_time_saved_auc_percent"],
            0.2,
        )
        self.assertAlmostEqual(env["delta"]["same_target_time_saved_auc"], 0.008)

    def test_missing_relative_metrics_are_not_recomputed_from_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            comparison = root / "comparison"
            _write_run(
                candidate,
                scheduler="candidate_sched",
                hv_delta=20.0,
                user_delta=3.0,
                include_relative_metrics=False,
            )
            _write_run(
                comparison,
                scheduler="comparison_sched",
                hv_delta=12.0,
                user_delta=1.0,
                include_relative_metrics=False,
            )

            _, _, summary = generate_report(
                run_root=candidate,
                comparison_run_root=comparison,
                output_path=root / "docs" / "report.md",
                title="Missing relative report",
                question="Do missing relative metrics get fabricated?",
                candidate_label="Candidate",
                comparison_label="Comparison",
            )

        env = summary["external_pareto"]["environments"][0]
        self.assertIsNone(
            env["candidate"]["relative_same_budget_memory_lift_auc_percent"]
        )
        self.assertIsNone(
            env["comparison"]["relative_same_budget_memory_lift_auc_percent"]
        )
        self.assertIsNone(env["delta"]["relative_same_budget_memory_lift_auc_percent"])
        self.assertIsNone(
            env["candidate"]["relative_same_target_time_saved_auc_percent"]
        )
        self.assertIsNone(
            env["comparison"]["relative_same_target_time_saved_auc_percent"]
        )
        self.assertIsNone(env["delta"]["relative_same_target_time_saved_auc_percent"])

    def test_display_path_is_repo_relative_for_repo_paths(self) -> None:
        self.assertEqual(
            _display_path(REPO_ROOT / "docs" / "report.md"),
            "docs/report.md",
        )


if __name__ == "__main__":
    unittest.main()
