from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.cli import run_tradeoff_config
from experiments.single_card_tradeoff.cli import tradeoff
from experiments.single_card_tradeoff.core.tradeoff_runner import _plot_sort_key
from simulator.batched_sweep.fsrs6_adr_policy import format_float_token
from simulator.experiment_infra import validate_scheduler_artifact
from simulator.fsrs6_adr_policy import FEATURE_VERSION_LOG_POLY, FSRS6ADRPolicy
from simulator.scheduler_catalog import fsrs6_adr_variant_for_feature_version


def _base_args(policy_path: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        days=5,
        particles=16,
        deck_scale=100,
        env="fsrs6_default",
        button_usage=None,
        review_markov_transition=False,
        user_id=None,
        benchmark_result=None,
        benchmark_partition="0",
        srs_benchmark_root=None,
        torch_device="cpu",
        scheduler_priority="low_retrievability",
        fuzz=False,
        no_progress=True,
        engine="vectorized",
        target_batch_size=0,
        fsrs6_adr_policy=policy_path,
        fsrs6_adr_policy_root=None,
        fsrs6_adr_train_run_root=None,
        fsrs6_adr_policy_manifest=None,
        fsrs6_adr_lambda_values=None,
    )


class SingleCardTradeoffADRTests(unittest.TestCase):
    def test_run_specs_accepts_fsrs6_adr(self) -> None:
        args = argparse.Namespace(sched="fsrs6,fsrs6_adr", fixed_intervals=None)

        specs = tradeoff._run_specs(args)

        self.assertEqual(
            specs,
            [("fsrs6", "fsrs6", None), ("fsrs6_adr", "fsrs6_adr", None)],
        )

    def test_policy_root_expands_portfolio_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index in range(2):
                policy_dir = root / "user_1" / "policies" / f"policy_{index}"
                policy_dir.mkdir(parents=True)
                FSRS6ADRPolicy(
                    coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_desired_retention=None,
                ).write_json(policy_dir / "policy.json")
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_adr",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "portfolio_index": index,
                            "action_space": "sd_retention_function_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            args = _base_args(None)
            args.user_id = 1
            args.fsrs6_adr_policy_root = root

            specs = tradeoff._load_fsrs6_adr_policy_specs(
                args,
                retention_values=[0.5, 0.6],
            )

        self.assertEqual([spec.policy_index for spec in specs], [0, 1])
        self.assertTrue(all(spec.baseline_desired_retention is None for spec in specs))

    def test_runs_single_policy_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(desired_retention=0.9).write_json(policy_path)
            args = _base_args(policy_path)

            rows = tradeoff._run_fsrs6_adr(
                args,
                environment_name="fsrs6_default",
                scheduler_name="fsrs6_adr",
                scheduler_spec="fsrs6_adr",
                seed=42,
                retention_values=[0.5, 0.6],
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scheduler"], "fsrs6_adr")
        self.assertEqual(rows[0]["fsrs6_adr_policy"], str(policy_path.resolve()))
        self.assertEqual(rows[0]["fsrs6_adr_baseline_desired_retention"], 0.9)
        self.assertGreater(rows[0]["card_total_reviews"], 0.0)

    def test_plot_sort_key_orders_native_adr_by_lambda(self) -> None:
        rows = [
            {
                "scheduler": "fsrs6_adr",
                "goal_cost_weight": "",
                "fsrs6_adr_policy_index": "",
                "fsrs6_adr_baseline_desired_retention": "",
                "fsrs6_adr_lambda_value": lambda_value,
                "fixed_interval": "",
                "desired_retention": "",
            }
            for lambda_value in ("0", "1024", "16", "256", "4", "64")
        ]

        sorted_rows = sorted(rows, key=_plot_sort_key)

        self.assertEqual(
            [row["fsrs6_adr_lambda_value"] for row in sorted_rows],
            ["0", "4", "16", "64", "256", "1024"],
        )

    def test_native_adr_train_run_root_and_manifest_discover_multiuser_policies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "native_adr_run"
            train_outputs = root / "train-overfit" / "train_outputs"
            action_space = fsrs6_adr_variant_for_feature_version(
                FEATURE_VERSION_LOG_POLY
            ).action_space
            created_paths: list[Path] = []
            manifest_entries: list[str] = [
                'family = "single_card_tradeoff"',
                "schema_version = 1",
                'generated_at = "2026-05-21T00:00:00Z"',
                f'feature_version = "{FEATURE_VERSION_LOG_POLY}"',
                'scheduler_name = "fsrs6_adr"',
                f'action_space = "{action_space}"',
                "",
            ]

            for user_id in (1, 2):
                for lambda_value in (16.0, 32.0):
                    job_dir = (
                        train_outputs
                        / f"user_{user_id}"
                        / f"lambda_{format_float_token(lambda_value)}"
                    )
                    job_dir.mkdir(parents=True)
                    policy_path = job_dir / "policy.json"
                    metrics_path = job_dir / "metrics.json"
                    metadata_path = job_dir / "metadata.json"
                    FSRS6ADRPolicy(
                        coefficients=FSRS6ADRPolicy.baseline(
                            desired_retention=0.9,
                        ).coefficients,
                        baseline_desired_retention=None,
                        feature_version=FEATURE_VERSION_LOG_POLY,
                    ).write_json(policy_path)
                    metrics_path.write_text(
                        json.dumps(
                            {
                                "job": {
                                    "user_id": user_id,
                                    "lambda_value": lambda_value,
                                },
                                "feature_version": FEATURE_VERSION_LOG_POLY,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "artifact_kind": "scheduler-policy",
                                "artifact_id": (
                                    f"single-card-adr-{user_id}-{lambda_value:g}"
                                ),
                                "family": "single_card_tradeoff",
                                "scheduler_name": "fsrs6_adr",
                                "environment": "fsrs6_default",
                                "engine": "batched",
                                "review_markov_transition": False,
                                "training_user_ids": [user_id],
                                "validation_user_ids": [],
                                "seed": 42,
                                "policy_path": "policy.json",
                                "feature_version": FEATURE_VERSION_LOG_POLY,
                                "action_space": action_space,
                                "created_at": "2026-05-21T00:00:00Z",
                                "code_commit": "deadbeef",
                                "lambda_value": lambda_value,
                                "baseline_desired_retention": None,
                                "initial_desired_retention": 0.9,
                                "config_snapshot_path": None,
                                "training_command_path": None,
                                "metrics_path": "metrics.json",
                                "capabilities": ["event", "batched"],
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    created_paths.append(policy_path.resolve())
                    created_paths.append(metrics_path.resolve())
                    created_paths.append(metadata_path.resolve())
                    manifest_entries.extend(
                        [
                            "[[policies]]",
                            f"user_id = {user_id}",
                            f"lambda_value = {lambda_value:g}",
                            f'path = "train-overfit/train_outputs/user_{user_id}/'
                            f'lambda_{format_float_token(lambda_value)}/policy.json"',
                            "",
                        ]
                    )

            manifest_path = root / "policy_manifest.toml"
            manifest_path.write_text(
                "\n".join(manifest_entries).rstrip() + "\n",
                encoding="utf-8",
            )

            args = _base_args(None)
            args.user_id = None
            args.fsrs6_adr_train_run_root = root
            args.fsrs6_adr_policy_manifest = None
            args.fsrs6_adr_policy_root = None
            args.fsrs6_adr_lambda_values = None

            train_specs = tradeoff._load_fsrs6_adr_policy_specs(
                args,
                retention_values=[0.5, 0.6],
                user_ids=[1, 2],
            )

            args.fsrs6_adr_train_run_root = None
            args.fsrs6_adr_policy_manifest = manifest_path
            manifest_specs = tradeoff._load_fsrs6_adr_policy_specs(
                args,
                retention_values=[0.5, 0.6],
                user_ids=[1, 2],
            )

            validated = validate_scheduler_artifact(
                train_outputs / "user_1" / "lambda_16" / "metadata.json",
                require_files=True,
            )

        self.assertCountEqual([spec.path for spec in train_specs], created_paths[0::3])
        self.assertCountEqual(
            [spec.path for spec in manifest_specs], created_paths[0::3]
        )
        self.assertEqual({spec.user_id for spec in train_specs}, {1, 2})
        self.assertEqual({spec.lambda_value for spec in train_specs}, {16.0, 32.0})
        self.assertTrue(
            all(spec.baseline_desired_retention is None for spec in train_specs)
        )
        self.assertIsNone(validated.baseline_desired_retention)
        self.assertEqual(validated.lambda_value, 16.0)


class SingleCardTradeoffConfigRunnerTests(unittest.TestCase):
    def test_loads_adr_vs_476_config_and_builds_user_command(self) -> None:
        config = run_tradeoff_config.load_config(
            REPO_ROOT / "experiments/single_card_tradeoff/configs/"
            "adr_vs_476_tradeoff_first8_users.toml"
        )

        command = run_tradeoff_config._tradeoff_command(config, 3)

        self.assertEqual(config.user_ids, tuple(range(1, 9)))
        self.assertIn("--fsrs6-adr-train-run-root", command)
        self.assertIn("--oracle-stationary-finite-distill-policy", command)
        self.assertIn("user_3_policy.pt", " ".join(command))
        self.assertFalse(config.review_markov_transition)
        self.assertNotIn("--review-markov-transition", command)
        self.assertNotIn("--no-plot", command)

        markov_command = run_tradeoff_config._tradeoff_command(
            replace(config, review_markov_transition=True),
            3,
        )
        self.assertIn("--review-markov-transition", markov_command)

    def test_mean_summary_rows_compare_configured_schedulers(self) -> None:
        rows = [
            {
                "user_id": 1,
                "scheduler": "fsrs6_adr",
                "review_markov_transition": "False",
                "same_target_time_saved_auc": "2.0",
                "relative_same_target_time_saved_auc_percent": "10.0",
                "span_coverage_percent": "80.0",
            },
            {
                "user_id": 2,
                "scheduler": "fsrs6_adr",
                "review_markov_transition": "False",
                "same_target_time_saved_auc": "-1.0",
                "relative_same_target_time_saved_auc_percent": "-5.0",
                "span_coverage_percent": "60.0",
            },
            {
                "user_id": 1,
                "scheduler": "fsrs6_oracle_stationary_finite_distill",
                "review_markov_transition": "False",
                "same_target_time_saved_auc": "3.0",
                "relative_same_target_time_saved_auc_percent": "15.0",
                "span_coverage_percent": "90.0",
            },
        ]

        summary = run_tradeoff_config._mean_summary_rows(rows)

        by_scheduler = {row["scheduler"]: row for row in summary}
        self.assertEqual(by_scheduler["fsrs6_adr"]["user_count"], 2)
        self.assertEqual(by_scheduler["fsrs6_adr"]["review_markov_transition"], "False")
        self.assertEqual(by_scheduler["fsrs6_adr"]["positive_user_count"], 1)
        self.assertEqual(
            by_scheduler["fsrs6_adr"]["mean_same_target_time_saved_auc"], 0.5
        )
        self.assertEqual(
            by_scheduler["fsrs6_oracle_stationary_finite_distill"]["user_count"], 1
        )

    def test_tradeoff_csv_and_auc_include_review_markov_mode(self) -> None:
        rows = [
            {
                "environment": "fsrs6",
                "scheduler": "fsrs6",
                "scheduler_spec": "fsrs6",
                "desired_retention": 0.5,
                "fixed_interval": None,
                "goal_cost_weight": None,
                "seed": 1,
                "days": 5,
                "particles": 16,
                "deck_scale": 100,
                "card_expected_retrievability": 0.1,
                "card_minutes_per_day": 0.1,
                "card_reviews_per_day": 1.0,
                "card_total_reviews": 5.0,
                "card_total_lapses": 0.0,
                "card_total_cost_seconds": 1.0,
                "card_final_projected_retrievability": 0.5,
                "observed_retention": 0.5,
                "deck_expected_memorized": 100.0,
                "deck_minutes_per_day": 10.0,
                "deck_reviews_per_day": 1.0,
                "total_reviews": 5.0,
                "total_lapses": 0.0,
                "total_cost_seconds": 1.0,
                "runtime_s": 0.1,
                "engine": "vectorized",
                "fuzz": False,
                "review_markov_transition": False,
            },
            {
                "environment": "fsrs6",
                "scheduler": "fsrs6",
                "scheduler_spec": "fsrs6",
                "desired_retention": 0.6,
                "fixed_interval": None,
                "goal_cost_weight": None,
                "seed": 1,
                "days": 5,
                "particles": 16,
                "deck_scale": 100,
                "card_expected_retrievability": 0.2,
                "card_minutes_per_day": 0.2,
                "card_reviews_per_day": 1.0,
                "card_total_reviews": 5.0,
                "card_total_lapses": 0.0,
                "card_total_cost_seconds": 1.0,
                "card_final_projected_retrievability": 0.6,
                "observed_retention": 0.6,
                "deck_expected_memorized": 200.0,
                "deck_minutes_per_day": 20.0,
                "deck_reviews_per_day": 1.0,
                "total_reviews": 5.0,
                "total_lapses": 0.0,
                "total_cost_seconds": 1.0,
                "runtime_s": 0.1,
                "engine": "vectorized",
                "fuzz": False,
                "review_markov_transition": False,
            },
            {
                "environment": "fsrs6",
                "scheduler": "candidate",
                "scheduler_spec": "candidate",
                "desired_retention": None,
                "fixed_interval": None,
                "goal_cost_weight": 1.0,
                "seed": 1,
                "days": 5,
                "particles": 16,
                "deck_scale": 100,
                "card_expected_retrievability": 0.1,
                "card_minutes_per_day": 0.08,
                "card_reviews_per_day": 1.0,
                "card_total_reviews": 5.0,
                "card_total_lapses": 0.0,
                "card_total_cost_seconds": 1.0,
                "card_final_projected_retrievability": 0.5,
                "observed_retention": 0.5,
                "deck_expected_memorized": 100.0,
                "deck_minutes_per_day": 8.0,
                "deck_reviews_per_day": 1.0,
                "total_reviews": 5.0,
                "total_lapses": 0.0,
                "total_cost_seconds": 1.0,
                "runtime_s": 0.1,
                "engine": "vectorized",
                "fuzz": False,
                "review_markov_transition": False,
            },
            {
                "environment": "fsrs6",
                "scheduler": "candidate",
                "scheduler_spec": "candidate",
                "desired_retention": None,
                "fixed_interval": None,
                "goal_cost_weight": 2.0,
                "seed": 1,
                "days": 5,
                "particles": 16,
                "deck_scale": 100,
                "card_expected_retrievability": 0.2,
                "card_minutes_per_day": 0.16,
                "card_reviews_per_day": 1.0,
                "card_total_reviews": 5.0,
                "card_total_lapses": 0.0,
                "card_total_cost_seconds": 1.0,
                "card_final_projected_retrievability": 0.6,
                "observed_retention": 0.6,
                "deck_expected_memorized": 200.0,
                "deck_minutes_per_day": 16.0,
                "deck_reviews_per_day": 1.0,
                "total_reviews": 5.0,
                "total_lapses": 0.0,
                "total_cost_seconds": 1.0,
                "runtime_s": 0.1,
                "engine": "vectorized",
                "fuzz": False,
                "review_markov_transition": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            tradeoff._write_csv(path, rows)
            header = path.read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("review_markov_transition", header)
        auc_rows = tradeoff._build_regret_auc_rows(rows)
        candidate = next(
            row
            for row in auc_rows
            if row["baseline_scheduler"] == "fsrs6" and row["scheduler"] == "candidate"
        )
        self.assertFalse(candidate["review_markov_transition"])


if __name__ == "__main__":
    unittest.main()
