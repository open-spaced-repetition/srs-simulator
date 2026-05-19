from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff import tradeoff
from experiments.single_card_tradeoff import run_tradeoff_config
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy


def _base_args(policy_path: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        days=5,
        particles=16,
        deck_scale=100,
        env="fsrs6_default",
        button_usage=None,
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
        self.assertNotIn("--no-plot", command)

    def test_mean_summary_rows_compare_configured_schedulers(self) -> None:
        rows = [
            {
                "user_id": 1,
                "scheduler": "fsrs6_adr",
                "same_target_time_saved_auc": "2.0",
                "relative_same_target_time_saved_auc_percent": "10.0",
                "span_coverage_percent": "80.0",
            },
            {
                "user_id": 2,
                "scheduler": "fsrs6_adr",
                "same_target_time_saved_auc": "-1.0",
                "relative_same_target_time_saved_auc_percent": "-5.0",
                "span_coverage_percent": "60.0",
            },
            {
                "user_id": 1,
                "scheduler": "fsrs6_oracle_stationary_finite_distill",
                "same_target_time_saved_auc": "3.0",
                "relative_same_target_time_saved_auc_percent": "15.0",
                "span_coverage_percent": "90.0",
            },
        ]

        summary = run_tradeoff_config._mean_summary_rows(rows)

        by_scheduler = {row["scheduler"]: row for row in summary}
        self.assertEqual(by_scheduler["fsrs6_adr"]["user_count"], 2)
        self.assertEqual(by_scheduler["fsrs6_adr"]["positive_user_count"], 1)
        self.assertEqual(
            by_scheduler["fsrs6_adr"]["mean_same_target_time_saved_auc"], 0.5
        )
        self.assertEqual(
            by_scheduler["fsrs6_oracle_stationary_finite_distill"]["user_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
