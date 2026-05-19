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
