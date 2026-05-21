from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from experiments.single_card_tradeoff.run_tradeoff_config import (
    _base_tradeoff_command,
    _multiuser_tradeoff_command,
    _split_combined_outputs,
    _tradeoff_command,
    load_config,
)


class SingleCardTradeoffRunConfigTests(unittest.TestCase):
    def test_oracle_cost_weights_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    name = "test_oracle_weights"
                    seed = 42

                    [experiment]
                    env = "fsrs6"
                    user_ids = [2]
                    schedulers = ["fsrs6_oracle_stationary_finite"]
                    days = 1825
                    particles = 100
                    deck_scale = 10000
                    target_retentions = [0.5, 0.9]
                    oracle_cost_weights = [0.0, 0.5, 1.0]
                    review_markov_transition = false
                    scheduler_priority = "low_retrievability"
                    benchmark_partition = "0"
                    no_plot = true
                    no_progress = true

                    [outputs]
                    root = "artifacts/single_card_tradeoff/test_oracle_weights"
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            command = _tradeoff_command(config, 2)

        self.assertIn("--oracle-cost-weights", command)
        flag_index = command.index("--oracle-cost-weights")
        self.assertEqual(command[flag_index + 1], "0,0.5,1")

    def test_multiuser_command_uses_single_batched_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    name = "test_multiuser"
                    seed = 42

                    [experiment]
                    env = "fsrs6_default"
                    user_ids = [1, 2]
                    schedulers = ["fsrs6"]
                    days = 30
                    particles = 64
                    deck_scale = 10000
                    target_retentions = [0.5, 0.9]
                    review_markov_transition = false
                    scheduler_priority = "low_retrievability"
                    benchmark_partition = "0"
                    no_plot = true
                    no_progress = true

                    [outputs]
                    root = "artifacts/single_card_tradeoff/test_multiuser"
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            command = _multiuser_tradeoff_command(config, (1, 2))

        self.assertIn("--user-ids", command)
        self.assertNotIn("--user-id", command)
        self.assertEqual(command[command.index("--user-ids") + 1], "1,2")
        self.assertTrue(any(part.endswith("combined_results.csv") for part in command))
        self.assertTrue(
            any(part.endswith("combined_regret_auc.csv") for part in command)
        )

    def test_base_command_builder_keeps_shared_tradeoff_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    name = "test_base_builder"
                    seed = 7

                    [experiment]
                    env = "fsrs6_default"
                    user_ids = [1]
                    schedulers = ["fsrs6", "fixed"]
                    days = 30
                    particles = 64
                    deck_scale = 10000
                    target_retentions = [0.5, 0.9]
                    review_markov_transition = false
                    scheduler_priority = "review_first"
                    benchmark_partition = "1"
                    no_plot = true
                    no_progress = true

                    [outputs]
                    root = "artifacts/single_card_tradeoff/test_base_builder"
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            command = _base_tradeoff_command(
                config,
                user_flag="--user-id",
                user_value="1",
                out_path=Path("results.csv"),
                regret_auc_path=Path("regret_auc.csv"),
            )

        self.assertEqual(command[command.index("--sched") + 1], "fsrs6,fixed")
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(
            command[command.index("--scheduler-priority") + 1], "review_first"
        )

    def test_multiuser_command_forwards_policy_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    name = "test_template"
                    seed = 42

                    [experiment]
                    env = "fsrs6"
                    user_ids = [1, 2]
                    schedulers = ["fsrs6_oracle_stationary_finite_distill"]
                    days = 30
                    particles = 64
                    deck_scale = 10000
                    target_retentions = [0.5]
                    review_markov_transition = false
                    scheduler_priority = "low_retrievability"
                    benchmark_partition = "0"
                    no_plot = true
                    no_progress = true

                    [stationary_finite_distill]
                    policy_template = "artifacts/policies/user_{user_id}_policy.pt"

                    [outputs]
                    root = "artifacts/single_card_tradeoff/test_template"
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            command = _multiuser_tradeoff_command(config, (1, 2))

        self.assertIn("--oracle-stationary-finite-distill-policy-template", command)
        template = command[
            command.index("--oracle-stationary-finite-distill-policy-template") + 1
        ]
        self.assertIn("user_{user_id}_policy.pt", template)

    def test_multiuser_policy_template_requires_user_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    name = "test_template"
                    seed = 42

                    [experiment]
                    env = "fsrs6"
                    user_ids = [1, 2]
                    schedulers = ["fsrs6_oracle_stationary_finite_distill"]
                    days = 30
                    particles = 64
                    deck_scale = 10000
                    target_retentions = [0.5]
                    review_markov_transition = false
                    scheduler_priority = "low_retrievability"
                    benchmark_partition = "0"
                    no_plot = true
                    no_progress = true

                    [stationary_finite_distill]
                    policy_template = "artifacts/policies/shared_policy.pt"

                    [outputs]
                    root = "artifacts/single_card_tradeoff/test_template"
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            with self.assertRaises(ValueError):
                _multiuser_tradeoff_command(config, (1, 2))

    def test_split_combined_outputs_writes_per_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            out_root = Path(tmp_dir) / "out"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    schema_version = 1
                    name = "test_split"
                    seed = 42

                    [experiment]
                    env = "fsrs6_default"
                    user_ids = [1, 2]
                    schedulers = ["fsrs6"]
                    days = 30
                    particles = 64
                    deck_scale = 10000
                    target_retentions = [0.5]
                    review_markov_transition = false
                    scheduler_priority = "low_retrievability"
                    benchmark_partition = "0"
                    no_plot = true
                    no_progress = true

                    [outputs]
                    root = "{out_root}"
                    """
                ).strip(),
                encoding="utf-8",
            )
            config = load_config(config_path)
            out_root.mkdir()
            (out_root / "combined_results.csv").write_text(
                "user_id,environment,scheduler\n"
                "1,fsrs6_default,fsrs6\n"
                "2,fsrs6_default,fsrs6\n",
                encoding="utf-8",
            )
            (out_root / "combined_regret_auc.csv").write_text(
                "user_id,baseline_scheduler,scheduler,review_markov_transition\n"
                "1,fsrs6,fsrs6,False\n"
                "2,fsrs6,fsrs6,False\n",
                encoding="utf-8",
            )

            _split_combined_outputs(config, (1, 2))

            self.assertTrue((out_root / "user_1" / "results.csv").exists())
            self.assertTrue((out_root / "user_2" / "regret_auc.csv").exists())
            user_1_results = (out_root / "user_1" / "results.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("1,fsrs6_default,fsrs6", user_1_results)
            self.assertNotIn("2,fsrs6_default,fsrs6", user_1_results)


if __name__ == "__main__":
    unittest.main()
