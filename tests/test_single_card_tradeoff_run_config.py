from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from experiments.single_card_tradeoff.run_tradeoff_config import (
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


if __name__ == "__main__":
    unittest.main()
