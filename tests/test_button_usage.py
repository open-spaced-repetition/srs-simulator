from __future__ import annotations

import unittest

from simulator.button_usage import normalize_button_usage


class ButtonUsageTests(unittest.TestCase):
    def test_invalid_probability_fallback_is_quiet_by_default(self) -> None:
        with self.assertNoLogs(level="WARNING"):
            config = normalize_button_usage(
                {"relearning_rating_prob": [float("nan"), float("nan"), float("nan")]}
            )

        self.assertEqual(config["relearning_rating_prob"], [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
