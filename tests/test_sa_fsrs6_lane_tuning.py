from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.tune_sa_fsrs6_lanes import (
    _parse_lane_values,
    _select_fastest_passed,
)


class SAFSRS6LaneTuningTests(unittest.TestCase):
    def test_parse_lane_values(self) -> None:
        self.assertEqual(_parse_lane_values("8, 16,32"), [8, 16, 32])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _parse_lane_values("8,8")
        with self.assertRaisesRegex(ValueError, ">= 1"):
            _parse_lane_values("0")

    def test_select_fastest_passed(self) -> None:
        self.assertEqual(
            _select_fastest_passed(
                [
                    {
                        "lanes": 8,
                        "status": "passed",
                        "candidate_days_per_second": 10.0,
                        "elapsed_seconds": 2.0,
                    },
                    {"lanes": 16, "status": "oom"},
                    {
                        "lanes": 32,
                        "status": "passed",
                        "candidate_days_per_second": 25.0,
                        "elapsed_seconds": 3.0,
                    },
                ]
            ),
            {
                "lanes": 32,
                "candidate_days_per_second": 25.0,
                "elapsed_seconds": 3.0,
            },
        )
        self.assertIsNone(_select_fastest_passed([{"lanes": 16, "status": "oom"}]))


if __name__ == "__main__":
    unittest.main()
