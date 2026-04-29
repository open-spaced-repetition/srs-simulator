from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra.status import collect_run_status


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ExperimentInfraStatusTests(unittest.TestCase):
    def test_collects_stage_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            _write_json(
                run_root / "preflight" / "preflight_summary.json",
                {"type": "preflight", "passed": True, "failures": []},
            )
            _write_json(
                run_root / "stage-baseline" / "baseline_summary.json",
                {"type": "stage-baseline", "passed": True, "failures": []},
            )

            status = collect_run_status(run_root)

        self.assertTrue(status["passed"])
        self.assertEqual(status["type"], "run-status")
        self.assertEqual(
            [stage["stage"] for stage in status["stages"]],
            ["preflight", "stage-baseline"],
        )

    def test_marks_missing_summary_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "preflight").mkdir(parents=True)

            status = collect_run_status(run_root)

        self.assertFalse(status["passed"])
        self.assertEqual(status["stages"][0]["type"], "missing-summary")
        self.assertEqual(status["stages"][0]["failures"], ["incomplete-output"])


if __name__ == "__main__":
    unittest.main()
