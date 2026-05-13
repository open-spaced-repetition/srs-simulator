from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.plot_pareto_run_comparison import _parse_series


class PlotParetoRunComparisonTests(unittest.TestCase):
    def test_parse_series_accepts_scheduler_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            outputs = run_root / "build-pareto" / "build_pareto_outputs"
            outputs.mkdir(parents=True)
            (outputs / "simulation_results_retention_sweep_user_1.json").write_text(
                "[]",
                encoding="utf-8",
            )

            series = _parse_series(
                [f"ADR time={run_root}::fsrs6_adr_time", f"ADR={run_root}::fsrs6_adr"]
            )

        self.assertEqual(series[0].label, "ADR time")
        self.assertEqual(series[0].scheduler, "fsrs6_adr_time")
        self.assertEqual(series[1].scheduler, "fsrs6_adr")

    def test_parse_series_defaults_scheduler_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            outputs = run_root / "build-pareto" / "build_pareto_outputs"
            outputs.mkdir(parents=True)
            (outputs / "simulation_results_retention_sweep_user_1.json").write_text(
                "[]",
                encoding="utf-8",
            )

            series = _parse_series([f"ADR={run_root}"])

        self.assertIsNone(series[0].scheduler)


if __name__ == "__main__":
    unittest.main()
