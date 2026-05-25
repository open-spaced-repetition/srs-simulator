from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ContinuousUniformHPolicyAnalysisTests(unittest.TestCase):
    def test_cli_writes_smoke_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "analysis"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.single_card_tradeoff.cli."
                    "continuous_uniform_h_policy_analysis",
                    "--env",
                    "fsrs6_default",
                    "--days",
                    "6",
                    "--cost-weights",
                    "0,16",
                    "--oracle-s-grid-size",
                    "8",
                    "--oracle-d-grid-size",
                    "8",
                    "--oracle-continuous-interval-chunk-size",
                    "2",
                    "--landmark-remaining-days",
                    "1,2,5",
                    "--torch-device",
                    "cpu",
                    "--out-dir",
                    str(out_dir),
                    "--no-plot",
                    "--no-progress",
                    "--no-dp-cache",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertTrue((out_dir / "uniform_remaining_summary.csv").exists())
            self.assertTrue(
                (out_dir / "fixed_vs_uniform_remaining_comparison.csv").exists()
            )
            self.assertTrue((out_dir / "uniform_start_vs_fixed_landmarks.csv").exists())
            self.assertTrue((out_dir / "remaining_landmarks.csv").exists())
            self.assertTrue((out_dir / "metadata.json").exists())
            self.assertTrue((out_dir / "README.md").exists())
            self.assertTrue((out_dir / "performance_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
