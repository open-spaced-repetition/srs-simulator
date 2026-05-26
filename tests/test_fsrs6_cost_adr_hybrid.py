import json
import tempfile
import unittest
from pathlib import Path

from experiments.rl_scheduler.build_fsrs6_cost_adr_hybrid import (
    HybridSource,
    build_hybrid_root,
)


class FSRS6CostADRHybridTests(unittest.TestCase):
    def test_build_hybrid_root_selects_per_user_training_hv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "a" / "train-overfit" / "train_outputs"
            source_b = root / "b" / "train-overfit" / "train_outputs"
            baseline_root = root / "a"
            (baseline_root / "stage-baseline").mkdir(parents=True)
            (baseline_root / "stage-baseline" / "baseline_summary.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            self._write_user(source_a / "user_1", best_hv=1.0)
            self._write_user(source_b / "user_1", best_hv=2.0)
            self._write_user(source_a / "user_2", best_hv=4.0)
            self._write_user(source_b / "user_2", best_hv=3.0)

            output = root / "hybrid_run"
            manifest = build_hybrid_root(
                sources=[
                    HybridSource("a", source_a),
                    HybridSource("b", source_b),
                ],
                output_run_root=output,
                user_ids=[1, 2],
                baseline_run_root=baseline_root,
            )

            selected = {
                row["user_id"]: row["selected_source"] for row in manifest["selection"]
            }
            self.assertEqual(selected, {1: "b", 2: "a"})
            self.assertTrue(
                (
                    output
                    / "train-overfit"
                    / "train_outputs"
                    / "user_1"
                    / "policy.json"
                ).exists()
            )
            summary = json.loads(
                (output / "train-overfit" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["artifact_paths"]), 2)
            self.assertTrue(
                (output / "stage-baseline" / "baseline_summary.json").exists()
            )

    def _write_user(self, path: Path, *, best_hv: float) -> None:
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(
            json.dumps(
                {
                    "best_hypervolume_delta": best_hv,
                    "best_objective_score": best_hv,
                    "passed_overfit_gate": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (path / "metadata.json").write_text("{}\n", encoding="utf-8")
        (path / "policy.json").write_text("{}\n", encoding="utf-8")
        (path / "training_progress.jsonl").write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
