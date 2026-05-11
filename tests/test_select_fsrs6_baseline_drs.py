# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.select_fsrs6_baseline_drs import (
    _EvaluationJob,
    _SelectionProgressLogger,
    _chunk_evaluation_jobs,
    _resolve_progress_log_path,
)


class FSRS6BaselineDRSelectionProgressTests(unittest.TestCase):
    def test_progress_logger_records_generation_hypervolume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.progress.jsonl"
            with _SelectionProgressLogger.open(path) as logger:
                logger.write_run_start(
                    config_path=Path("config.toml"),
                    output_manifest=Path("manifest.json"),
                    user_ids=[1],
                    selection_environment="fsrs6",
                    target_count=16,
                    population_size=32,
                    generations=10,
                    max_lanes_per_batch=8192,
                )
                logger.write_generation(
                    user_id=1,
                    generation=3,
                    candidate_count=32,
                    generation_best_hypervolume=1.25,
                    generation_mean_hypervolume=1.0,
                    generation_min_hypervolume=0.5,
                    generation_max_hypervolume=1.25,
                    incumbent_best_hypervolume=1.25,
                    improved=True,
                    generation_best_desired_retention_values=(0.51, 0.71),
                    incumbent_desired_retention_values=(0.51, 0.71),
                )
                logger.write_run_complete(
                    output_manifest=Path("manifest.json"),
                    completed_users=1,
                )

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [record["event"] for record in records],
            [
                "run_started",
                "generation_evaluated",
                "run_completed",
            ],
        )
        generation_record = records[1]
        self.assertEqual(records[0]["max_lanes_per_batch"], 8192)
        self.assertEqual(generation_record["user_id"], 1)
        self.assertEqual(generation_record["generation"], 3)
        self.assertEqual(generation_record["candidate_count"], 32)
        self.assertEqual(generation_record["generation_best_hypervolume"], 1.25)
        self.assertEqual(generation_record["generation_mean_hypervolume"], 1.0)
        self.assertEqual(generation_record["incumbent_best_hypervolume"], 1.25)
        self.assertTrue(generation_record["improved"])

    def test_progress_log_defaults_next_to_manifest(self) -> None:
        path = _resolve_progress_log_path(
            output_manifest=Path("/tmp/manifest.json"),
            progress_log=None,
            no_progress_log=False,
        )

        self.assertEqual(path, Path("/tmp/manifest.progress.jsonl"))

    def test_progress_log_can_be_disabled(self) -> None:
        path = _resolve_progress_log_path(
            output_manifest=Path("/tmp/manifest.json"),
            progress_log=Path("/tmp/progress.jsonl"),
            no_progress_log=True,
        )

        self.assertIsNone(path)

    def test_chunk_evaluation_jobs_respects_lane_cap(self) -> None:
        jobs = [
            _EvaluationJob(
                user_id=1,
                candidate_index=0,
                desired_retention_values=(0.51, 0.61),
            ),
            _EvaluationJob(
                user_id=1,
                candidate_index=1,
                desired_retention_values=(0.52, 0.62),
            ),
            _EvaluationJob(
                user_id=2,
                candidate_index=0,
                desired_retention_values=(0.53, 0.63),
            ),
        ]

        chunks = _chunk_evaluation_jobs(jobs, max_lanes_per_batch=4)

        self.assertEqual(
            [[job.user_id for job in chunk] for chunk in chunks],
            [
                [1, 1],
                [2],
            ],
        )

    def test_chunk_evaluation_jobs_keeps_oversized_single_job(self) -> None:
        job = _EvaluationJob(
            user_id=1,
            candidate_index=0,
            desired_retention_values=(0.51, 0.61, 0.71),
        )

        chunks = _chunk_evaluation_jobs([job], max_lanes_per_batch=2)

        self.assertEqual(chunks, [[job]])


if __name__ == "__main__":
    unittest.main()
