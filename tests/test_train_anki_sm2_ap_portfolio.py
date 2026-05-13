from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from experiments.rl_scheduler.policy_search_common import CandidateMetrics
from experiments.rl_scheduler.portfolio_selection import ObjectivePoint
from experiments.rl_scheduler.train_anki_sm2_ap_portfolio import (
    AnkiSM2APPortfolioCandidate,
    AnkiSM2APPortfolioSettings,
    AnkiSM2APPortfolioTrainJob,
    AnkiSM2APSettings,
    SelectedAnkiSM2APPortfolioChild,
    UserAnkiSM2APPortfolioResult,
    _decode_parameter_delta_tensor,
    _write_portfolio_artifacts,
)
from simulator.anki_sm2_ap_policy import ANKI_SM2_AP_DEFAULT_PARAMS
from simulator.experiment_infra.artifacts import validate_scheduler_artifact
from simulator.experiment_infra.schemas import ExperimentConfig


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "rl_scheduler"
    / "configs"
    / "anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml"
)


class TrainAnkiSM2APPortfolioTests(unittest.TestCase):
    def test_decode_parameter_delta_tensor_clips_to_bounds(self) -> None:
        vectors = torch.tensor([[0.0] * 7, [1000.0] * 7], dtype=torch.float32)

        decoded = _decode_parameter_delta_tensor(
            search_vectors=vectors,
            parameter_delta_scale=1.0,
        )

        for actual, expected in zip(
            decoded[0], ANKI_SM2_AP_DEFAULT_PARAMS, strict=True
        ):
            self.assertAlmostEqual(float(actual), expected)
        self.assertEqual(float(decoded[1, 0]), 100.0)
        self.assertEqual(float(decoded[1, 1]), 100.0)

    def test_write_portfolio_artifacts_exports_anki_sm2_ap_metadata(self) -> None:
        config = ExperimentConfig.from_toml(CONFIG_PATH)
        metrics = CandidateMetrics(
            memorized_average=10.0,
            time_average=2.0,
            memorized_per_minute=5.0,
            total_reviews=20,
            total_lapses=1,
            total_cost=120.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train_outputs" / "user_1"
            result = UserAnkiSM2APPortfolioResult(
                job=AnkiSM2APPortfolioTrainJob(
                    user_id=1,
                    output_dir=output_dir,
                    command_record_path=None,
                ),
                baseline_desired_retention_values=(0.5,),
                baseline_metrics=[metrics],
                baseline_hypervolume=1.0,
                portfolio_hypervolume=2.0,
                hypervolume_improvement=1.0,
                final_population_hypervolume=2.0,
                final_population_hypervolume_improvement=1.0,
                reference_point=ObjectivePoint(
                    memorized_average=0.0,
                    negative_time_average=0.0,
                ),
                selected_children=[
                    SelectedAnkiSM2APPortfolioChild(
                        portfolio_index=0,
                        candidate=AnkiSM2APPortfolioCandidate(
                            candidate_id=1,
                            search_vector=(0.0,) * 7,
                            params=ANKI_SM2_AP_DEFAULT_PARAMS,
                            metrics=metrics,
                        ),
                        hypervolume_contribution=1.0,
                        pareto_rank=0,
                    )
                ],
                final_population=[],
                base_params=ANKI_SM2_AP_DEFAULT_PARAMS,
                history=[],
                passed=True,
            )

            artifacts = _write_portfolio_artifacts(
                result=result,
                config=config,
                config_path=CONFIG_PATH,
                ap_settings=AnkiSM2APSettings(),
                portfolio=AnkiSM2APPortfolioSettings(portfolio_size=1),
            )
            metadata = validate_scheduler_artifact(artifacts[0], require_files=True)

        self.assertEqual(metadata.scheduler_name, "anki_sm2_ap")
        self.assertEqual(metadata.action_space, "anki_sm2_ap_params_portfolio_child")
        self.assertIsNone(metadata.baseline_desired_retention)


if __name__ == "__main__":
    unittest.main()
