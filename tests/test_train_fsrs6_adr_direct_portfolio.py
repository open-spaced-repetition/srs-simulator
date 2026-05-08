from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_fsrs6_adr_direct import CandidateMetrics
from experiments.rl_scheduler.train_fsrs6_adr_direct_portfolio import (
    ObjectivePoint,
    PortfolioCandidate,
    exclusive_hypervolume_contributions,
    hypervolume_2d,
    non_dominated_indices,
    select_sms_emoa_survivors,
)
from simulator.batched_sweep.fsrs6_adr_direct_policy import (
    resolve_fsrs6_adr_direct_policy_specs,
)
from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy


def _metrics(memorized: float, time_average: float) -> CandidateMetrics:
    return CandidateMetrics(
        memorized_average=memorized,
        time_average=time_average,
        memorized_per_minute=memorized / time_average,
        total_reviews=0,
        total_lapses=0,
        total_cost=0.0,
    )


def _candidate(
    candidate_id: int, memorized: float, time_average: float
) -> PortfolioCandidate:
    return PortfolioCandidate(
        candidate_id=candidate_id,
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metrics=_metrics(memorized, time_average),
    )


class FSRS6ADRDirectPortfolioMathTests(unittest.TestCase):
    def test_hypervolume_2d_uses_non_dominated_union(self) -> None:
        hv = hypervolume_2d(
            [
                ObjectivePoint(1.0, 5.0),
                ObjectivePoint(3.0, 2.0),
                ObjectivePoint(1.5, 1.0),
            ],
            reference=ObjectivePoint(0.0, 0.0),
        )

        self.assertAlmostEqual(hv, 9.0)
        self.assertEqual(
            non_dominated_indices(
                [
                    ObjectivePoint(1.0, 5.0),
                    ObjectivePoint(3.0, 2.0),
                    ObjectivePoint(1.5, 1.0),
                ]
            ),
            [0, 1],
        )

    def test_baseline_dominated_candidate_contribution_is_zero(self) -> None:
        contributions = exclusive_hypervolume_contributions(
            baseline_points=[ObjectivePoint(2.0, 2.0)],
            candidate_points=[ObjectivePoint(1.0, 1.0)],
            reference=ObjectivePoint(0.0, 0.0),
        )

        self.assertEqual(contributions, [0.0])

    def test_sms_emoa_selection_drops_baseline_dominated_candidate_first(self) -> None:
        survivors = select_sms_emoa_survivors(
            baseline_points=[ObjectivePoint(1.0, -1.0)],
            candidates=[
                _candidate(1, 0.5, 2.0),
                _candidate(2, 2.0, 1.5),
                _candidate(3, 1.5, 0.5),
            ],
            population_size=2,
            reference=ObjectivePoint(0.0, -3.0),
        )

        self.assertEqual({candidate.candidate_id for candidate in survivors}, {2, 3})


class FSRS6ADRDirectPortfolioResolverTests(unittest.TestCase):
    def test_policy_resolver_discovers_portfolio_children_without_dr_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                policy_dir = (
                    root / "user_1" / "lambda_0" / "policies" / f"policy_{index}"
                )
                policy_dir.mkdir(parents=True)
                FSRS6ADRDirectPolicy(
                    coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_desired_retention=None,
                ).write_json(policy_dir / "policy.json")
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_adr_direct",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "lambda_value": 0.0,
                            "portfolio_index": index,
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_adr_direct_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
                lambda_values=[0.0],
            )

        self.assertEqual(len(specs), 2)
        self.assertEqual([spec.policy_index for spec in specs], [0, 1])
        self.assertEqual(
            [spec.baseline_desired_retention for spec in specs], [None, None]
        )

    def test_policy_resolver_rejects_null_metadata_with_numeric_policy_dr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_dir = root / "user_1" / "lambda_0" / "policies" / "policy_0"
            policy_dir.mkdir(parents=True)
            FSRS6ADRDirectPolicy(
                coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                baseline_desired_retention=0.9,
            ).write_json(policy_dir / "policy.json")
            (policy_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "scheduler_name": "fsrs6_adr_direct",
                        "training_user_ids": [1],
                        "policy_path": "policy.json",
                        "baseline_desired_retention": None,
                        "lambda_value": 0.0,
                        "portfolio_index": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "metadata has null"):
                resolve_fsrs6_adr_direct_policy_specs(
                    user_ids=[1],
                    dr_values=[0.90],
                    policy_root=root,
                    lambda_values=[0.0],
                )


if __name__ == "__main__":
    unittest.main()
