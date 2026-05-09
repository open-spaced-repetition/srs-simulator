from __future__ import annotations

import json
import os
import subprocess
from itertools import combinations
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import CandidateMetrics
from experiments.rl_scheduler.train_fsrs6_adr_direct_portfolio import (
    ObjectivePoint,
    PortfolioCandidate,
    _SelectionTask,
    _selection_payload,
    _selection_executor,
    _selection_process_pool_enabled,
    _selection_process_pool_worker_count,
    _select_survivors_for_generation,
    _select_portfolio_children,
    exclusive_hypervolume_contributions,
    hypervolume_2d,
    non_dominated_indices,
    select_sms_emoa_survivors,
)
from experiments.rl_scheduler.portfolio_selection import (
    SelectionPoint,
    assert_selection_worker_is_lightweight,
    hypervolume_2d as payload_hypervolume_2d,
    select_sms_emoa_survivor_indices,
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


def _oracle_dominates(lhs: ObjectivePoint, rhs: ObjectivePoint) -> bool:
    no_worse = (
        lhs.memorized_average >= rhs.memorized_average
        and lhs.negative_time_average >= rhs.negative_time_average
    )
    strictly_better = (
        lhs.memorized_average > rhs.memorized_average
        or lhs.negative_time_average > rhs.negative_time_average
    )
    return no_worse and strictly_better


def _oracle_non_dominated_indices(points: list[ObjectivePoint]) -> list[int]:
    indices: list[int] = []
    for index, candidate in enumerate(points):
        if not any(
            index != other_index and _oracle_dominates(other, candidate)
            for other_index, other in enumerate(points)
        ):
            indices.append(index)
    return indices


def _oracle_hypervolume_2d(
    points: list[ObjectivePoint], *, reference: ObjectivePoint
) -> float:
    clipped = [
        ObjectivePoint(
            memorized_average=max(point.memorized_average, reference.memorized_average),
            negative_time_average=max(
                point.negative_time_average,
                reference.negative_time_average,
            ),
        )
        for point in points
        if point.memorized_average > reference.memorized_average
        and point.negative_time_average > reference.negative_time_average
    ]
    if not clipped:
        return 0.0
    frontier = [clipped[index] for index in _oracle_non_dominated_indices(clipped)]
    frontier.sort(
        key=lambda point: (point.memorized_average, point.negative_time_average)
    )
    hv = 0.0
    previous_x = reference.memorized_average
    for point in frontier:
        width = max(0.0, point.memorized_average - previous_x)
        height = max(0.0, point.negative_time_average - reference.negative_time_average)
        hv += width * height
        previous_x = max(previous_x, point.memorized_average)
    return float(hv)


def _oracle_exclusive_hypervolume_contributions(
    *,
    baseline_points: list[ObjectivePoint],
    candidate_points: list[ObjectivePoint],
    reference: ObjectivePoint,
) -> list[float]:
    all_points = [*baseline_points, *candidate_points]
    total = _oracle_hypervolume_2d(all_points, reference=reference)
    contributions: list[float] = []
    baseline_count = len(baseline_points)
    for candidate_index in range(len(candidate_points)):
        without = [
            point
            for index, point in enumerate(all_points)
            if index != baseline_count + candidate_index
        ]
        contributions.append(
            max(0.0, total - _oracle_hypervolume_2d(without, reference=reference))
        )
    return contributions


def _oracle_baseline_aware_candidate_ranks(
    *,
    baseline_points: list[ObjectivePoint],
    candidate_points: list[ObjectivePoint],
) -> list[int]:
    ranks = [-1 for _candidate in candidate_points]
    baseline_dominated = {
        index
        for index, candidate in enumerate(candidate_points)
        if any(_oracle_dominates(baseline, candidate) for baseline in baseline_points)
    }
    remaining = [
        index
        for index in range(len(candidate_points))
        if index not in baseline_dominated
    ]
    rank = 0
    while remaining:
        layer_points = [
            *baseline_points,
            *[candidate_points[index] for index in remaining],
        ]
        nd = _oracle_non_dominated_indices(layer_points)
        selected = [
            remaining[index - len(baseline_points)]
            for index in nd
            if index >= len(baseline_points)
        ]
        if not selected:
            break
        for index in selected:
            ranks[index] = rank
        selected_set = set(selected)
        remaining = [index for index in remaining if index not in selected_set]
        rank += 1
    worst_rank = rank + len(candidate_points) + 1
    return [rank if rank >= 0 else worst_rank for rank in ranks]


def _oracle_select_sms_emoa_survivors(
    *,
    baseline_points: list[ObjectivePoint],
    candidates: list[PortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> list[PortfolioCandidate]:
    survivors = list(candidates)
    while len(survivors) > population_size:
        candidate_points = [candidate.point for candidate in survivors]
        ranks = _oracle_baseline_aware_candidate_ranks(
            baseline_points=baseline_points,
            candidate_points=candidate_points,
        )
        contributions = _oracle_exclusive_hypervolume_contributions(
            baseline_points=baseline_points,
            candidate_points=candidate_points,
            reference=reference,
        )
        worst_rank = max(ranks)
        removal_candidates = [
            index for index, rank in enumerate(ranks) if rank == worst_rank
        ]
        remove_index = min(
            removal_candidates,
            key=lambda index: (
                contributions[index],
                survivors[index].metrics.memorized_average,
                -survivors[index].metrics.time_average,
                -survivors[index].candidate_id,
            ),
        )
        del survivors[remove_index]
    return survivors


class FSRS6ADRDirectPortfolioMathTests(unittest.TestCase):
    def test_portfolio_selection_module_does_not_import_torch(self) -> None:
        code = (
            "import sys\n"
            "import experiments.rl_scheduler.portfolio_selection\n"
            "raise SystemExit(1 if 'torch' in sys.modules else 0)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_selection_worker_initializer_rejects_torch_import(self) -> None:
        with patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaises(RuntimeError):
                assert_selection_worker_is_lightweight()

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

    def test_non_dominated_indices_match_naive_oracle(self) -> None:
        rng = random.Random(1234)
        values = [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        for count in range(13):
            for _case in range(60):
                points = [
                    ObjectivePoint(
                        memorized_average=rng.choice(values),
                        negative_time_average=rng.choice(values),
                    )
                    for _index in range(count)
                ]

                self.assertEqual(
                    non_dominated_indices(points),
                    _oracle_non_dominated_indices(points),
                )

    def test_hypervolume_and_contributions_match_naive_oracle(self) -> None:
        rng = random.Random(5678)
        values = [-1.0, 0.0, 0.0, 1.0, 2.0, 4.0, 8.0]
        for _case in range(200):
            baseline_points = [
                ObjectivePoint(rng.choice(values), rng.choice(values))
                for _index in range(rng.randint(0, 5))
            ]
            candidate_points = [
                ObjectivePoint(rng.choice(values), rng.choice(values))
                for _index in range(rng.randint(0, 8))
            ]
            reference = ObjectivePoint(
                memorized_average=rng.choice([-2.0, -1.0, 0.0]),
                negative_time_average=rng.choice([-2.0, -1.0, 0.0]),
            )

            self.assertAlmostEqual(
                hypervolume_2d(
                    [*baseline_points, *candidate_points],
                    reference=reference,
                ),
                _oracle_hypervolume_2d(
                    [*baseline_points, *candidate_points],
                    reference=reference,
                ),
            )
            self.assertEqual(
                len(
                    exclusive_hypervolume_contributions(
                        baseline_points=baseline_points,
                        candidate_points=candidate_points,
                        reference=reference,
                    )
                ),
                len(candidate_points),
            )
            for actual, expected in zip(
                exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=reference,
                ),
                _oracle_exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=reference,
                ),
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected)

    def test_contribution_edge_cases_match_naive_oracle(self) -> None:
        cases = [
            (
                [ObjectivePoint(2.0, 2.0)],
                [ObjectivePoint(1.0, 1.0), ObjectivePoint(3.0, 1.5)],
            ),
            (
                [ObjectivePoint(2.0, 2.0)],
                [ObjectivePoint(2.0, 2.0), ObjectivePoint(2.0, 2.0)],
            ),
            (
                [],
                [ObjectivePoint(1.0, 4.0), ObjectivePoint(4.0, 1.0)],
            ),
            (
                [ObjectivePoint(5.0, 5.0)],
                [ObjectivePoint(1.0, 4.0), ObjectivePoint(4.0, 1.0)],
            ),
            (
                [ObjectivePoint(-1.0, 10.0)],
                [ObjectivePoint(1.0, -1.0), ObjectivePoint(2.0, 2.0)],
            ),
            ([], []),
        ]
        reference = ObjectivePoint(0.0, 0.0)
        for baseline_points, candidate_points in cases:
            self.assertEqual(
                exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=reference,
                ),
                _oracle_exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=reference,
                ),
            )

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

    def test_sms_emoa_selection_matches_naive_oracle_for_random_cases(self) -> None:
        rng = random.Random(9012)
        for _case in range(100):
            baseline_points = [
                ObjectivePoint(
                    memorized_average=rng.choice([0.0, 1.0, 2.0, 4.0, 8.0]),
                    negative_time_average=-rng.choice([1.0, 2.0, 4.0, 8.0]),
                )
                for _index in range(rng.randint(0, 4))
            ]
            candidates = [
                _candidate(
                    candidate_id=index,
                    memorized=rng.choice([0.0, 1.0, 2.0, 4.0, 8.0]),
                    time_average=rng.choice([1.0, 2.0, 4.0, 8.0]),
                )
                for index in range(rng.randint(1, 10))
            ]
            population_size = rng.randint(1, len(candidates))
            reference = ObjectivePoint(0.0, -10.0)

            actual = select_sms_emoa_survivors(
                baseline_points=baseline_points,
                candidates=candidates,
                population_size=population_size,
                reference=reference,
            )
            expected = _oracle_select_sms_emoa_survivors(
                baseline_points=baseline_points,
                candidates=candidates,
                population_size=population_size,
                reference=reference,
            )

            self.assertEqual(
                [candidate.candidate_id for candidate in actual],
                [candidate.candidate_id for candidate in expected],
            )

    def test_payload_api_matches_object_api(self) -> None:
        baseline_points = [
            ObjectivePoint(1.0, -4.0),
            ObjectivePoint(4.0, -7.0),
        ]
        candidates = [
            _candidate(10, 1.5, 3.0),
            _candidate(11, 5.0, 8.0),
            _candidate(12, 3.0, 2.5),
            _candidate(13, 3.0, 2.5),
            _candidate(14, 6.0, 9.0),
        ]
        reference = ObjectivePoint(0.0, -10.0)
        payload = _selection_payload(
            baseline_points=baseline_points,
            candidates=candidates,
            population_size=3,
            reference=reference,
        )

        survivor_indices = select_sms_emoa_survivor_indices(payload)
        object_survivors = select_sms_emoa_survivors(
            baseline_points=baseline_points,
            candidates=candidates,
            population_size=3,
            reference=reference,
        )

        self.assertEqual(
            [candidates[index].candidate_id for index in survivor_indices],
            [candidate.candidate_id for candidate in object_survivors],
        )

    def test_payload_hypervolume_matches_object_wrapper(self) -> None:
        points = [
            ObjectivePoint(1.0, 5.0),
            ObjectivePoint(3.0, 2.0),
            ObjectivePoint(1.5, 1.0),
        ]
        reference = ObjectivePoint(0.0, 0.0)

        self.assertAlmostEqual(
            payload_hypervolume_2d(
                [
                    SelectionPoint(
                        point.memorized_average,
                        point.negative_time_average,
                    )
                    for point in points
                ],
                reference=SelectionPoint(
                    reference.memorized_average,
                    reference.negative_time_average,
                ),
            ),
            hypervolume_2d(points, reference=reference),
        )

    def test_local_and_process_pool_backends_match(self) -> None:
        baseline_points = [ObjectivePoint(1.0, -4.0)]
        reference = ObjectivePoint(0.0, -10.0)
        candidate_groups = [
            [
                _candidate(10, 1.5, 3.0),
                _candidate(11, 5.0, 8.0),
                _candidate(12, 3.0, 2.5),
            ],
            [
                _candidate(20, 2.0, 4.0),
                _candidate(21, 2.0, 4.0),
                _candidate(22, 6.0, 9.0),
            ],
        ]
        tasks = [
            _SelectionTask(
                candidates=tuple(candidates),
                payload=_selection_payload(
                    baseline_points=baseline_points,
                    candidates=candidates,
                    population_size=2,
                    reference=reference,
                ),
            )
            for candidates in candidate_groups
        ]
        local_survivors, _local_seconds = _select_survivors_for_generation(
            tasks=tasks,
            executor=None,
        )

        with (
            patch(
                "experiments.rl_scheduler.portfolio_selection.os.cpu_count",
                return_value=2,
            ),
            patch.dict(
                os.environ,
                {
                    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "1",
                    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_WORKERS": "2",
                },
                clear=True,
            ),
        ):
            executor = _selection_executor(len(tasks))
            self.assertIsNotNone(executor)
            assert executor is not None
            try:
                pool_survivors, _pool_seconds = _select_survivors_for_generation(
                    tasks=tasks,
                    executor=executor,
                )
            finally:
                executor.shutdown()

        self.assertEqual(
            [
                [candidate.candidate_id for candidate in survivor_group]
                for survivor_group in pool_survivors
            ],
            [
                [candidate.candidate_id for candidate in survivor_group]
                for survivor_group in local_survivors
            ],
        )

    def test_selection_process_pool_is_disabled_by_default_for_small_jobs(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_selection_process_pool_enabled(4))
            self.assertIsNone(_selection_executor(4))

    def test_selection_process_pool_is_enabled_by_default_for_large_jobs(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_selection_process_pool_enabled(8))

    def test_selection_process_pool_can_be_disabled_by_env(self) -> None:
        with patch.dict(
            os.environ,
            {"FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "0"},
            clear=True,
        ):
            self.assertFalse(_selection_process_pool_enabled(128))

    def test_selection_process_pool_can_be_forced_by_env(self) -> None:
        with patch.dict(
            os.environ,
            {"FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "1"},
            clear=True,
        ):
            self.assertTrue(_selection_process_pool_enabled(4))

    def test_selection_process_pool_worker_count_defaults_to_thirty_two(self) -> None:
        with (
            patch(
                "experiments.rl_scheduler.portfolio_selection.os.cpu_count",
                return_value=64,
            ),
            patch.dict(
                os.environ,
                {"FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "1"},
                clear=True,
            ),
        ):
            self.assertEqual(_selection_process_pool_worker_count(128), 32)

    def test_selection_process_pool_worker_count_respects_cap(self) -> None:
        with (
            patch(
                "experiments.rl_scheduler.portfolio_selection.os.cpu_count",
                return_value=32,
            ),
            patch.dict(
                os.environ,
                {
                    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "1",
                    "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_WORKERS": "2",
                },
                clear=True,
            ),
        ):
            self.assertEqual(_selection_process_pool_worker_count(128), 2)

    def test_selection_process_pool_worker_count_rejects_invalid_cap(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_PROCESS_POOL": "1",
                "FSRS6_ADR_DIRECT_PORTFOLIO_SELECTION_WORKERS": "0",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                _selection_process_pool_worker_count(128)

    def test_portfolio_child_selection_greedily_maximizes_exported_hv(
        self,
    ) -> None:
        baseline_points = [ObjectivePoint(0.1, -9.9)]
        reference = ObjectivePoint(0.0, -10.0)
        candidates = [
            _candidate(1, 5.0, 1.0),
            _candidate(2, 10.0, 9.0),
            _candidate(3, 6.0, 4.0),
            _candidate(4, 9.0, 2.0),
        ]

        children = _select_portfolio_children(
            baseline_points=baseline_points,
            candidates=candidates,
            portfolio_size=2,
            reference=reference,
        )

        self.assertEqual([child.candidate.candidate_id for child in children], [4, 1])
        selected_hv = hypervolume_2d(
            [*baseline_points, *[child.candidate.point for child in children]],
            reference=reference,
        )
        best_pair_hv = max(
            hypervolume_2d(
                [*baseline_points, *[candidate.point for candidate in pair]],
                reference=reference,
            )
            for pair in combinations(candidates, 2)
        )
        baseline_hv = hypervolume_2d(baseline_points, reference=reference)
        self.assertAlmostEqual(selected_hv, best_pair_hv)
        self.assertAlmostEqual(
            sum(child.hypervolume_contribution for child in children),
            selected_hv - baseline_hv,
        )


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
