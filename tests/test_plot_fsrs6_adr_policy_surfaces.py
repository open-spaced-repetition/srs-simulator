from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler import plot_fsrs6_adr_policy_surfaces as plot  # noqa: E402
from simulator.batched_sweep.fsrs6_adr_policy import format_float_token  # noqa: E402
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy  # noqa: E402


COEFFICIENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _write_policy(policy_dir: Path, *, baseline_dr: float | None) -> Path:
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "policy.json"
    FSRS6ADRPolicy(
        coefficients=COEFFICIENTS,
        baseline_desired_retention=baseline_dr,
    ).write_json(policy_path)
    return policy_path


def _write_portfolio_child(
    root: Path,
    *,
    policy_index: int,
    memorized_average: float | None,
    config_path: Path | None = None,
) -> Path:
    policy_dir = root / "user_1" / "lambda_0" / "policies" / f"policy_{policy_index}"
    policy_path = _write_policy(policy_dir, baseline_dr=None)
    metrics: dict[str, object] = {}
    if memorized_average is not None:
        metrics["memorized_average"] = memorized_average
    (policy_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "portfolio_index": policy_index,
            }
        ),
        encoding="utf-8",
    )
    metadata: dict[str, object] = {
        "training_user_ids": [1],
        "lambda_value": 0.0,
        "baseline_desired_retention": None,
        "metrics_path": "metrics.json",
        "portfolio_index": policy_index,
    }
    if config_path is not None:
        metadata["config_snapshot_path"] = str(config_path)
    (policy_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return policy_path


class PlotFSRS6ADRPolicySurfacesTests(unittest.TestCase):
    def test_discovers_baseline_dr_policies_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dr in (0.90, 0.52):
                _write_policy(
                    root / "user_1" / "lambda_0" / f"dr_{format_float_token(dr)}",
                    baseline_dr=dr,
                )

            entries = plot._discover_policies(
                root,
                users=(1,),
                start_user=None,
                end_user=None,
                dr_values=None,
                lambda_values=None,
                deck_size=None,
            )

        self.assertEqual({entry.plot_mode for entry in entries}, {"baseline_dr"})
        self.assertEqual(
            [entry.baseline_desired_retention for entry in entries],
            [0.52, 0.9],
        )
        self.assertEqual(list(plot._group_entries(entries)), [(1, 0.0, "baseline_dr")])

    def test_discovers_portfolio_children_and_sorts_by_memorization_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text("[simulation]\ndeck = 100\n", encoding="utf-8")
            _write_portfolio_child(
                root,
                policy_index=0,
                memorized_average=80.0,
                config_path=config_path,
            )
            _write_portfolio_child(
                root,
                policy_index=1,
                memorized_average=50.0,
                config_path=config_path,
            )

            entries = plot._discover_policies(
                root,
                users=(1,),
                start_user=None,
                end_user=None,
                dr_values=None,
                lambda_values=None,
                deck_size=None,
            )

        self.assertEqual({entry.plot_mode for entry in entries}, {"memorization_ratio"})
        sorted_entries = plot._sorted_entries_for_mode(entries, "memorization_ratio")
        self.assertEqual([entry.portfolio_index for entry in sorted_entries], [1, 0])
        self.assertEqual(
            [entry.memorization_ratio for entry in sorted_entries],
            [0.5, 0.8],
        )
        self.assertEqual(
            list(plot._group_entries(entries)),
            [(1, 0.0, "memorization_ratio")],
        )

    def test_portfolio_uses_deck_size_fallback_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_portfolio_child(
                root,
                policy_index=0,
                memorized_average=50.0,
                config_path=None,
            )

            entries = plot._discover_policies(
                root,
                users=(1,),
                start_user=None,
                end_user=None,
                dr_values=None,
                lambda_values=None,
                deck_size=200,
            )

        self.assertEqual(entries[0].deck_size, 200)
        self.assertEqual(entries[0].memorization_ratio, 0.25)

    def test_portfolio_requires_metrics_and_deck_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_portfolio_child(
                root,
                policy_index=0,
                memorized_average=50.0,
                config_path=None,
            )

            with self.assertRaisesRegex(SystemExit, "deck size"):
                plot._discover_policies(
                    root,
                    users=(1,),
                    start_user=None,
                    end_user=None,
                    dr_values=None,
                    lambda_values=None,
                    deck_size=None,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text("[simulation]\ndeck = 100\n", encoding="utf-8")
            _write_portfolio_child(
                root,
                policy_index=0,
                memorized_average=None,
                config_path=config_path,
            )

            with self.assertRaisesRegex(SystemExit, "memorized_average"):
                plot._discover_policies(
                    root,
                    users=(1,),
                    start_user=None,
                    end_user=None,
                    dr_values=None,
                    lambda_values=None,
                    deck_size=None,
                )


if __name__ == "__main__":
    unittest.main()
