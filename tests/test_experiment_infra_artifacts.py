from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import (
    EngineName,
    ExperimentConfig,
    SchedulerArtifactMetadata,
    get_scheduler_capability,
    supports_scheduler,
    validate_scheduler_artifact,
)
from simulator.experiment_infra.runner import (  # noqa: E402
    _validate_sweep_artifact_metadata,
    _validate_train_artifacts,
)


def _metadata(policy_path: str = "policy.pt") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "scheduler-policy",
        "artifact_id": "user-1-overfit",
        "family": "rl_scheduler",
        "scheduler_name": "fsrs6",
        "environment": "lstm",
        "engine": "batched",
        "training_user_ids": [1],
        "validation_user_ids": [2],
        "seed": 42,
        "policy_path": policy_path,
        "feature_version": "v1",
        "action_space": "desired_retention_delta",
        "created_at": "2026-04-29T00:00:00Z",
        "code_commit": "abcdef",
        "lambda_value": 0.5,
        "review_markov_transition": False,
        "capabilities": ["batched"],
    }


class ExperimentInfraArtifactTests(unittest.TestCase):
    def test_loads_valid_scheduler_artifact_metadata(self) -> None:
        metadata = SchedulerArtifactMetadata.from_mapping(_metadata())

        self.assertEqual(metadata.engine, EngineName.BATCHED)
        self.assertEqual(metadata.training_user_ids, (1,))
        self.assertEqual(metadata.validation_user_ids, (2,))
        self.assertFalse(metadata.review_markov_transition)

    def test_resolves_relative_paths_from_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.pt").write_bytes(b"policy")
            path = root / "metadata.json"
            path.write_text(json.dumps(_metadata()), encoding="utf-8")

            metadata = validate_scheduler_artifact(path, require_files=True)

        self.assertTrue(metadata.policy_path.is_absolute())

    def test_rejects_overlapping_users(self) -> None:
        raw = _metadata()
        raw["validation_user_ids"] = [1]

        with self.assertRaisesRegex(ValueError, "disjoint"):
            SchedulerArtifactMetadata.from_mapping(raw)

    def test_rejects_unsupported_engine_scheduler_pair(self) -> None:
        raw = _metadata()
        raw["scheduler_name"] = "dash"
        raw["engine"] = "batched"

        with self.assertRaisesRegex(ValueError, "does not support"):
            SchedulerArtifactMetadata.from_mapping(raw)

    def test_capability_registry_matches_known_support(self) -> None:
        self.assertTrue(
            supports_scheduler(scheduler="fsrs6", engine="batched", environment="lstm")
        )
        self.assertFalse(
            supports_scheduler(scheduler="sspmmc", engine="batched", environment="lstm")
        )
        self.assertTrue(
            supports_scheduler(
                scheduler="fsrs6_adr", engine="batched", environment="lstm"
            )
        )
        adr_capability = get_scheduler_capability("fsrs6_adr")
        self.assertFalse(adr_capability.supports_desired_retention)
        capability = get_scheduler_capability("dash")
        self.assertTrue(capability.supports(engine="event", environment="fsrs6"))
        self.assertFalse(capability.supports(engine="batched", environment="fsrs6"))

    def test_accepts_lambda_less_cost_adr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.json").write_text("{}", encoding="utf-8")
            metadata_path = root / "metadata.json"
            raw = _metadata(policy_path="policy.json")
            raw.update(
                {
                    "artifact_id": "cost-adr-user-1",
                    "scheduler_name": "fsrs6_cost_adr",
                    "environment": "fsrs6",
                    "feature_version": "fsrs6_cost_adr_interval_mono_v1",
                    "action_space": "sd_cost_interval_function",
                    "lambda_value": None,
                    "baseline_desired_retention": None,
                    "validation_user_ids": [],
                    "capabilities": ["event", "batched"],
                }
            )
            metadata_path.write_text(json.dumps(raw), encoding="utf-8")
            config_path = root / "experiment.toml"
            config_path.write_text(
                """
schema_version = 1
name = "cost-adr-artifact"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/cost-adr-artifact"
stages = ["train-overfit", "sweep"]

[users]
train = [1]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "logs/retention_sweep"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.90]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[training]
lambda_grid = [0.5]

[training.policy_search]
baseline_desired_retention = 0.90
""".lstrip(),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_toml(config_path)

            metadata = validate_scheduler_artifact(metadata_path, require_files=True)
            train_note = _validate_train_artifacts(
                artifact_paths=[metadata_path],
                config=config,
                user_id=1,
                lambda_value=None,
                baseline_desired_retention=0.90,
            )
            sweep_note = _validate_sweep_artifact_metadata(
                metadata_path=metadata_path,
                metadata=metadata,
                config=config,
            )

        self.assertIsNone(train_note)
        self.assertIsNone(sweep_note)


if __name__ == "__main__":
    unittest.main()
