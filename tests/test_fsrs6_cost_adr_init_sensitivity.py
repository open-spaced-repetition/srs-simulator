from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler import (  # noqa: E402
    analyze_fsrs6_cost_adr_init_sensitivity as analyze,
)
from experiments.rl_scheduler import (  # noqa: E402
    run_fsrs6_cost_adr_init_sensitivity as run,
)
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
    FSRS6CostConditionedADRPolicy,
)


class FSRS6CostADRInitSensitivityTests(unittest.TestCase):
    def test_provided_vector_condition_writes_requested_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_root = root / "zero"
            provided_root = root / "provided"

            conditions = run._conditions(
                zero_policy_root=zero_root,
                provided_policy_root=provided_root,
            )
            provided = next(
                condition
                for condition in conditions
                if condition.name == "provided_vector"
            )

            self.assertEqual(
                provided.initial_coefficients,
                run.USER_PROVIDED_INITIAL_COEFFICIENTS,
            )
            self.assertEqual(
                provided.config_line(),
                f'initial_policy_root = "{provided_root.as_posix()}"',
            )

            run._write_policy_root(
                provided_root,
                coefficients=run.USER_PROVIDED_INITIAL_COEFFICIENTS,
                title="test provided vector",
            )
            policy = FSRS6CostConditionedADRPolicy.from_json(
                provided_root / "user_1" / "policy.json"
            )

        self.assertEqual(policy.coefficients, run.USER_PROVIDED_INITIAL_COEFFICIENTS)
        self.assertEqual(policy.action_head, ACTION_HEAD_RETENTION)
        self.assertEqual(
            policy.feature_version,
            FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
        )
        self.assertEqual(policy.retention_min, 0.30)
        self.assertEqual(policy.retention_max, 0.995)

    def test_paired_metric_stats_include_two_sided_sign_test(self) -> None:
        stats = analyze._paired_metric_stats([-3.0, -2.0, -1.0])

        self.assertEqual(stats["mean"], -2.0)
        self.assertEqual(stats["positive_count"], 0.0)
        self.assertEqual(stats["negative_count"], 3.0)
        self.assertEqual(stats["nonzero_count"], 3.0)
        self.assertEqual(stats["two_sided_sign_test_p"], 0.25)


if __name__ == "__main__":
    unittest.main()
