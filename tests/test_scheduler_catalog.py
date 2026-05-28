from __future__ import annotations

import os
import unittest

from simulator.fsrs6_adr_policy import (
    FEATURE_VERSION_LOG_LINEAR,
    FEATURE_VERSION_LOG_POLY,
    FEATURE_VERSION_LOG_POLY_TIME,
)
from simulator.fsrs6_cost_conditioned_adr_policy import (
    FEATURE_VERSION_INTERVAL_MONO as COST_ADR_FEATURE_VERSION_INTERVAL_MONO,
    FEATURE_VERSION_RETENTION_MONO as COST_ADR_FEATURE_VERSION_RETENTION_MONO,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z as COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2 as COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
    FEATURE_VERSION_RETENTION_MONO_DROP_XD2 as COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_XD2,
    FEATURE_VERSION_RETENTION_MONO_Z2_ONLY as COST_ADR_FEATURE_VERSION_RETENTION_MONO_Z2_ONLY,
)
from simulator.scheduler_catalog import (
    action_space_allows_lambda_none,
    batched_scheduler_names,
    event_scheduler_names,
    fsrs6_cost_adr_action_space_for_feature_version,
    fsrs6_adr_variant_for_feature_version,
    run_id_scoped_sweep_schedulers,
)


class SchedulerCatalogTests(unittest.TestCase):
    def test_event_schedulers_match_simulate_factories(self) -> None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import simulate  # noqa: E402

        self.assertEqual(
            set(event_scheduler_names()),
            set(simulate.SCHEDULER_FACTORIES),
        )

    def test_batched_schedulers_are_subset_of_event_schedulers(self) -> None:
        self.assertTrue(
            set(batched_scheduler_names()).issubset(set(event_scheduler_names()))
        )

    def test_run_id_scoped_schedulers(self) -> None:
        self.assertEqual(
            run_id_scoped_sweep_schedulers(),
            frozenset(
                {
                    "anki_sm2_ap",
                    "fsrs6_adr",
                    "fsrs6_adr_time",
                    "fsrs6_cost_adr",
                    "fsrs6_oracle_stationary_finite_distill",
                    "fsrs6_ap",
                    "fsrs6_default_adr",
                }
            ),
        )

    def test_fsrs6_adr_variant_for_feature_version(self) -> None:
        poly = fsrs6_adr_variant_for_feature_version(FEATURE_VERSION_LOG_POLY)
        linear = fsrs6_adr_variant_for_feature_version(FEATURE_VERSION_LOG_LINEAR)
        time = fsrs6_adr_variant_for_feature_version(FEATURE_VERSION_LOG_POLY_TIME)

        self.assertEqual(poly.scheduler_name, "fsrs6_adr")
        self.assertEqual(linear.scheduler_name, "fsrs6_adr")
        self.assertEqual(time.scheduler_name, "fsrs6_adr_time")
        self.assertEqual(time.action_space, "sdt_retention_function")
        self.assertEqual(
            time.portfolio_child_action_space,
            "sdt_retention_function_portfolio_child",
        )

    def test_fsrs6_cost_adr_action_space_allows_lambda_none(self) -> None:
        interval_action_space = fsrs6_cost_adr_action_space_for_feature_version(
            COST_ADR_FEATURE_VERSION_INTERVAL_MONO
        )
        retention_action_space = fsrs6_cost_adr_action_space_for_feature_version(
            COST_ADR_FEATURE_VERSION_RETENTION_MONO
        )
        compressed_retention_versions = (
            COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z,
            COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_XD2,
            COST_ADR_FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2,
            COST_ADR_FEATURE_VERSION_RETENTION_MONO_Z2_ONLY,
        )

        self.assertEqual(interval_action_space, "sd_cost_interval_function")
        self.assertEqual(retention_action_space, "sd_cost_retention_function")
        for feature_version in compressed_retention_versions:
            self.assertEqual(
                fsrs6_cost_adr_action_space_for_feature_version(feature_version),
                "sd_cost_retention_function",
            )
        self.assertTrue(action_space_allows_lambda_none(interval_action_space))
        self.assertTrue(action_space_allows_lambda_none(retention_action_space))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
