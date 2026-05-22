from __future__ import annotations

import argparse
import ast
import importlib
import sys
import unittest
from pathlib import Path

from experiments.single_card_tradeoff.core.defaults import DEFAULT_TARGET_RETENTIONS
from experiments.single_card_tradeoff.oracles.dp_cache import (
    OracleDPCacheConfig,
    resolve_oracle_dp_cache_config,
    set_default_oracle_dp_cache_config,
)
from experiments.single_card_tradeoff.models.policy_net import PolicyValueNet
from experiments.single_card_tradeoff.core.results import build_regret_auc_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_CARD_ROOT = PROJECT_ROOT / "experiments" / "single_card_tradeoff"


class SingleCardArchitectureBoundaryTests(unittest.TestCase):
    def test_package_defaults_do_not_import_tradeoff(self) -> None:
        sys.modules.pop("experiments.single_card_tradeoff.cli.tradeoff", None)
        package = importlib.import_module("experiments.single_card_tradeoff")

        self.assertEqual(package.DEFAULT_TARGET_RETENTIONS, DEFAULT_TARGET_RETENTIONS)
        self.assertNotIn("experiments.single_card_tradeoff.cli.tradeoff", sys.modules)

    def test_extracted_modules_expose_existing_entry_points(self) -> None:
        self.assertTrue(DEFAULT_TARGET_RETENTIONS)
        self.assertIsNotNone(PolicyValueNet)
        self.assertEqual(build_regret_auc_rows([]), [])

    def test_load_config_does_not_mutate_default_dp_cache(self) -> None:
        from experiments.single_card_tradeoff.core.config import (
            load_single_card_fsrs6_config,
        )

        original = OracleDPCacheConfig(enabled=True, cache_dir=Path("original"))
        set_default_oracle_dp_cache_config(original)
        try:
            load_single_card_fsrs6_config(
                argparse.Namespace(
                    env="fsrs6_default",
                    user_id=1,
                    benchmark_result=None,
                    benchmark_partition="0",
                    srs_benchmark_root=None,
                    button_usage=None,
                    dp_cache_enabled=False,
                    dp_cache_dir=Path("changed"),
                    refresh_dp_cache=True,
                )
            )
            self.assertEqual(resolve_oracle_dp_cache_config(), original)
        finally:
            set_default_oracle_dp_cache_config(OracleDPCacheConfig())

    def test_tradeoff_custom_scheduler_registry_covers_custom_schedulers(self) -> None:
        from experiments.single_card_tradeoff.cli import tradeoff
        from experiments.single_card_tradeoff.core.defaults import (
            FSRS6_ORACLE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INFINITE_SCHEDULER,
            FSRS6_ORACLE_INTERVAL_BILINEAR_ACTION_SCHEDULER,
            FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INTERVAL_SCHEDULER,
            FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER,
            FSRS6_ORACLE_SCHEDULER,
            FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER,
            UVFA_PPO_RNN_INTERVAL_SCHEDULER,
            UVFA_PPO_SCHEDULER,
        )

        self.assertEqual(
            set(tradeoff._CUSTOM_SINGLE_USER_RUNNERS),
            {
                FSRS6_ORACLE_SCHEDULER,
                FSRS6_ORACLE_INFINITE_SCHEDULER,
                FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER,
                FSRS6_ORACLE_INTERVAL_SCHEDULER,
                FSRS6_ORACLE_INTERVAL_BILINEAR_ACTION_SCHEDULER,
                FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER,
                FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
                FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
                FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER,
                UVFA_PPO_SCHEDULER,
                FSRS6_ORACLE_DISTILL_SCHEDULER,
                UVFA_PPO_RNN_INTERVAL_SCHEDULER,
            },
        )
        self.assertEqual(
            [evaluator.__name__ for evaluator in tradeoff._SCHEDULER_EVALUATORS],
            [
                "_custom_scheduler_evaluator",
                "_adr_scheduler_evaluator",
                "_standard_scheduler_evaluator",
            ],
        )
        self.assertIn("fsrs6", tradeoff._VECTORIZED_BATCH_SCHEDULERS)
        self.assertIn("memrise", tradeoff._VECTORIZED_BATCH_SCHEDULERS)

    def test_package_root_contains_only_package_entrypoints(self) -> None:
        self.assertEqual(
            sorted(path.name for path in SINGLE_CARD_ROOT.glob("*.py")),
            ["__init__.py", "__main__.py"],
        )

    def test_library_layers_do_not_import_cli_layer(self) -> None:
        offenders = []
        for layer in ("core", "models", "oracles"):
            for path in (SINGLE_CARD_ROOT / layer).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                if _imports_cli_layer(tree) or _dynamically_imports_cli_layer(tree):
                    offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

        self.assertEqual(offenders, [])


def _imports_cli_layer(tree: ast.AST) -> bool:
    cli_prefix = "experiments.single_card_tradeoff.cli"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == cli_prefix or node.module.startswith(f"{cli_prefix}."):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == cli_prefix or alias.name.startswith(f"{cli_prefix}."):
                    return True
    return False


def _dynamically_imports_cli_layer(tree: ast.AST) -> bool:
    cli_prefix = "experiments.single_card_tradeoff.cli"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "import_module":
            continue
        if not node.args:
            continue
        target = node.args[0]
        if not isinstance(target, ast.Constant) or not isinstance(target.value, str):
            continue
        if target.value == cli_prefix or target.value.startswith(f"{cli_prefix}."):
            return True
    return False


if __name__ == "__main__":
    unittest.main()
