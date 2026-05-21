from __future__ import annotations

import argparse
import ast
import importlib
import sys
import unittest
from pathlib import Path

from experiments.single_card_tradeoff.defaults import DEFAULT_TARGET_RETENTIONS
from experiments.single_card_tradeoff.oracle_dp_cache import (
    OracleDPCacheConfig,
    resolve_oracle_dp_cache_config,
    set_default_oracle_dp_cache_config,
)
from experiments.single_card_tradeoff.policy_net import PolicyValueNet
from experiments.single_card_tradeoff.results import build_regret_auc_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_CARD_ROOT = PROJECT_ROOT / "experiments" / "single_card_tradeoff"


class SingleCardArchitectureBoundaryTests(unittest.TestCase):
    def test_package_defaults_do_not_import_tradeoff(self) -> None:
        sys.modules.pop("experiments.single_card_tradeoff.tradeoff", None)
        package = importlib.import_module("experiments.single_card_tradeoff")

        self.assertEqual(package.DEFAULT_TARGET_RETENTIONS, DEFAULT_TARGET_RETENTIONS)
        self.assertNotIn("experiments.single_card_tradeoff.tradeoff", sys.modules)

    def test_extracted_modules_expose_existing_entry_points(self) -> None:
        self.assertTrue(DEFAULT_TARGET_RETENTIONS)
        self.assertIsNotNone(PolicyValueNet)
        self.assertEqual(build_regret_auc_rows([]), [])

    def test_load_config_does_not_mutate_default_dp_cache(self) -> None:
        from experiments.single_card_tradeoff.config import (
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
        from experiments.single_card_tradeoff import tradeoff
        from experiments.single_card_tradeoff.defaults import (
            FSRS6_ORACLE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INFINITE_SCHEDULER,
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

    def test_single_card_modules_have_no_internal_import_cycles(self) -> None:
        edges = _single_card_import_edges()
        self.assertEqual(_strongly_connected_components(edges), [])
        self.assertEqual(
            sorted(
                source for source, targets in edges.items() if "tradeoff" in targets
            ),
            ["__main__"],
        )


def _single_card_import_edges() -> dict[str, set[str]]:
    module_names = {
        path.stem if path.name != "__init__.py" else "__init__"
        for path in SINGLE_CARD_ROOT.glob("*.py")
    }
    edges = {name: set[str]() for name in module_names}
    for path in SINGLE_CARD_ROOT.glob("*.py"):
        source = path.stem if path.name != "__init__.py" else "__init__"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                _add_import_from_edges(edges, module_names, source, node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    _add_import_edge(edges, module_names, source, alias.name)
    return edges


def _add_import_from_edges(
    edges: dict[str, set[str]],
    module_names: set[str],
    source: str,
    node: ast.ImportFrom,
) -> None:
    module_name = node.module
    if module_name is None:
        return
    if module_name == "experiments.single_card_tradeoff":
        for alias in node.names:
            if alias.name in module_names:
                edges[source].add(alias.name)
        return
    _add_import_edge(edges, module_names, source, module_name)


def _add_import_edge(
    edges: dict[str, set[str]],
    module_names: set[str],
    source: str,
    module_name: str,
) -> None:
    prefix = "experiments.single_card_tradeoff."
    if not module_name.startswith(prefix):
        return
    target = module_name.removeprefix(prefix).split(".", maxsplit=1)[0]
    if target in module_names:
        edges[source].add(target)


def _strongly_connected_components(edges: dict[str, set[str]]) -> list[list[str]]:
    index_by_node: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        index_by_node[node] = len(index_by_node)
        lowlink[node] = index_by_node[node]
        stack.append(node)
        on_stack.add(node)
        for target in edges[node]:
            if target not in edges:
                continue
            if target not in index_by_node:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], index_by_node[target])

        if lowlink[node] != index_by_node[node]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        if len(component) > 1:
            components.append(sorted(component))

    for node in sorted(edges):
        if node not in index_by_node:
            visit(node)
    return sorted(components)


if __name__ == "__main__":
    unittest.main()
