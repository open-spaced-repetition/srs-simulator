from __future__ import annotations

from collections.abc import Callable

from experiments.single_card_tradeoff.core import tradeoff_runner as _runner
from experiments.single_card_tradeoff.core import tradeoff_args as _tradeoff_args
from experiments.single_card_tradeoff.core.tradeoff_runner import *  # noqa: F403
from typing import Any

_parse_user_ids_csv = _tradeoff_args._parse_user_ids_csv
_resolve_user_ids = _tradeoff_args._resolve_user_ids
_resolve_user_policy_template = _tradeoff_args._resolve_user_policy_template
_run_specs = _tradeoff_args._run_specs

_CUSTOM_SINGLE_USER_RUNNERS: dict[str, Callable[..., list[dict[str, Any]]]] = (
    _runner._CUSTOM_SINGLE_USER_RUNNERS
)
_SCHEDULER_EVALUATORS = _runner._SCHEDULER_EVALUATORS
_VECTORIZED_BATCH_SCHEDULERS = _runner._VECTORIZED_BATCH_SCHEDULERS
_load_fsrs6_adr_policy_specs = _runner._load_fsrs6_adr_policy_specs
_run_fsrs6_adr = _runner._run_fsrs6_adr
_build_regret_auc_rows = _runner._build_regret_auc_rows
_write_csv = _runner._write_csv


def __getattr__(name: str) -> object:
    return getattr(_runner, name)


def main() -> None:
    _runner.main()


if __name__ == "__main__":
    main()
