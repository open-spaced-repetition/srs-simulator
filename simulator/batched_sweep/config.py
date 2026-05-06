from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_END_RETENTION,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_PRIORITY,
    DEFAULT_RETENTION_STEP,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
    DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    DEFAULT_START_RETENTION,
)
from simulator.batched_sweep.utils import dr_values


@dataclass(frozen=True, slots=True)
class BatchedSweepConfig:
    path: Path
    args: argparse.Namespace
    envs: tuple[str, ...]
    schedulers: tuple[str, ...]

    @classmethod
    def from_toml(
        cls, path: str | Path, *, repo_root: Path | None = None
    ) -> BatchedSweepConfig:
        config_path = Path(path).expanduser()
        base_path = repo_root if repo_root is not None else config_path.parent
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("Batched sweep config must be a TOML table.")
        schema_version = raw.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"schema_version must be 1, got {schema_version!r}.")

        users = _table(raw, "users", required=True)
        user_ids, start_user, end_user = _parse_users(users)

        sweep = _table(raw, "sweep", required=False)
        envs = tuple(_str_list(_table_value(raw, sweep, "envs", ["lstm"]), "envs"))
        schedulers = tuple(
            _str_list(_table_value(raw, sweep, "schedulers", ["fsrs6"]), "schedulers")
        )
        if not envs:
            raise ValueError("sweep.envs must not be empty.")
        if not schedulers:
            raise ValueError("sweep.schedulers must not be empty.")

        retention = _table(raw, "retention", required=False)
        start_retention = _float(
            retention.get("start", DEFAULT_START_RETENTION), "retention.start"
        )
        end_retention = _float(
            retention.get("end", DEFAULT_END_RETENTION), "retention.end"
        )
        step = _float(retention.get("step", DEFAULT_RETENTION_STEP), "retention.step")
        retention_values = dr_values(start_retention, end_retention, step)
        if any(value <= 0.0 or value >= 1.0 for value in retention_values):
            raise ValueError("Retention grid values must satisfy 0 < value < 1.")

        simulation = _table(raw, "simulation", required=False)
        execution = _table(raw, "execution", required=False)
        paths = _table(raw, "paths", required=False)
        logging_config = _table(raw, "logging", required=False)
        short_term = _table(raw, "short_term", required=False)
        sa_fsrs6 = _table(raw, "sa_fsrs6", required=False)

        args = argparse.Namespace(
            config=config_path,
            user_ids=list(user_ids) if user_ids is not None else None,
            start_user=start_user,
            end_user=end_user,
            batch_size=_int(execution.get("batch_size", 1000), "execution.batch_size"),
            max_lanes_per_batch=_optional_int(
                execution.get("max_lanes_per_batch"),
                "execution.max_lanes_per_batch",
            ),
            env=",".join(envs),
            sched=",".join(schedulers),
            start_retention=start_retention,
            end_retention=end_retention,
            step=step,
            days=_int(simulation.get("days", DEFAULT_DAYS), "simulation.days"),
            deck=_int(simulation.get("deck", DEFAULT_DECK_SIZE), "simulation.deck"),
            learn_limit=_int(
                simulation.get("learn_limit", DEFAULT_LEARN_LIMIT),
                "simulation.learn_limit",
            ),
            review_limit=_optional_int(
                simulation.get("review_limit", DEFAULT_REVIEW_LIMIT),
                "simulation.review_limit",
            ),
            cost_limit_minutes=_optional_float(
                simulation.get("cost_limit_minutes", DEFAULT_COST_LIMIT_MINUTES),
                "simulation.cost_limit_minutes",
            ),
            seed=_int(simulation.get("seed", DEFAULT_SEED), "simulation.seed"),
            priority=_choice(
                simulation.get("priority", DEFAULT_PRIORITY),
                "simulation.priority",
                {"review-first", "new-first"},
            ),
            scheduler_priority=_str(
                simulation.get("scheduler_priority", DEFAULT_SCHEDULER_PRIORITY),
                "simulation.scheduler_priority",
            ),
            fuzz=_bool(simulation.get("fuzz", False), "simulation.fuzz"),
            button_usage=_optional_path(
                paths.get("button_usage", DEFAULT_BUTTON_USAGE_PATH),
                "paths.button_usage",
                base_path=base_path,
            ),
            benchmark_result=_optional_str(
                paths.get("benchmark_result"), "paths.benchmark_result"
            ),
            benchmark_partition=_str(
                paths.get("benchmark_partition", "0"), "paths.benchmark_partition"
            ),
            srs_benchmark_root=_optional_path(
                paths.get("srs_benchmark_root"),
                "paths.srs_benchmark_root",
                base_path=base_path,
            ),
            log_dir=_optional_path(
                paths.get("log_dir"),
                "paths.log_dir",
                base_path=base_path,
            ),
            log_layout=_choice(
                logging_config.get("log_layout", "user"),
                "logging.log_layout",
                {"user", "sweep"},
            ),
            no_log=_bool(logging_config.get("no_log", False), "logging.no_log"),
            no_progress=_bool(
                logging_config.get("no_progress", False), "logging.no_progress"
            ),
            diagnostic_csv_logs=_bool(
                logging_config.get("diagnostic_csv_logs", False),
                "logging.diagnostic_csv_logs",
            ),
            short_term_source=_optional_choice(
                short_term.get("source"),
                "short_term.source",
                {"steps", "sched"},
            ),
            learning_steps=_optional_str(
                short_term.get("learning_steps"), "short_term.learning_steps"
            ),
            relearning_steps=_optional_str(
                short_term.get("relearning_steps"), "short_term.relearning_steps"
            ),
            short_term_threshold=_float(
                short_term.get("threshold", 0.5), "short_term.threshold"
            ),
            short_term_loops_limit=_optional_int(
                short_term.get("loops_limit", DEFAULT_SHORT_TERM_LOOPS_LIMIT),
                "short_term.loops_limit",
            ),
            torch_device=_optional_str(
                execution.get("torch_device"), "execution.torch_device"
            ),
            cuda_devices=_cuda_devices(execution.get("cuda_devices")),
            dry_run=_bool(execution.get("dry_run", False), "execution.dry_run"),
            sa_fsrs6_policy=_optional_path(
                sa_fsrs6.get("policy"),
                "sa_fsrs6.policy",
                base_path=base_path,
            ),
            sa_fsrs6_policy_root=_optional_path(
                sa_fsrs6.get("policy_root"),
                "sa_fsrs6.policy_root",
                base_path=base_path,
            ),
            sa_fsrs6_train_run_root=_optional_path(
                sa_fsrs6.get("train_run_root"),
                "sa_fsrs6.train_run_root",
                base_path=base_path,
            ),
            sa_fsrs6_policy_manifest=_optional_path(
                sa_fsrs6.get("policy_manifest"),
                "sa_fsrs6.policy_manifest",
                base_path=base_path,
            ),
            sa_fsrs6_lambda_values=_optional_float_list(
                sa_fsrs6.get("lambda_values"),
                "sa_fsrs6.lambda_values",
            ),
        )
        return cls(path=config_path, args=args, envs=envs, schedulers=schedulers)


def load_batched_sweep_config(
    path: str | Path, *, repo_root: Path | None = None
) -> BatchedSweepConfig:
    return BatchedSweepConfig.from_toml(path, repo_root=repo_root)


def _parse_users(
    users: Mapping[str, Any],
) -> tuple[tuple[int, ...] | None, int, int]:
    has_ids = "ids" in users
    has_range = "start" in users or "end" in users
    if has_ids and has_range:
        raise ValueError("Use either users.ids or users.start/users.end, not both.")
    if not has_ids and not has_range:
        raise ValueError("Configure exactly one of users.ids or users.start/users.end.")
    if has_ids:
        user_ids = tuple(_int_list(users["ids"], "users.ids"))
        if not user_ids:
            raise ValueError("users.ids must not be empty.")
        if len(set(user_ids)) != len(user_ids):
            raise ValueError("users.ids must not contain duplicates.")
        return user_ids, int(user_ids[0]), int(user_ids[-1])
    if "start" not in users or "end" not in users:
        raise ValueError("users.start and users.end must be configured together.")
    start_user = _int(users["start"], "users.start")
    end_user = _int(users["end"], "users.end")
    if start_user < 1:
        raise ValueError("users.start must be >= 1.")
    if end_user < start_user:
        raise ValueError("users.end must be >= users.start.")
    return None, start_user, end_user


def _table(
    raw: Mapping[str, Any],
    name: str,
    *,
    required: bool,
) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise ValueError(f"[{name}] table is required.")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[{name}] must be a TOML table.")
    return value


def _table_value(
    raw: Mapping[str, Any],
    table: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in table:
        return table[key]
    return raw.get(key, default)


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _int(value, field_name)


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _float(value, field_name)


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _str(value, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return bool(value)


def _choice(value: Any, field_name: str, choices: set[str]) -> str:
    item = _str(value, field_name)
    if item not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}.")
    return item


def _optional_choice(
    value: Any,
    field_name: str,
    choices: set[str],
) -> str | None:
    if value is None:
        return None
    return _choice(value, field_name, choices)


def _optional_path(value: Any, field_name: str, *, base_path: Path) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    path = Path(_str(value, field_name)).expanduser()
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


def _str_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [
            _str(item, f"{field_name}[{index}]") for index, item in enumerate(value)
        ]
    raise ValueError(f"{field_name} must be a string or array of strings.")


def _int_list(value: Any, field_name: str) -> list[int]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array of integers.")
    return [_int(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _optional_float_list(value: Any, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(float(item) for item in items)
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array of numbers.")
    return tuple(
        _float(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _cuda_devices(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        parts: list[str] = []
        for index, item in enumerate(value):
            if isinstance(item, int) and not isinstance(item, bool):
                parts.append(str(item))
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
            else:
                raise ValueError(
                    f"execution.cuda_devices[{index}] must be an integer or string."
                )
        return ",".join(parts)
    raise ValueError("execution.cuda_devices must be a string or array.")


__all__ = ["BatchedSweepConfig", "load_batched_sweep_config"]
