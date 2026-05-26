from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.scheduler_catalog import PolicySource, schedulers_for_policy_source
from simulator.batched_sweep.fsrs6_cost_adr_policy import DEFAULT_COST_WEIGHTS
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_END_RETENTION,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_MAX_LANES_PER_BATCH,
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
        if _looks_like_experiment_config(raw):
            raw = _adapt_experiment_config(raw, base_path=base_path)

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
        raw_env_batch_overrides = execution.get("env_overrides")
        env_batch_overrides_field_name = "execution.env_overrides"
        if raw_env_batch_overrides is None:
            raw_env_batch_overrides = sweep.get("env_overrides")
            env_batch_overrides_field_name = "sweep.env_overrides"
        paths = _table(raw, "paths", required=False)
        logging_config = _table(raw, "logging", required=False)
        short_term = _table(raw, "short_term", required=False)
        fsrs6_adr = _table(raw, "fsrs6_adr", required=False)
        fsrs6_cost_adr = _table(raw, "fsrs6_cost_adr", required=False)
        fsrs6_oracle_distill = _table(
            raw, "fsrs6_oracle_stationary_finite_distill", required=False
        )
        fsrs6_ap = _table(raw, "fsrs6_ap", required=False)
        anki_sm2_ap = _table(raw, "anki_sm2_ap", required=False)
        fsrs6 = _table(raw, "fsrs6", required=False)

        args = argparse.Namespace(
            config=config_path,
            user_ids=list(user_ids) if user_ids is not None else None,
            start_user=start_user,
            end_user=end_user,
            batch_size=_optional_int(
                execution.get("batch_size"), "execution.batch_size"
            ),
            max_lanes_per_batch=_optional_int(
                execution.get("max_lanes_per_batch", DEFAULT_MAX_LANES_PER_BATCH),
                "execution.max_lanes_per_batch",
            ),
            env_batch_overrides=_env_batch_overrides(
                raw_env_batch_overrides,
                env_batch_overrides_field_name,
            ),
            env=",".join(envs),
            sched=",".join(schedulers),
            run_id=_optional_str(_table_value(raw, sweep, "run_id", None), "run_id"),
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
            review_markov_transition=_bool(
                simulation.get("review_markov_transition", False),
                "simulation.review_markov_transition",
            ),
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
            fsrs6_adr_policy=_optional_path(
                fsrs6_adr.get("policy"),
                "fsrs6_adr.policy",
                base_path=base_path,
            ),
            fsrs6_adr_policy_root=_optional_path(
                fsrs6_adr.get("policy_root"),
                "fsrs6_adr.policy_root",
                base_path=base_path,
            ),
            fsrs6_adr_train_run_root=_optional_path(
                fsrs6_adr.get("train_run_root"),
                "fsrs6_adr.train_run_root",
                base_path=base_path,
            ),
            fsrs6_adr_policy_manifest=_optional_path(
                fsrs6_adr.get("policy_manifest"),
                "fsrs6_adr.policy_manifest",
                base_path=base_path,
            ),
            fsrs6_adr_lambda_values=_optional_float_list(
                fsrs6_adr.get("lambda_values"),
                "fsrs6_adr.lambda_values",
            ),
            fsrs6_cost_adr_policy=_optional_path(
                fsrs6_cost_adr.get("policy"),
                "fsrs6_cost_adr.policy",
                base_path=base_path,
            ),
            fsrs6_cost_adr_policy_root=_optional_path(
                fsrs6_cost_adr.get("policy_root"),
                "fsrs6_cost_adr.policy_root",
                base_path=base_path,
            ),
            fsrs6_cost_adr_train_run_root=_optional_path(
                fsrs6_cost_adr.get("train_run_root"),
                "fsrs6_cost_adr.train_run_root",
                base_path=base_path,
            ),
            fsrs6_cost_adr_policy_manifest=_optional_path(
                fsrs6_cost_adr.get("policy_manifest"),
                "fsrs6_cost_adr.policy_manifest",
                base_path=base_path,
            ),
            fsrs6_cost_adr_cost_weights=_optional_float_list(
                fsrs6_cost_adr.get("cost_weights"),
                "fsrs6_cost_adr.cost_weights",
            )
            or DEFAULT_COST_WEIGHTS,
            fsrs6_oracle_stationary_finite_distill_policy=_optional_path(
                fsrs6_oracle_distill.get("policy"),
                "fsrs6_oracle_stationary_finite_distill.policy",
                base_path=base_path,
            ),
            fsrs6_oracle_stationary_finite_distill_policy_root=_optional_path(
                fsrs6_oracle_distill.get("policy_root"),
                "fsrs6_oracle_stationary_finite_distill.policy_root",
                base_path=base_path,
            ),
            fsrs6_oracle_stationary_finite_distill_train_run_root=_optional_path(
                fsrs6_oracle_distill.get("train_run_root"),
                "fsrs6_oracle_stationary_finite_distill.train_run_root",
                base_path=base_path,
            ),
            fsrs6_oracle_stationary_finite_distill_policy_manifest=_optional_path(
                fsrs6_oracle_distill.get("policy_manifest"),
                "fsrs6_oracle_stationary_finite_distill.policy_manifest",
                base_path=base_path,
            ),
            fsrs6_ap_policy=_optional_path(
                fsrs6_ap.get("policy"),
                "fsrs6_ap.policy",
                base_path=base_path,
            ),
            fsrs6_ap_policy_root=_optional_path(
                fsrs6_ap.get("policy_root"),
                "fsrs6_ap.policy_root",
                base_path=base_path,
            ),
            fsrs6_ap_train_run_root=_optional_path(
                fsrs6_ap.get("train_run_root"),
                "fsrs6_ap.train_run_root",
                base_path=base_path,
            ),
            fsrs6_ap_policy_manifest=_optional_path(
                fsrs6_ap.get("policy_manifest"),
                "fsrs6_ap.policy_manifest",
                base_path=base_path,
            ),
            fsrs6_ap_lambda_values=_optional_float_list(
                fsrs6_ap.get("lambda_values"),
                "fsrs6_ap.lambda_values",
            ),
            anki_sm2_ap_policy=_optional_path(
                anki_sm2_ap.get("policy"),
                "anki_sm2_ap.policy",
                base_path=base_path,
            ),
            anki_sm2_ap_policy_root=_optional_path(
                anki_sm2_ap.get("policy_root"),
                "anki_sm2_ap.policy_root",
                base_path=base_path,
            ),
            anki_sm2_ap_train_run_root=_optional_path(
                anki_sm2_ap.get("train_run_root"),
                "anki_sm2_ap.train_run_root",
                base_path=base_path,
            ),
            anki_sm2_ap_policy_manifest=_optional_path(
                anki_sm2_ap.get("policy_manifest"),
                "anki_sm2_ap.policy_manifest",
                base_path=base_path,
            ),
            fsrs6_dr_manifest=_optional_path(
                fsrs6.get("dr_manifest"),
                "fsrs6.dr_manifest",
                base_path=base_path,
            ),
        )
        return cls(path=config_path, args=args, envs=envs, schedulers=schedulers)


def load_batched_sweep_config(
    path: str | Path, *, repo_root: Path | None = None
) -> BatchedSweepConfig:
    return BatchedSweepConfig.from_toml(path, repo_root=repo_root)


def _looks_like_experiment_config(raw: Mapping[str, Any]) -> bool:
    users = raw.get("users")
    sweep = raw.get("sweep")
    return (
        isinstance(users, Mapping)
        and "train" in users
        and isinstance(sweep, Mapping)
        and ("output_root" in raw or raw.get("family") == "rl_scheduler")
    )


def _adapt_experiment_config(
    raw: Mapping[str, Any], *, base_path: Path
) -> dict[str, Any]:
    users = _table(raw, "users", required=True)
    sweep = _table(raw, "sweep", required=True)
    simulation = _table(raw, "simulation", required=False)
    performance = _table(raw, "performance", required=False)
    training = _table(raw, "training", required=False)
    training_policy_search = _nested_table(training, "policy_search", required=False)

    adapted_simulation = dict(simulation)
    adapted_simulation["seed"] = raw.get(
        "seed",
        adapted_simulation.get("seed", DEFAULT_SEED),
    )

    paths = {
        "log_dir": sweep.get("log_dir"),
        "benchmark_result": sweep.get("benchmark_result"),
        "benchmark_partition": sweep.get("benchmark_partition", "0"),
        "srs_benchmark_root": sweep.get("srs_benchmark_root"),
        "button_usage": sweep.get("button_usage"),
    }
    paths = {key: value for key, value in paths.items() if value is not None}

    execution = {
        "batch_size": sweep.get("batch_size"),
        "max_lanes_per_batch": sweep.get("max_lanes_per_batch"),
        "env_overrides": sweep.get("env_overrides"),
        "torch_device": sweep.get("torch_device", performance.get("device")),
        "cuda_devices": sweep.get("cuda_devices"),
        "dry_run": sweep.get("dry_run", False),
    }
    execution = {key: value for key, value in execution.items() if value is not None}

    short_term = {
        "source": adapted_simulation.get("short_term_source"),
        "learning_steps": training_policy_search.get("learning_steps"),
        "relearning_steps": training_policy_search.get("relearning_steps"),
        "threshold": training_policy_search.get("short_term_threshold", 0.5),
        "loops_limit": training_policy_search.get("short_term_loops_limit"),
    }
    short_term = {key: value for key, value in short_term.items() if value is not None}

    schedulers = _str_list(sweep.get("schedulers", ["fsrs6"]), "sweep.schedulers")
    scheduler_names = {item.split("@", 1)[0] for item in schedulers}
    uses_fsrs6_adr_policy_source = bool(
        scheduler_names & schedulers_for_policy_source(PolicySource.FSRS6_ADR)
    )
    uses_fsrs6_cost_adr_policy_source = bool(
        scheduler_names & schedulers_for_policy_source(PolicySource.FSRS6_COST_ADR)
    )
    uses_fsrs6_oracle_distill_policy_source = (
        "fsrs6_oracle_stationary_finite_distill" in scheduler_names
    )
    lambda_grid = training.get("lambda_grid")
    if _training_uses_portfolio_trainer(training):
        lambda_grid = None
    default_train_run_root = _infer_experiment_train_run_root(
        raw,
        sweep=sweep,
        base_path=base_path,
    )
    run_id = _experiment_run_id(
        raw,
        sweep=sweep,
        default_train_run_root=default_train_run_root,
    )
    fsrs6_adr = _adapt_experiment_policy_source(
        sweep,
        prefix="fsrs6_adr",
        lambda_grid=lambda_grid,
        default_train_run_root=default_train_run_root
        if uses_fsrs6_adr_policy_source
        else None,
    )
    fsrs6_cost_adr = _adapt_experiment_policy_source(
        sweep,
        prefix="fsrs6_cost_adr",
        lambda_grid=None,
        default_train_run_root=default_train_run_root
        if uses_fsrs6_cost_adr_policy_source
        else None,
    )
    if "cost_weights" not in fsrs6_cost_adr:
        fsrs6_cost_adr["cost_weights"] = sweep.get(
            "fsrs6_cost_adr_cost_weights",
            list(DEFAULT_COST_WEIGHTS),
        )
    fsrs6_oracle_distill = _adapt_experiment_policy_source(
        sweep,
        prefix="fsrs6_oracle_stationary_finite_distill",
        lambda_grid=None,
        default_train_run_root=default_train_run_root
        if uses_fsrs6_oracle_distill_policy_source
        else None,
    )
    fsrs6_ap = _adapt_experiment_policy_source(
        sweep,
        prefix="fsrs6_ap",
        lambda_grid=lambda_grid,
        default_train_run_root=default_train_run_root
        if "fsrs6_ap" in scheduler_names
        else None,
    )
    anki_sm2_ap = _adapt_experiment_policy_source(
        sweep,
        prefix="anki_sm2_ap",
        lambda_grid=None,
        default_train_run_root=default_train_run_root
        if "anki_sm2_ap" in scheduler_names
        else None,
    )
    baseline_dr_selection = _table(raw, "baseline_dr_selection", required=False)
    fsrs6 = {}
    if baseline_dr_selection.get("manifest") is not None:
        fsrs6["dr_manifest"] = baseline_dr_selection["manifest"]

    return {
        "schema_version": raw.get("schema_version", 1),
        "users": {"ids": users.get("train")},
        "sweep": {
            "envs": sweep.get("envs", ["lstm"]),
            "schedulers": schedulers,
            "run_id": run_id,
        },
        "retention": {
            "start": sweep.get("start_retention", DEFAULT_START_RETENTION),
            "end": sweep.get("end_retention", DEFAULT_END_RETENTION),
            "step": sweep.get("step", DEFAULT_RETENTION_STEP),
        },
        "simulation": adapted_simulation,
        "paths": paths,
        "execution": execution,
        "logging": {
            "log_layout": sweep.get("log_layout", "user"),
            "no_log": sweep.get("no_log", False),
            "no_progress": False,
            "diagnostic_csv_logs": performance.get("diagnostic_csv_logs", False),
        },
        "short_term": short_term,
        "fsrs6": fsrs6,
        "fsrs6_adr": fsrs6_adr,
        "fsrs6_cost_adr": fsrs6_cost_adr,
        "fsrs6_oracle_stationary_finite_distill": fsrs6_oracle_distill,
        "fsrs6_ap": fsrs6_ap,
        "anki_sm2_ap": anki_sm2_ap,
    }


def _adapt_experiment_policy_source(
    sweep: Mapping[str, Any],
    *,
    prefix: str,
    lambda_grid: Any,
    default_train_run_root: Path | None,
) -> dict[str, Any]:
    adapted: dict[str, Any] = {}
    field_map = {
        "policy": f"{prefix}_policy",
        "policy_root": f"{prefix}_policy_root",
        "train_run_root": f"{prefix}_train_run_root",
        "policy_manifest": f"{prefix}_policy_manifest",
        "lambda_values": f"{prefix}_lambda_values",
    }
    for target, source in field_map.items():
        if source in sweep:
            adapted[target] = sweep[source]
    has_source = any(
        key in adapted for key in ("policy_root", "train_run_root", "policy_manifest")
    )
    if not has_source and default_train_run_root is not None:
        adapted["train_run_root"] = str(default_train_run_root)
    if "lambda_values" not in adapted and lambda_grid is not None:
        adapted["lambda_values"] = lambda_grid
    return adapted


def _training_uses_portfolio_trainer(training: Mapping[str, Any]) -> bool:
    if "portfolio" in training and "optimizer" not in training:
        return True
    batch = training.get("batch")
    if isinstance(batch, Mapping):
        trainer = batch.get("trainer")
        if trainer in {
            "fsrs6_cost_adr_cmaes",
            "fsrs6_adr_portfolio",
            "fsrs6_oracle_stationary_finite_distill_portfolio",
            "fsrs6_ap_portfolio",
            "anki_sm2_ap_portfolio",
        }:
            return True
    command_template = training.get("command_template", [])
    if isinstance(command_template, str) or not isinstance(command_template, Sequence):
        return False
    script_names = {Path(str(item)).name for item in command_template}
    return bool(
        {
            "train_cmaes_fsrs6_cost_adr.py",
            "train_fsrs6_adr_portfolio.py",
            "train_fsrs6_oracle_stationary_finite_distill_portfolio.py",
            "train_fsrs6_ap_portfolio.py",
            "train_anki_sm2_ap_portfolio.py",
        }
        & script_names
    )


def _experiment_run_id(
    raw: Mapping[str, Any],
    *,
    sweep: Mapping[str, Any],
    default_train_run_root: Path | None,
) -> str | None:
    run_id_raw = sweep.get("run_id", raw.get("run_id"))
    if run_id_raw is not None:
        return _str(run_id_raw, "run_id")
    if default_train_run_root is not None:
        return default_train_run_root.name
    return None


def _infer_experiment_train_run_root(
    raw: Mapping[str, Any],
    *,
    sweep: Mapping[str, Any],
    base_path: Path,
) -> Path | None:
    output_root_raw = raw.get("output_root")
    if output_root_raw is None:
        return None
    output_root = _optional_path(
        output_root_raw,
        "output_root",
        base_path=base_path,
    )
    if output_root is None or not output_root.is_dir():
        return None

    run_id_raw = sweep.get("run_id", raw.get("run_id"))
    if run_id_raw is not None:
        run_id = _str(run_id_raw, "run_id")
        run_root = output_root / run_id
        if _has_passed_training_summary(run_root):
            return run_root

    candidates = [
        child
        for child in output_root.iterdir()
        if child.is_dir() and _has_passed_training_summary(child)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            path.joinpath("train-overfit", "training_summary.json").stat().st_mtime,
            path.name,
        ),
    )


def _has_passed_training_summary(run_root: Path) -> bool:
    summary_path = run_root / "train-overfit" / "training_summary.json"
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    artifact_paths = payload.get("artifact_paths")
    return payload.get("passed") is True and isinstance(artifact_paths, list)


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


def _nested_table(
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


def _positive_optional_int(value: Any, field_name: str) -> int | None:
    result = _optional_int(value, field_name)
    if result is not None and result < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return result


def _env_batch_overrides(
    value: Any,
    field_name: str,
) -> dict[str, dict[str, int | None]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table.")
    overrides: dict[str, dict[str, int | None]] = {}
    for environment, raw in value.items():
        if not isinstance(environment, str) or not environment.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field_name}.{environment} must be a table.")
        config: dict[str, int | None] = {}
        for key, item in raw.items():
            if key not in {"batch_size", "max_lanes_per_batch"}:
                raise ValueError(
                    f"{field_name}.{environment} may contain only batch_size and "
                    "max_lanes_per_batch."
                )
            config[key] = _positive_optional_int(
                item,
                f"{field_name}.{environment}.{key}",
            )
        overrides[environment] = config
    return overrides


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
