from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import tomllib
from typing import Any

from experiments.single_card_tradeoff.core.reporting import REPO_ROOT, resolve_repo_path
from experiments.single_card_tradeoff.core.workflow_config import WorkflowTask
from simulator.batched_sweep.fsrs6_adr_policy import format_float_token


TASK_MODULES: Mapping[str, str] = {
    "tradeoff": "experiments.single_card_tradeoff.cli.tradeoff",
    "tradeoff_config": "experiments.single_card_tradeoff.cli.run_tradeoff_config",
    "stationary_finite_distill_multiuser": (
        "experiments.single_card_tradeoff.cli."
        "oracle_stationary_finite_distill_multiuser"
    ),
    "stationary_finite_exact_value_eval": (
        "experiments.single_card_tradeoff.cli.stationary_finite_exact_value_eval"
    ),
    "train_weight_control_analysis": (
        "experiments.single_card_tradeoff.cli.train_weight_control_analysis"
    ),
    "oracle_interval_distill": (
        "experiments.single_card_tradeoff.cli.oracle_interval_distill"
    ),
    "oracle_continuous_stationary_finite_distill": (
        "experiments.single_card_tradeoff.cli."
        "oracle_continuous_stationary_finite_distill"
    ),
    "oracle_interval_distill_hparam_search": (
        "experiments.single_card_tradeoff.cli.oracle_interval_distill_hparam_search"
    ),
    "low_param_direct_policy_search_multiuser": (
        "experiments.single_card_tradeoff.cli.low_param_direct_policy_search_multiuser"
    ),
    "fsrs6_adr_train_multiuser": (
        "experiments.single_card_tradeoff.cli.fsrs6_adr_train_multiuser"
    ),
    "oracle_distill_hparam_search": (
        "experiments.single_card_tradeoff.cli.oracle_distill_hparam_search"
    ),
    "oracle_policy_outputs": (
        "experiments.single_card_tradeoff.cli.oracle_policy_outputs"
    ),
    "oracle_stationary_finite_cpu_gpu_benchmark": (
        "experiments.single_card_tradeoff.cli."
        "oracle_stationary_finite_cpu_gpu_benchmark"
    ),
    "oracle_stationary_finite_multiuser_cpu_gpu_benchmark": (
        "experiments.single_card_tradeoff.cli."
        "oracle_stationary_finite_multiuser_cpu_gpu_benchmark"
    ),
    "stationary_finite_model_size_ablation": (
        "experiments.single_card_tradeoff.cli.stationary_finite_model_size_ablation"
    ),
    "oracle_stationary_finite_policy_viz": (
        "experiments.single_card_tradeoff.cli.oracle_stationary_finite_policy_viz"
    ),
    "uvfa_ppo_hparam_search": (
        "experiments.single_card_tradeoff.cli.uvfa_ppo_hparam_search"
    ),
    "generate_experiment_report": (
        "experiments.single_card_tradeoff.cli.generate_experiment_report"
    ),
}

REPEAT_OPTIONS = {
    "distill_policy",
    "direct_policy",
    "treatment",
}


def task_command(task: WorkflowTask) -> list[str]:
    module = TASK_MODULES.get(task.kind)
    if module is None:
        raise ValueError(f"Unknown single-card workflow task kind: {task.kind!r}")
    command = [sys.executable, "-m", module]
    if task.kind == "tradeoff_config":
        config_path = task.options.get("config")
        if config_path is None:
            config_path = _display_repo_path(task.config_path)
        command.extend(["--config", str(config_path)])
        for key, value in task.options.items():
            if key == "config":
                continue
            _append_option(command, key, value)
        return command
    for key, value in task.options.items():
        _append_option(command, key, value)
    return command


def task_record(task: WorkflowTask) -> dict[str, Any]:
    command = task_command(task)
    artifacts = expected_artifacts(task)
    return {
        "name": task.name,
        "stage": task.stage.value,
        "kind": task.kind,
        "description": task.description,
        "source": task.source,
        "enabled": task.enabled,
        "command": command,
        "expected_artifacts": [_display_repo_path(path) for path in artifacts],
    }


def expected_artifacts(task: WorkflowTask) -> tuple[Path, ...]:
    options = task.options
    if task.kind == "tradeoff_config":
        return _tradeoff_config_artifacts(
            Path(str(options.get("config", task.config_path)))
        )
    if task.kind == "tradeoff":
        return _tradeoff_artifacts(options)
    if task.kind == "stationary_finite_distill_multiuser":
        return _stationary_finite_distill_multiuser_artifacts(options)
    if task.kind == "stationary_finite_exact_value_eval":
        return _out_dir_artifacts(
            options,
            [
                "results.csv",
                "regret_auc.csv",
                "summary.csv",
                "mean_summary.csv",
                "metadata.json",
                "performance_summary.json",
                "gpu_monitor/summary.json",
            ],
        )
    if task.kind == "train_weight_control_analysis":
        return _out_dir_artifacts(
            options,
            [
                "train_weight_summary.csv",
                "train_weight_by_user.csv",
                "train_weight_direct_vs_exact.csv",
                "train_weight_user2_dense_loww.csv",
                "train_weight_user2_segment_auc.csv",
                "train_weight_user2_frontier_formal.png",
                "train_weight_user2_frontier_dense_loww.png",
            ],
        )
    if task.kind == "low_param_direct_policy_search_multiuser":
        return _out_dir_artifacts(
            options,
            [
                "results.csv",
                "summary.csv",
                "policy.pt",
                "metadata.json",
                "mean_summary.csv",
                "regret_auc.csv",
                "train_history.csv",
                "performance_summary.json",
                "gpu_monitor/summary.json",
            ],
        )
    if task.kind == "fsrs6_adr_train_multiuser":
        return _fsrs6_adr_train_multiuser_artifacts(options)
    if task.kind == "oracle_interval_distill":
        return _explicit_file_artifacts(options, ["model_out", "out"])
    if task.kind == "oracle_continuous_stationary_finite_distill":
        return _explicit_file_artifacts(options, ["model_out", "out"])
    if task.kind == "oracle_interval_distill_hparam_search":
        return _root_artifacts(
            "artifacts/single_card_tradeoff",
            [
                "fsrs6_oracle_interval_distill_hparam_summary.csv",
                "fsrs6_oracle_interval_distill_hparam_detail.csv",
                "fsrs6_oracle_interval_distill_hparam_models",
            ],
        )
    if task.kind == "oracle_distill_hparam_search":
        return _root_artifacts(
            "artifacts/single_card_tradeoff",
            [
                "fsrs6_oracle_distill_hparam_summary.csv",
                "fsrs6_oracle_distill_hparam_detail.csv",
                "fsrs6_oracle_distill_hparam_models",
            ],
        )
    if task.kind == "uvfa_ppo_hparam_search":
        return _root_artifacts(
            "artifacts/single_card_tradeoff",
            [
                "uvfa_ppo_hparam_search_summary.csv",
                "uvfa_ppo_hparam_search_detail.csv",
                "uvfa_ppo_hparam_search_models",
            ],
        )
    if task.kind == "oracle_policy_outputs":
        return _oracle_policy_outputs_artifacts(options)
    if task.kind in {
        "oracle_stationary_finite_cpu_gpu_benchmark",
        "oracle_stationary_finite_multiuser_cpu_gpu_benchmark",
    }:
        return _benchmark_artifacts(
            options, multiuser=task.kind.endswith("multiuser_cpu_gpu_benchmark")
        )
    if task.kind == "stationary_finite_model_size_ablation":
        return _model_size_ablation_artifacts(options)
    if task.kind == "oracle_stationary_finite_policy_viz":
        return _policy_viz_artifacts(options)
    if task.kind == "generate_experiment_report":
        return _report_artifacts(options)
    return ()


def profile_task(
    task: WorkflowTask, *, future_available_paths: set[Path] | None = None
) -> dict[str, Any]:
    future_available_paths = future_available_paths or set()
    artifacts = expected_artifacts(task)
    return {
        "name": task.name,
        "stage": task.stage.value,
        "kind": task.kind,
        "description": task.description,
        "command": task_command(task),
        "expected_output_paths": [_display_repo_path(path) for path in artifacts],
        "expected_outputs": len(artifacts),
        "available_expected_outputs": sum(
            1 for path in artifacts if path.exists() or path in future_available_paths
        ),
    }


def _append_option(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, list | tuple):
        if key in REPEAT_OPTIONS:
            for item in value:
                command.extend([flag, _format_value(item)])
            return
        command.extend([flag, ",".join(_format_value(item) for item in value)])
        return
    command.extend([flag, _format_value(value)])


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _tradeoff_config_artifacts(config_path: Path) -> tuple[Path, ...]:
    config_path = resolve_repo_path(config_path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        return ()
    root = outputs.get("root")
    if not isinstance(root, str):
        return ()
    return _root_artifacts(
        root,
        [
            "combined_results.csv",
            "combined_regret_auc.csv",
            "summary.csv",
            "mean_summary.csv",
            "same_target_time_saved_auc_by_user.png",
            "relative_time_saved_by_user.png",
            "span_coverage_by_user.png",
        ],
    )


def _tradeoff_artifacts(options: Mapping[str, Any]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for key in ("out", "regret_auc_out", "plot_path"):
        value = options.get(key)
        if isinstance(value, str):
            paths.append(resolve_repo_path(Path(value)))
    out = options.get("out")
    if isinstance(out, str):
        out_dir = resolve_repo_path(Path(out)).parent
        paths.append(out_dir / "performance_summary.json")
        paths.append(out_dir / "gpu_monitor" / "summary.json")
    return tuple(_dedupe(paths))


def _stationary_finite_distill_multiuser_artifacts(
    options: Mapping[str, Any],
) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        return ()
    if bool(options.get("eval_exact_vs_distill")):
        names = [
            "mean_summary.csv",
            "summary.csv",
            "regret_auc.csv",
            "performance_summary.json",
            "gpu_monitor/summary.json",
        ]
    else:
        names = [
            "train_summary.csv",
            "summary.csv",
            "regret_auc.csv",
            "performance_summary.json",
            "gpu_monitor/summary.json",
            "run_config.json",
        ]
    paths = [out_dir / name for name in names]
    if bool(options.get("per_user_models")):
        user_ids = _parse_user_ids(options.get("user_ids"))
        if user_ids:
            paths.append(out_dir / f"user_{user_ids[0]}_policy.pt")
            paths.append(out_dir / f"user_{user_ids[-1]}_policy.pt")
    return tuple(_dedupe(paths))


def _oracle_policy_outputs_artifacts(options: Mapping[str, Any]) -> tuple[Path, ...]:
    out = options.get("out")
    if isinstance(out, str):
        base = resolve_repo_path(Path(out))
        detail = options.get("detail_out")
        detail_path = (
            resolve_repo_path(Path(detail))
            if isinstance(detail, str)
            else base.with_name(base.stem + "_detail.csv")
        )
        return (base, detail_path, base.with_suffix(".png"))
    source = str(options.get("source", "rollout"))
    root = Path("artifacts/single_card_tradeoff/analysis")
    if source == "table":
        stem = "no_sub05_fsrs6_oracle_policy_outputs_table_1825"
    else:
        stem = "no_sub05_fsrs6_oracle_policy_outputs_rollout_1825_p2048"
    return _root_artifacts(
        root.as_posix(), [f"{stem}.csv", f"{stem}_detail.csv", f"{stem}.png"]
    )


def _benchmark_artifacts(
    options: Mapping[str, Any],
    *,
    multiuser: bool,
) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        default = (
            "oracle_stationary_finite_multiuser_cpu_gpu_benchmark"
            if multiuser
            else "oracle_stationary_finite_cpu_gpu_benchmark"
        )
        out_dir = resolve_repo_path(Path("artifacts/single_card_tradeoff") / default)
    paths = [out_dir / "summary.csv", out_dir / "runs.csv", out_dir / "metadata.json"]
    devices = _parse_csv_strings(options.get("devices")) or ["cpu", "cuda"]
    repeats = int(options.get("repeats", 1))
    for device in devices:
        repeat_root = out_dir / device / "repeat_1"
        if multiuser:
            paths.extend(
                [
                    repeat_root / "results.csv",
                    repeat_root / "summary.csv",
                    repeat_root / "train_summary.csv",
                    repeat_root / "performance_summary.json",
                ]
            )
        else:
            paths.extend(
                [
                    repeat_root / "results.csv",
                    repeat_root / "policy.pt",
                    repeat_root / "performance_summary.json",
                ]
            )
        if device == "cuda":
            paths.append(repeat_root / "gpu_monitor" / "summary.json")
    if repeats > 1:
        paths.append(out_dir / devices[0] / f"repeat_{repeats}")
    return tuple(_dedupe(paths))


def _model_size_ablation_artifacts(options: Mapping[str, Any]) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        out_dir = resolve_repo_path(
            Path("artifacts/single_card_tradeoff/stationary_finite_model_size_ablation")
        )
    prefix = str(options.get("summary_prefix", "model_size_ablation"))
    candidate_names = _model_size_candidates(options)
    paths = [
        out_dir / f"{prefix}_summary.csv",
        out_dir / f"{prefix}_seed_metrics.csv",
        out_dir / "performance_summary.json",
        out_dir / "gpu_monitor" / "summary.json",
    ]
    for name in candidate_names:
        paths.append(out_dir / f"{name}_policy.pt")
    return tuple(_dedupe(paths))


def _fsrs6_adr_train_multiuser_artifacts(
    options: Mapping[str, Any],
) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        out_dir = resolve_repo_path(
            Path(
                "artifacts/single_card_tradeoff/fsrs6_adr_single_card_direct_multiuser"
            )
        )
    paths = [
        out_dir / "summary.csv",
        out_dir / "train_history.csv",
        out_dir / "metadata.json",
        out_dir / "policy_manifest.toml",
        out_dir / "performance_summary.json",
        out_dir / "gpu_monitor" / "summary.json",
    ]
    user_ids = _parse_user_ids(options.get("user_ids"))
    cost_weights = _parse_csv_floats(options.get("cost_weights"))
    if user_ids and cost_weights:
        for user_id in user_ids:
            for cost_weight in cost_weights:
                job_root = (
                    out_dir
                    / "train-overfit"
                    / "train_outputs"
                    / f"user_{user_id}"
                    / f"lambda_{format_float_token(cost_weight)}"
                )
                paths.extend(
                    [
                        job_root / "policy.json",
                        job_root / "metadata.json",
                        job_root / "metrics.json",
                    ]
                )
    return tuple(_dedupe(paths))


def _policy_viz_artifacts(options: Mapping[str, Any]) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        out_dir = resolve_repo_path(
            Path("artifacts/single_card_tradeoff/stationary_finite_policy_viz")
        )
    names = [
        "action_summary.csv",
        "binned_actions.csv",
        "grid_actions.csv",
        "findings.md",
        "action_distribution.png",
        "policy_heatmaps.png",
    ]
    if _option_is_set(options, "distill_policy"):
        names.extend(
            [
                "distill_action_summary.csv",
                "distill_grid_actions.csv",
                "distill_binned_actions.csv",
                "distill_exact_comparison.csv",
                "distill_action_distribution.png",
                "distill_policy_heatmaps.png",
                "distill_exact_difference_heatmaps.png",
            ]
        )
    return tuple(_dedupe([out_dir / name for name in names]))


def _model_size_candidates(options: Mapping[str, Any]) -> list[str]:
    raw_candidates = options.get("candidates")
    if isinstance(raw_candidates, str) and raw_candidates.strip():
        return [item.strip() for item in raw_candidates.split(",") if item.strip()]
    candidate_set = str(options.get("candidate_set", "default"))
    if candidate_set == "sub216":
        return ["r5d1", "r4d1", "r3d1", "mlp8", "mlp6", "mlp4", "linear", "quadratic"]
    if candidate_set == "all":
        return [
            "r16d2",
            "r12d2",
            "r10d2",
            "r8d2",
            "r8d1",
            "r6d1",
            "r5d1",
            "r4d1",
            "r3d1",
            "mlp8",
            "mlp6",
            "mlp4",
            "linear",
            "quadratic",
        ]
    return ["r16d2", "r12d2", "r10d2", "r8d2", "r8d1", "r6d1"]


def _report_artifacts(options: Mapping[str, Any]) -> tuple[Path, ...]:
    config = options.get("config")
    if not isinstance(config, str):
        return ()
    config_path = resolve_repo_path(Path(config))
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    report = raw.get("report")
    if not isinstance(report, Mapping):
        return ()
    root = report.get("root", "artifacts/single_card_tradeoff/reports/current")
    output_path = report.get("output_path")
    paths = _root_artifacts(
        str(root),
        [
            "report_summary.json",
            "index.md",
        ],
    )
    if isinstance(output_path, str):
        return paths + (resolve_repo_path(Path(output_path)),)
    return paths


def _explicit_file_artifacts(
    options: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[Path, ...]:
    paths = []
    for key in keys:
        value = options.get(key)
        if isinstance(value, str):
            paths.append(resolve_repo_path(Path(value)))
    return tuple(_dedupe(paths))


def _out_dir_artifacts(
    options: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[Path, ...]:
    out_dir = _option_path(options, "out_dir")
    if out_dir is None:
        return ()
    return tuple(out_dir / name for name in names)


def _root_artifacts(root: str, names: Sequence[str]) -> tuple[Path, ...]:
    root_path = resolve_repo_path(Path(root))
    return tuple(root_path / name for name in names)


def _option_path(options: Mapping[str, Any], key: str) -> Path | None:
    value = options.get(key)
    if not isinstance(value, str):
        return None
    return resolve_repo_path(Path(value))


def _option_is_set(options: Mapping[str, Any], key: str) -> bool:
    value = options.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence):
        return bool(value)
    return True


def _parse_user_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [int(item) for item in value]
    return []


def _parse_csv_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _parse_csv_floats(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(item) for item in value.split(",") if item.strip()]
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    return []


def _dedupe(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _display_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
