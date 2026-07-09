from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HybridSource:
    label: str
    train_outputs_dir: Path


def build_hybrid_root(
    *,
    sources: Sequence[HybridSource],
    output_run_root: Path,
    user_ids: Sequence[int],
    baseline_run_root: Path | None = None,
    selection_metric: str = "best_hypervolume_delta",
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source is required")
    train_outputs = output_run_root / "train-overfit" / "train_outputs"
    train_outputs.mkdir(parents=True, exist_ok=True)

    selection: list[dict[str, Any]] = []
    for user_id in user_ids:
        rows: dict[str, dict[str, Any]] = {}
        for source in sources:
            user_dir = source.train_outputs_dir / f"user_{user_id}"
            metrics_path = user_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            metrics = _read_json(metrics_path)
            metric_value = float(metrics[selection_metric])
            rows[source.label] = {
                "path": user_dir,
                "metric": metric_value,
                "best_objective_score": float(metrics.get("best_objective_score", 0.0)),
                "passed_overfit_gate": bool(metrics.get("passed_overfit_gate", False)),
            }
        selected_label = max(rows, key=lambda label: rows[label]["metric"])
        selected = rows[selected_label]
        destination = train_outputs / f"user_{user_id}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(selected["path"], destination)
        _copy_referenced_training_command(
            source_user_dir=selected["path"],
            destination_user_dir=destination,
        )
        selection.append(
            {
                "user_id": user_id,
                "selected_source": selected_label,
                "selected_train_output": str(selected["path"]),
                "selected_metric_name": selection_metric,
                "selected_metric_value": selected["metric"],
                "selected_best_objective_score": selected["best_objective_score"],
                "selected_passed_overfit_gate": selected["passed_overfit_gate"],
                "source_metrics": {
                    label: {
                        selection_metric: row["metric"],
                        "best_objective_score": row["best_objective_score"],
                        "passed_overfit_gate": row["passed_overfit_gate"],
                    }
                    for label, row in rows.items()
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "hybrid_name": output_run_root.name,
        "selection_rule": (
            "per user, select the policy artifact with the largest training "
            f"{selection_metric}; external sweep metrics are not used"
        ),
        "sources": [
            {
                "label": source.label,
                "train_outputs_dir": str(source.train_outputs_dir),
            }
            for source in sources
        ],
        "output_train_outputs": str(train_outputs),
        "selection": selection,
    }
    _write_json(output_run_root / "hybrid_selection_manifest.json", manifest)
    _write_training_summaries(output_run_root=output_run_root, selection=selection)
    if baseline_run_root is not None:
        _copy_stage_baseline(
            source_run_root=baseline_run_root,
            output_run_root=output_run_root,
        )
    return manifest


def _copy_referenced_training_command(
    *,
    source_user_dir: Path,
    destination_user_dir: Path,
) -> None:
    metadata_path = source_user_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        return
    raw_command_path = metadata.get("training_command_path")
    if not isinstance(raw_command_path, str) or not raw_command_path:
        return

    source_command_path = Path(raw_command_path)
    if not source_command_path.is_absolute():
        source_command_path = (source_user_dir / source_command_path).resolve()
    if not source_command_path.is_file():
        raise FileNotFoundError(source_command_path)

    destination_command_path = Path(raw_command_path)
    if destination_command_path.is_absolute():
        return
    destination_command_path = (
        destination_user_dir / destination_command_path
    ).resolve()
    destination_command_path.parent.mkdir(parents=True, exist_ok=True)
    if source_command_path != destination_command_path:
        shutil.copy2(source_command_path, destination_command_path)


def _write_training_summaries(
    *,
    output_run_root: Path,
    selection: Sequence[Mapping[str, Any]],
) -> None:
    train_root = output_run_root / "train-overfit"
    outputs = train_root / "train_outputs"
    artifact_paths = [
        str((outputs / f"user_{row['user_id']}" / "metadata.json").resolve())
        for row in selection
    ]
    progress_paths = [
        str((outputs / f"user_{row['user_id']}" / "training_progress.jsonl").resolve())
        for row in selection
    ]
    command_results = [
        {
            "artifact_paths_reported": [
                str((outputs / f"user_{row['user_id']}" / "metadata.json").resolve())
            ],
            "baseline_desired_retention": 0.9,
            "baseline_desired_retention_token": "0p9",
            "execution_mode": "hybrid_selection",
            "exit_code": 0,
            "output_dir": str((outputs / f"user_{row['user_id']}").resolve()),
            "overfit_gate_passed": bool(row["selected_passed_overfit_gate"]),
            "stderr_path": None,
            "stdout_path": None,
            "timed_out": False,
            "trainer": "fsrs6_cost_adr_hybrid_selection",
            "training_progress_path": str(
                (
                    outputs / f"user_{row['user_id']}" / "training_progress.jsonl"
                ).resolve()
            ),
            "user_id": row["user_id"],
            "selected_source": row["selected_source"],
            "selected_metric_name": row["selected_metric_name"],
            "selected_metric_value": row["selected_metric_value"],
        }
        for row in selection
    ]
    summary = {
        "type": "train-overfit",
        "stage": "train-overfit",
        "passed": all(bool(row["selected_passed_overfit_gate"]) for row in selection),
        "failures": [],
        "run_id": output_run_root.name,
        "stage_root": str(train_root.resolve()),
        "outputs_root": str(outputs.resolve()),
        "artifact_metadata_glob": "metadata.json",
        "artifact_paths": artifact_paths,
        "baseline_desired_retention_values": [0.9],
        "command_results": command_results,
        "training_progress_paths": progress_paths,
        "training_batch": {
            "enabled": True,
            "trainer": "fsrs6_cost_adr_hybrid_selection",
            "max_lanes_per_batch": 6400,
            "batch_size": None,
        },
        "notes": [
            "Hybrid train root assembled from prior per-user artifacts; "
            "see hybrid_selection_manifest.json."
        ],
    }
    _write_json(train_root / "training_summary.json", summary)
    _write_json(
        train_root / "gate_summary.json",
        {
            "gate_name": "train-overfit",
            "passed": summary["passed"],
            "failures": [] if summary["passed"] else ["hybrid-source-failed"],
            "metrics": {
                "commands_attempted": float(len(selection)),
                "commands_succeeded": float(
                    sum(1 for row in selection if row["selected_passed_overfit_gate"])
                ),
                "artifacts_validated": float(len(artifact_paths)),
            },
            "thresholds": {},
        },
    )
    _write_json(
        train_root / "performance_summary.json",
        {
            "stage": "train-overfit",
            "passed": summary["passed"],
            "failure_class": None if summary["passed"] else "hybrid-source-failed",
            "runtime_metrics": {
                "commands_attempted": len(selection),
                "commands_succeeded": sum(
                    1 for row in selection if row["selected_passed_overfit_gate"]
                ),
                "artifacts_validated": len(artifact_paths),
            },
            "execution_shape": {
                "resolved_batch_trainer": "fsrs6_cost_adr_hybrid_selection"
            },
            "gpu_metrics": {"gpu_monitor_shared_memory_spill_detected": None},
        },
    )


def _copy_stage_baseline(*, source_run_root: Path, output_run_root: Path) -> None:
    source = source_run_root / "stage-baseline"
    destination = output_run_root / "stage-baseline"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _source_arg(value: str) -> HybridSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must have the form label=path")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("source label must not be empty")
    path = _resolve_train_outputs_dir(Path(raw_path))
    return HybridSource(label=label, train_outputs_dir=path)


def _resolve_train_outputs_dir(path: Path) -> Path:
    if path.name == "train_outputs":
        return path
    train_outputs = path / "train-overfit" / "train_outputs"
    if train_outputs.is_dir():
        return train_outputs
    return path


def _parse_user_ids(values: Sequence[str]) -> list[int]:
    user_ids: list[int] = []
    for value in values:
        if "-" in value:
            start, end = value.split("-", 1)
            user_ids.extend(range(int(start), int(end) + 1))
        else:
            user_ids.append(int(value))
    return user_ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a per-user FSRS6 Cost-ADR hybrid train root.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_source_arg,
        help="Source artifact root as label=path. Path may be a run root or train_outputs dir.",
    )
    parser.add_argument("--output-run-root", required=True, type=Path)
    parser.add_argument("--baseline-run-root", type=Path)
    parser.add_argument("--users", nargs="+", required=True)
    parser.add_argument(
        "--selection-metric",
        default="best_hypervolume_delta",
    )
    args = parser.parse_args(argv)
    manifest = build_hybrid_root(
        sources=args.source,
        output_run_root=args.output_run_root,
        user_ids=_parse_user_ids(args.users),
        baseline_run_root=args.baseline_run_root,
        selection_metric=args.selection_metric,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
