from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY_FILENAMES = (
    "all_summary.json",
    "preflight_summary.json",
    "baseline_summary.json",
    "gate_summary.json",
    "build_pareto_summary.json",
    "analyze_pareto_summary.json",
)


def collect_run_status(run_root: Path) -> dict[str, Any]:
    stage_statuses: list[dict[str, Any]] = []
    if run_root.exists():
        for stage_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
            summary_path = _find_summary(stage_dir)
            if summary_path is None:
                stage_statuses.append(
                    {
                        "stage": stage_dir.name,
                        "summary_path": None,
                        "type": "missing-summary",
                        "passed": False,
                        "exit_code": None,
                        "failures": ["incomplete-output"],
                    }
                )
                continue
            summary = _load_json(summary_path)
            stage_statuses.append(
                {
                    "stage": stage_dir.name,
                    "summary_path": str(summary_path),
                    "type": summary.get("type"),
                    "passed": _summary_passed(summary),
                    "exit_code": summary.get("exit_code"),
                    "failures": list(summary.get("failures", [])),
                }
            )
    passed = bool(stage_statuses) and all(
        bool(item["passed"]) for item in stage_statuses if item["stage"] != "all"
    )
    return {
        "type": "run-status",
        "run_root": str(run_root),
        "exists": run_root.exists(),
        "passed": passed,
        "stages": stage_statuses,
    }


def _find_summary(stage_dir: Path) -> Path | None:
    for filename in SUMMARY_FILENAMES:
        path = stage_dir / filename
        if path.exists():
            return path
    matches = sorted(stage_dir.glob("*_summary.json"))
    return matches[0] if matches else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _summary_passed(summary: dict[str, Any]) -> bool:
    if "passed" in summary:
        return bool(summary["passed"])
    exit_code = summary.get("exit_code")
    return exit_code == 0
