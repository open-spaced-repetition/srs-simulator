from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DEFAULT_RESULT_BASE = {
    "dash": "DASH",
    "fsrs3": "FSRSv3",
    "fsrs6": "FSRS-6-recency",
    "hlr": "HLR",
}

SHORT_TERM_RESULT_BASE = {
    "dash": "DASH-short",
    # The benchmark repo uses "-short-secs" naming for FSRSv3 short-term weights.
    "fsrs3": "FSRSv3-short-secs",
    "fsrs6": "FSRS-6-short-recency",
    "hlr": "HLR-short",
}


def parse_result_overrides(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid benchmark result override '{item}'. Expected key=value."
            )
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid benchmark result override '{item}'. Expected key=value."
            )
        mapping[key] = value
    return mapping


def resolve_benchmark_root(repo_root: Path, benchmark_root: Path | None) -> Path:
    return benchmark_root or (repo_root.parent / "srs-benchmark")


def resolve_result_path(benchmark_root: Path, base_name: str) -> Path:
    return benchmark_root / "result" / f"{base_name}.jsonl"


def load_result_parameters(
    path: Path,
    *,
    user_id: int,
    partition_key: str = "0",
) -> list[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("user") != user_id:
                continue
            params = record.get("parameters")
            if params is None:
                raise ValueError(
                    f"Entry for user {user_id} in {path} missing parameters."
                )
            if isinstance(params, dict):
                if partition_key not in params:
                    available = ", ".join(map(str, params.keys()))
                    raise ValueError(
                        f"Partition '{partition_key}' not found in {path}. "
                        f"Available keys: {available}"
                    )
                return [float(x) for x in params[partition_key]]
            if isinstance(params, list):
                return [float(x) for x in params]
            raise TypeError(f"Unsupported parameter format in {path}: {type(params)}")
    raise ValueError(f"User {user_id} not found in {path}")


def load_benchmark_weights(
    *,
    repo_root: Path,
    benchmark_root: Path | None,
    environment: str,
    user_id: int,
    partition_key: str = "0",
    overrides: dict[str, str] | None = None,
    short_term: bool = False,
) -> list[float]:
    env_key = environment.lower()
    overrides = overrides or {}
    base_name = overrides.get(env_key)
    if not base_name and short_term:
        base_name = SHORT_TERM_RESULT_BASE.get(env_key)
    if not base_name:
        base_name = DEFAULT_RESULT_BASE.get(env_key)
    if not base_name:
        raise ValueError(
            f"No default benchmark result base name for environment '{environment}'. "
            "Provide --benchmark-result to specify one."
        )
    root = resolve_benchmark_root(repo_root, benchmark_root)
    result_path = resolve_result_path(root, base_name)
    if not result_path.exists():
        raise FileNotFoundError(f"Benchmark result file not found: {result_path}")
    return load_result_parameters(
        result_path,
        user_id=user_id,
        partition_key=partition_key,
    )


__all__ = [
    "DEFAULT_RESULT_BASE",
    "SHORT_TERM_RESULT_BASE",
    "load_benchmark_weights",
    "parse_result_overrides",
    "resolve_benchmark_root",
    "resolve_result_path",
]
