from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch


DEFAULT_DP_CACHE_DIR = Path("artifacts") / "single_card_tradeoff" / "dp_cache"
DP_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OracleDPCacheConfig:
    enabled: bool = True
    cache_dir: Path = DEFAULT_DP_CACHE_DIR
    refresh: bool = False


@dataclass
class OracleDPCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    refreshes: int = 0


_GLOBAL_STATS = OracleDPCacheStats()
_DEFAULT_CONFIG = OracleDPCacheConfig()


def add_oracle_dp_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dp-cache-dir",
        type=Path,
        default=DEFAULT_DP_CACHE_DIR,
        help=(
            "Directory for single-card oracle DP cache artifacts "
            f"(default: {DEFAULT_DP_CACHE_DIR})."
        ),
    )
    parser.add_argument(
        "--no-dp-cache",
        dest="dp_cache_enabled",
        action="store_false",
        default=True,
        help="Disable single-card oracle DP cache reads and writes.",
    )
    parser.add_argument(
        "--refresh-dp-cache",
        action="store_true",
        help="Recompute oracle DP entries and overwrite matching cache files.",
    )


def oracle_dp_cache_config_from_args(args: argparse.Namespace) -> OracleDPCacheConfig:
    return OracleDPCacheConfig(
        enabled=bool(getattr(args, "dp_cache_enabled", True)),
        cache_dir=Path(getattr(args, "dp_cache_dir", DEFAULT_DP_CACHE_DIR)),
        refresh=bool(getattr(args, "refresh_dp_cache", False)),
    )


def set_default_oracle_dp_cache_config(config: OracleDPCacheConfig) -> None:
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config


def resolve_oracle_dp_cache_config(
    config: OracleDPCacheConfig | None = None,
) -> OracleDPCacheConfig:
    return config or _DEFAULT_CONFIG


def reset_oracle_dp_cache_stats() -> None:
    _GLOBAL_STATS.hits = 0
    _GLOBAL_STATS.misses = 0
    _GLOBAL_STATS.writes = 0
    _GLOBAL_STATS.refreshes = 0


def oracle_dp_cache_stats_snapshot() -> dict[str, int]:
    return {
        "hits": _GLOBAL_STATS.hits,
        "misses": _GLOBAL_STATS.misses,
        "writes": _GLOBAL_STATS.writes,
        "refreshes": _GLOBAL_STATS.refreshes,
    }


def oracle_dp_cache_performance_payload() -> dict[str, Any]:
    return {
        "dp_cache_hits": _GLOBAL_STATS.hits,
        "dp_cache_misses": _GLOBAL_STATS.misses,
        "dp_cache_writes": _GLOBAL_STATS.writes,
        "dp_cache_refreshes": _GLOBAL_STATS.refreshes,
    }


def cache_key(key_parts: dict[str, Any]) -> str:
    canonical = _canonicalize(
        {
            "schema_version": DP_CACHE_SCHEMA_VERSION,
            **key_parts,
        }
    )
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache_entry(
    config: OracleDPCacheConfig,
    *,
    key_parts: dict[str, Any],
    map_location: torch.device | str | None = None,
) -> dict[str, Any] | None:
    if not config.enabled:
        return None
    if config.refresh:
        _GLOBAL_STATS.refreshes += 1
        _GLOBAL_STATS.misses += 1
        return None

    key = cache_key(key_parts)
    path = _cache_path(config.cache_dir, key)
    if not path.exists():
        _GLOBAL_STATS.misses += 1
        return None
    try:
        payload = torch.load(path, map_location=map_location)
    except Exception:
        _GLOBAL_STATS.misses += 1
        return None
    if not isinstance(payload, dict) or payload.get("cache_key") != key:
        _GLOBAL_STATS.misses += 1
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        _GLOBAL_STATS.misses += 1
        return None
    _GLOBAL_STATS.hits += 1
    return data


def write_cache_entry(
    config: OracleDPCacheConfig,
    *,
    key_parts: dict[str, Any],
    data: dict[str, Any],
) -> None:
    if not config.enabled:
        return
    key = cache_key(key_parts)
    path = _cache_path(config.cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": DP_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "key_parts": _canonicalize(key_parts),
        "created_at": time.time(),
        "data": _to_cpu(data),
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        _GLOBAL_STATS.writes += 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.pt"


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _canonicalize(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".17g")
    return str(value)
