from __future__ import annotations

import csv
import json
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_toml_profile(path: Path) -> dict[str, Any]:
    path = resolve_repo_path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected TOML object at {path}.")
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"schema_version must be 1 in {path}.")
    return raw


def config_path(
    config: Mapping[str, Any],
    *,
    section_name: str,
    field_name: str,
    default: Path,
) -> Path:
    section = config.get(section_name)
    if section is None:
        return default
    if not isinstance(section, Mapping):
        raise ValueError(f"{section_name} must be a TOML table.")
    value = section.get(field_name)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{field_name} must be a non-empty string.")
    return Path(value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return raw


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def format_float(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def format_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
