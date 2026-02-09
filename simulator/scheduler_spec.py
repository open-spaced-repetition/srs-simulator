from __future__ import annotations


def parse_scheduler_spec(value: str) -> tuple[str, float | None, str]:
    raw = value.strip()
    if "@" not in raw:
        aliases = {
            "fsrs6-default": "fsrs6_default",
        }
        return aliases.get(raw, raw), None, raw
    name, param = raw.split("@", 1)
    name = name.strip()
    param = param.strip()
    if not name or not param:
        raise ValueError(
            f"Invalid scheduler spec '{value}'. Expected name or name@value."
        )
    if name != "fixed":
        raise ValueError("Only the 'fixed' scheduler accepts @<interval>.")
    try:
        interval = float(param)
    except ValueError as exc:
        raise ValueError(
            f"Invalid fixed interval '{param}'. Expected a number."
        ) from exc
    if interval <= 0:
        raise ValueError("Fixed interval must be > 0.")
    return name, interval, raw


def normalize_fixed_interval(value: float | None, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def format_float(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def scheduler_uses_desired_retention(scheduler: str) -> bool:
    return scheduler not in {"fixed", "anki_sm2", "memrise"}
