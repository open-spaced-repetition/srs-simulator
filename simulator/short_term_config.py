from __future__ import annotations

import argparse


def parse_steps(value: str | None) -> list[float]:
    if not value:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def resolve_short_term_config(
    args: argparse.Namespace,
) -> tuple[str | None, list[float], list[float]]:
    short_term_source = getattr(args, "short_term_source", None)
    learning_raw = getattr(args, "learning_steps", None)
    relearning_raw = getattr(args, "relearning_steps", None)
    if short_term_source == "steps":
        if learning_raw is None:
            learning_raw = "1,10"
        if relearning_raw is None:
            relearning_raw = "10"
    learning_steps = parse_steps(learning_raw)
    relearning_steps = parse_steps(relearning_raw)
    if short_term_source is None:
        if learning_steps or relearning_steps:
            raise SystemExit("Learning steps require --short-term-source=steps.")
        return None, learning_steps, relearning_steps
    if short_term_source == "sched" and (learning_steps or relearning_steps):
        raise SystemExit(
            "--short-term-source=sched cannot be combined with "
            "--learning-steps or --relearning-steps."
        )
    return short_term_source, learning_steps, relearning_steps
