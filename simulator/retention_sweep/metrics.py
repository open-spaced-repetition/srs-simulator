from __future__ import annotations

DEFAULT_PARETO_MEMORY_FIELD = "memorized_average"
DEFAULT_PARETO_TIME_FIELD = "time_average"
PARETO_MEMORY_FIELDS = (
    DEFAULT_PARETO_MEMORY_FIELD,
    "review_memory_gain_average",
)
PARETO_TIME_FIELDS = (
    DEFAULT_PARETO_TIME_FIELD,
    "review_time_average",
)


def memory_axis_label(memory_field: str) -> str:
    labels = {
        "memorized_average": "Memorized cards (average, all days)",
        "review_memory_gain_average": (
            "Memory gain from reviews vs no-review baseline (average, all days)"
        ),
    }
    return labels.get(memory_field, memory_field)


def time_axis_label(time_field: str) -> str:
    labels = {
        "time_average": "Minutes of studying per day (average)",
        "review_time_average": "Minutes of reviews per day (average)",
    }
    return labels.get(time_field, time_field)
