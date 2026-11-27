"""Summarize repair outcomes and measured costs."""
from collections import Counter
from dataclasses import dataclass
from statistics import mean

from active_debug.schemas import RunResult, Strategy


COST_FIELDS = ("actions", "seconds", "tokens")


@dataclass(frozen=True, order=True)
class RunKey:
    case_id: str
    revision: str
    seed: int


@dataclass(frozen=True)
class ScoredRun:
    key: RunKey
    strategy: Strategy
    status: str
    passed: bool
    stop_reason: str
    actions: int | None = None
    seconds: float | None = None
    tokens: int | None = None
    location_correct: bool | None = None


def summarize(results: list[RunResult]) -> dict[str, float]:
    if not results:
        raise ValueError("no completed runs")
    n = len(results)
    return {"success_rate": sum(r.validation.passed for r in results) / n,
            "mean_actions": sum(r.actions for r in results) / n,
            "mean_seconds": sum(r.seconds for r in results) / n}


def summarize_strategy(records: list[ScoredRun], planned: int) -> dict:
    """Use scheduled runs as the denominator and report observed costs separately."""
    successes = sum(run.passed for run in records)
    locations = [run.location_correct for run in records if run.location_correct is not None]
    costs = {}
    for field in COST_FIELDS:
        samples = [value for run in records if (value := getattr(run, field)) is not None]
        costs[field] = {"observed": len(samples), "mean": mean(samples) if samples else None}
    return {
        "planned": planned, "observed": len(records), "missing": planned - len(records),
        "repaired": successes, "repair_rate": successes / planned,
        "status_counts": dict(Counter(run.status for run in records)),
        "stop_reason_counts": dict(Counter(run.stop_reason for run in records)),
        "location_predictions": len(locations),
        "locations_correct": sum(locations),
        "costs": costs,
    }
