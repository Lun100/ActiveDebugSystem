"""Exact expected entropy reduction under a deterministic finite outcome model."""
from math import log2
from active_debug.schemas import DiagnosisState, Experiment


def information_gain(experiment: Experiment, state: DiagnosisState) -> float:
    missing = [h.name for h in state.hypotheses if h.name not in experiment.predictions]
    if missing:
        raise ValueError("experiment lacks predictions for: " + ", ".join(missing))
    masses: dict[str, float] = {}
    for h in state.hypotheses:
        outcome = experiment.predictions[h.name]
        masses[outcome] = masses.get(outcome, 0.0) + h.belief
    return -sum(p * log2(p) for p in masses.values() if p)


def select(options: list[Experiment], state: DiagnosisState, weight: float) -> Experiment:
    if not options:
        raise ValueError("no candidate experiments")
    return max(options, key=lambda e: information_gain(e, state) - weight * e.cost)
