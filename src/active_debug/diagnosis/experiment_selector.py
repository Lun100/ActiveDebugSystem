"""Exact expected entropy reduction under a deterministic finite outcome model."""
from math import log2
from active_debug.diagnosis.belief_updater import UNEXPLAINED_CAUSE
from active_debug.schemas import DiagnosisState, Experiment

#: Sentinel outcome for the reserved open-world cause: its output is by
#: definition whatever the listed hypotheses did not predict.
_UNEXPLAINED_OUTCOME = "\x00unexpected"


def information_gain(experiment: Experiment, state: DiagnosisState) -> float:
    listed = [h for h in state.hypotheses if h.name != UNEXPLAINED_CAUSE]
    missing = [h.name for h in listed if h.name not in experiment.predictions]
    if missing:
        raise ValueError("experiment lacks predictions for: " + ", ".join(missing))
    masses: dict[str, float] = {}
    for h in listed:
        outcome = experiment.predictions[h.name]
        masses[outcome] = masses.get(outcome, 0.0) + h.belief
    open_mass = sum(
        h.belief for h in state.hypotheses if h.name == UNEXPLAINED_CAUSE
    )
    if open_mass:
        masses[_UNEXPLAINED_OUTCOME] = open_mass
    return -sum(p * log2(p) for p in masses.values() if p)


def expected_utility(experiment: Experiment, state: DiagnosisState, weight: float) -> float:
    """Information gain minus the cost penalty charged to the experiment."""
    return information_gain(experiment, state) - weight * experiment.cost


def worthwhile(
    options: list[Experiment], state: DiagnosisState, weight: float
) -> list[Experiment]:
    """Keep only probes whose expected utility is strictly positive.

    A non-positive score means the probe cannot repay its cost: either every
    live hypothesis predicts the same output (0 bit) or the cost penalty
    outweighs the remaining entropy.  Callers must stop diagnosing when this
    returns an empty list instead of running the highest-scoring candidate.
    """
    return [e for e in options if expected_utility(e, state, weight) > 0]


def select(options: list[Experiment], state: DiagnosisState, weight: float) -> Experiment:
    if not options:
        raise ValueError("no candidate experiments")
    return max(options, key=lambda e: expected_utility(e, state, weight))
