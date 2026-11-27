"""Construct and validate a shared belief state."""
from active_debug.schemas import DiagnosisState, Experiment, Hypothesis, Observation


def initialize(
    candidates: list[Hypothesis],
    history: list[tuple[Experiment, Observation]] | None = None,
) -> DiagnosisState:
    return DiagnosisState(list(candidates), list(history or []))
