"""Construct and validate a shared belief state."""
from active_debug.schemas import DiagnosisState, Hypothesis


def initialize(candidates: list[Hypothesis]) -> DiagnosisState:
    return DiagnosisState(list(candidates))
