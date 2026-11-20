"""Produce unused probes with explicit demo-model predictions."""
from active_debug.schemas import DiagnosisState, Experiment


def generate(state: DiagnosisState) -> list[Experiment]:
    used = {e.argument for e, _ in state.history}
    return [Experiment(x, {"adds_two": str(x + 2), "double_minus_one": str(2*x - 1)})
            for x in (3, 1, 2) if x not in used]
