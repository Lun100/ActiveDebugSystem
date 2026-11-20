"""Produce unused probes with explicit demo-model predictions."""
from active_debug.schemas import DiagnosisState, Experiment

MODELS = {
    "adds_two": lambda x: x + 2,
    "double_minus_one": lambda x: 2 * x - 1,
}
PROBES = (3, 1, 2)


def generate(state: DiagnosisState) -> list[Experiment]:
    """Return probes predicted by every live hypothesis, or none if impossible.

    An empty list is the deterministic contract for causes the demo models do
    not cover; the controller then leaves diagnosis and moves to repair.
    """
    names = [h.name for h in state.hypotheses]
    if any(name not in MODELS for name in names):
        return []
    used = {e.argument for e, _ in state.history}
    return [Experiment(x, {name: str(MODELS[name](x)) for name in names})
            for x in PROBES if x not in used]
