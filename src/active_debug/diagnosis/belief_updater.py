"""Condition on predicted output. Unexplained outcomes preserve prior beliefs."""
from active_debug.schemas import DiagnosisState, Experiment, Hypothesis, Observation


def update(state: DiagnosisState, probe: Experiment, obs: Observation) -> DiagnosisState:
    matches = [h for h in state.hypotheses if probe.predictions[h.name] == obs.stdout.strip()]
    mass = sum(h.belief for h in matches)
    beliefs = state.hypotheses
    if not obs.returncode and not obs.timed_out and mass:
        beliefs = [Hypothesis(h.name, h.belief / mass if h in matches else 0)
                   for h in beliefs]
    return DiagnosisState(beliefs, [*state.history, (probe, obs)])
