"""Condition on predicted output. Unexplained outcomes preserve prior beliefs."""
from active_debug.schemas import DiagnosisState, Experiment, Hypothesis, Observation


def update(state: DiagnosisState, probe: Experiment, obs: Observation) -> DiagnosisState:
    history = [*state.history, (probe, obs)]
    # Errors and timeouts are inconclusive: keep the prior without consulting
    # predictions, so incomplete predictions cannot mask an inconclusive run.
    if obs.returncode or obs.timed_out:
        return DiagnosisState(state.hypotheses, history)
    missing = [h.name for h in state.hypotheses if h.name not in probe.predictions]
    if missing:
        raise ValueError("experiment lacks predictions for: " + ", ".join(missing))
    observed = obs.stdout.strip()
    matches = [h for h in state.hypotheses if probe.predictions[h.name] == observed]
    mass = sum(h.belief for h in matches)
    if not mass:
        return DiagnosisState(state.hypotheses, history)
    beliefs = [Hypothesis(h.name, h.belief / mass if h in matches else 0)
               for h in state.hypotheses]
    return DiagnosisState(beliefs, history)
