"""Condition on predicted output and open the hypothesis space on surprise.

An outcome that no live hypothesis predicts is not treated as "no evidence".
The first surprise introduces a reserved ``unexpected_output`` cause and each
further surprise hands it a larger share of the probability mass.  The agent
controllers use :func:`is_unexplained` to detect that situation and ask the
provider for a revised hypothesis space and new probes, so an anomalous probe
redirects diagnosis instead of leaving the posterior untouched.
"""
from active_debug.schemas import DiagnosisState, Experiment, Hypothesis, Observation

#: Reserved cause name for "none of the listed hypotheses explain the outcome".
UNEXPLAINED_CAUSE = "unexpected_output"

#: Fraction of the remaining mass handed to the open-world cause per surprise.
ANOMALY_SHARE = 0.2


def is_unexplained(probe: Experiment, obs: Observation) -> bool:
    """Return True when a conclusive run produced an unpredicted stdout token.

    Errors and timeouts stay inconclusive rather than anomalous: the probe
    itself failed, so its output says nothing about the hypotheses.  A probe
    with no predictions at all is unable to explain any output.
    """
    if obs.returncode or obs.timed_out:
        return False
    observed = obs.stdout.strip()
    return all(
        prediction != observed
        for name, prediction in probe.predictions.items()
        if name != UNEXPLAINED_CAUSE
    )


def update(state: DiagnosisState, probe: Experiment, obs: Observation) -> DiagnosisState:
    history = [*state.history, (probe, obs)]
    # Errors and timeouts are inconclusive: keep the prior without consulting
    # predictions, so incomplete predictions cannot mask an inconclusive run.
    if obs.returncode or obs.timed_out:
        return DiagnosisState(state.hypotheses, history)

    listed = [h for h in state.hypotheses if h.name != UNEXPLAINED_CAUSE]
    missing = [h.name for h in listed if h.name not in probe.predictions]
    if missing:
        raise ValueError("experiment lacks predictions for: " + ", ".join(missing))

    observed = obs.stdout.strip()
    matches = [h for h in listed if probe.predictions[h.name] == observed]
    mass = sum(h.belief for h in matches)

    if mass:
        # A listed cause explains the probe.  Condition normally and retire the
        # open-world mass: the listed causes have regained explanatory power.
        beliefs = [
            Hypothesis(h.name, h.belief / mass if h in matches else 0.0)
            for h in listed
        ]
        return DiagnosisState(beliefs, history)

    # No listed cause explains the observation.  Grow the open-world cause and
    # scale the listed beliefs down so their absolute probabilities change.
    prior = next(
        (h.belief for h in state.hypotheses if h.name == UNEXPLAINED_CAUSE), 0.0
    )
    known_total = 1.0 - prior
    surprise = prior + (1.0 - prior) * ANOMALY_SHARE
    if known_total <= 0:
        beliefs: list[Hypothesis] = []
    else:
        scale = (1.0 - surprise) / known_total
        beliefs = [Hypothesis(h.name, h.belief * scale) for h in listed]
    beliefs.append(Hypothesis(UNEXPLAINED_CAUSE, surprise))
    return DiagnosisState(beliefs, history)
