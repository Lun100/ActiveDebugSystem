from active_debug.schemas import DiagnosisState


def should_stop(state: DiagnosisState, threshold: float) -> bool:
    return max(h.belief for h in state.hypotheses) >= threshold
