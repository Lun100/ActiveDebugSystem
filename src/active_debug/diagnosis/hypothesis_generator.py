"""Handwritten alternatives for the double demo, both predicting solve(3)=5."""
from active_debug.schemas import FailureTrace, Hypothesis


def generate(trace: FailureTrace) -> list[Hypothesis]:
    if (trace.case.argument, trace.case.expected, trace.actual.stdout.strip(),
        trace.actual.returncode) != (3, 6, "5", 0):
        return [Hypothesis("unknown", 1.0)]
    return [Hypothesis("adds_two", 0.5), Hypothesis("double_minus_one", 0.5)]
