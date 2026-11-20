"""Evaluate the frozen candidate in new processes without agent feedback."""
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import Case, TestResult


def validate(source: str, cases: list[Case], timeout: float) -> TestResult:
    if not cases:
        raise ValueError("validation suite is empty")
    outcomes = [ToolExecutor(source, case, timeout).verify_patch().passed for case in cases]
    return TestResult(all(outcomes), len(outcomes), len(cases))
