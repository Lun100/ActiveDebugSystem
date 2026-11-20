"""Injectable provider. The deterministic mock only supports the double fixture."""
from typing import Protocol
from active_debug.schemas import DebugTask, DiagnosisState, Experiment, FailureTrace, Patch


class Provider(Protocol):
    def choose(self, state: DiagnosisState, options: list[Experiment]) -> Experiment: ...
    def repair(self, task: DebugTask, failure: FailureTrace, state: DiagnosisState | None,
               feedback: str) -> Patch: ...
    def free_probe(self, task: DebugTask, failure: FailureTrace,
                   feedback: str) -> Experiment | None: ...


class MockProvider:
    def choose(self, state: DiagnosisState, options: list[Experiment]) -> Experiment:
        return options[-1]

    def free_probe(self, task: DebugTask, failure: FailureTrace,
                   feedback: str) -> Experiment | None:
        return None if feedback else Experiment(1)

    def repair(self, task: DebugTask, failure: FailureTrace, state: DiagnosisState | None,
               feedback: str) -> Patch:
        if '"""Return twice x."""' not in task.source:
            raise NotImplementedError("Mock repair only supports the double fixture")
        return Patch(task.source.replace("return x + 2", "return 2 * x"))
