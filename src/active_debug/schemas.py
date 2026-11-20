"""Shared value objects. Only the evaluator owns held-out cases."""
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite, isclose


class Strategy(StrEnum):
    DIRECT = "direct_repair"
    FREE = "free_debug"
    LLM = "hypothesis_llm"
    EIG = "hypothesis_eig"


@dataclass(frozen=True)
class Case:
    """A public or evaluator-only integer input/output pair."""
    argument: int
    expected: int

    def __post_init__(self):
        if type(self.argument) is not int or type(self.expected) is not int:
            raise ValueError("integer cases required")


@dataclass(frozen=True)
class DebugTask:
    """Agent-visible source and one public reproducer."""
    name: str
    source: str
    public: Case

    def __post_init__(self):
        if not self.name or not self.source.strip():
            raise ValueError("empty task")


@dataclass(frozen=True)
class Observation:
    """Captured process outcome. Timeout and errors are inconclusive."""
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


@dataclass(frozen=True)
class FailureTrace:
    """Public expected output paired with the actual run."""
    case: Case
    actual: Observation


@dataclass(frozen=True)
class Hypothesis:
    """Candidate cause with its current probability mass."""
    name: str
    belief: float

    def __post_init__(self):
        if not self.name or not 0 <= self.belief <= 1:
            raise ValueError("invalid hypothesis")


@dataclass(frozen=True)
class Experiment:
    """Probe solve(argument), with optional deterministic output by cause."""
    argument: int
    predictions: dict[str, str] = field(default_factory=dict)
    cost: float = 1.0

    def __post_init__(self):
        if type(self.argument) is not int or not isfinite(self.cost) or self.cost <= 0:
            raise ValueError("invalid probe")


@dataclass(frozen=True)
class Patch:
    """Full replacement for buggy.py in a disposable workspace."""
    source: str

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError("empty replacement")


@dataclass(frozen=True)
class PatchResult:
    """Whether the replacement compiled and was installed."""
    applied: bool
    detail: str = ""


@dataclass(frozen=True)
class TestResult:
    """Outcome and counts from public or independent validation."""
    passed: bool
    checked: int
    total: int


@dataclass
class DiagnosisState:
    """Normalized beliefs and completed probes in order."""
    hypotheses: list[Hypothesis]
    history: list[tuple[Experiment, Observation]] = field(default_factory=list)

    def __post_init__(self):
        names = {h.name for h in self.hypotheses}
        if len(names) != len(self.hypotheses) or not isclose(
            sum(h.belief for h in self.hypotheses), 1.0
        ):
            raise ValueError("unique normalized hypotheses required")


@dataclass(frozen=True)
class RunConfig:
    """Shared action cap, per-process seconds, threshold and cost penalty."""
    max_actions: int = 12
    timeout: float = 2.0
    confidence: float = 0.95
    cost_weight: float = 0.1

    def __post_init__(self):
        if type(self.max_actions) is not int or self.max_actions < 1:
            raise ValueError("positive action budget required")
        if not isfinite(self.timeout) or self.timeout <= 0 or not 0 < self.confidence <= 1:
            raise ValueError("invalid timeout or confidence")
        if not isfinite(self.cost_weight) or self.cost_weight < 0:
            raise ValueError("invalid cost weight")


@dataclass(frozen=True)
class RunResult:
    """Independent verdict and observed usage for a single run."""
    task: str
    strategy: Strategy
    validation: TestResult
    actions: int
    seconds: float
    stop_reason: str
