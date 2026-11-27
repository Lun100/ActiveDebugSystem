"""Evaluate the frozen candidate in new processes without agent feedback."""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import subprocess
from typing import Literal
from uuid import uuid4

from active_debug.agent.tool_executor import ToolExecutor
from active_debug.evaluation.task_filter import project_validation_paths
from active_debug.schemas import Case, TestResult


SuiteStatus = Literal["passed", "failed", "error", "timeout"]


@dataclass(frozen=True)
class ProjectVerdict:
    public: SuiteStatus
    held_out: SuiteStatus

    @property
    def passed(self) -> bool:
        return self.public == self.held_out == "passed"


def validate(source: str, cases: list[Case], timeout: float) -> TestResult:
    if not cases:
        raise ValueError("validation suite is empty")
    outcomes = [ToolExecutor(source, case, timeout).verify_patch().passed for case in cases]
    return TestResult(all(outcomes), len(outcomes), len(cases))


def run_suite(candidate: Path, suite: Path, image: str, timeout: float) -> SuiteStatus:
    name = f"active-debug-validation-{uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--name", name, "--pull=never",
        "--network", "none", "--read-only", "--memory", "256m", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--workdir", "/tmp",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
        "--log-driver", "none",
        "--mount", f"type=bind,source={candidate},target=/candidate,readonly",
        "--mount", f"type=bind,source={suite},target=/suite,readonly",
        "--env", "PYTHONPATH=/candidate", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        image, "python", "-B", "/opt/active-debug/pytest_suite.py",
    ]
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
             "stderr": subprocess.DEVNULL}
    try:
        result = subprocess.run(command, timeout=timeout, **quiet)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "rm", "--force", name], timeout=5, **quiet)
        except subprocess.TimeoutExpired:
            pass
        return "timeout"

    # A normal test pass is distinct from a candidate ending the interpreter with exit 0.
    if result.returncode == 10:
        return "passed"
    if result.returncode == 1:
        return "failed"
    return "error"


def validate_project(candidate: Path, public_suite: Path, held_out_suite: Path, *,
                     image: str, timeout: float = 30) -> ProjectVerdict:
    """Run evaluator tests outside the candidate. Timeout applies per suite."""
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")

    source, public, held_out = project_validation_paths(candidate, public_suite, held_out_suite)
    return ProjectVerdict(
        public=run_suite(source, public, image, timeout),
        held_out=run_suite(source, held_out, image, timeout),
    )
