"""Trusted local fixtures execution with process-group cleanup and output caps."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

from active_debug.schemas import (
    Case,
    Experiment,
    Observation,
    Patch,
    PatchResult,
    TestResult,
)

_DEFAULT_MAX_OUTPUT_CHARS = 8192


def _cap_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _defines_solve(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "solve":
                    return True
    return False


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


class ToolExecutor:
    def __init__(
        self,
        source: str,
        public: Case,
        timeout: float,
        *,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_output_chars < 64:
            raise ValueError("max_output_chars must be at least 64")
        self.source = source
        self.public = public
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def run_experiment(self, experiment: Experiment) -> Observation:
        with TemporaryDirectory() as tmp:
            script = Path(tmp, "buggy.py")
            script.write_text(self.source, encoding="utf-8")
            driver = (
                "import runpy, sys; "
                "print(runpy.run_path('buggy.py')['solve'](int(sys.argv[1])))"
            )
            with open(os.devnull, "rb") as stdin:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-B", "-c", driver, str(experiment.argument)],
                    cwd=tmp,
                    env={},
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=(os.name == "posix"),
                )
                try:
                    raw_out, raw_err = proc.communicate(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(proc)
                    raw_out, raw_err = proc.communicate()
                    stdout = _cap_output(raw_out.decode("utf-8", errors="replace"), self.max_output_chars)
                    stderr = _cap_output(raw_err.decode("utf-8", errors="replace"), self.max_output_chars)
                    return Observation(stdout, stderr, -1, True)

        stdout = _cap_output(raw_out.decode("utf-8", errors="replace"), self.max_output_chars)
        stderr = _cap_output(raw_err.decode("utf-8", errors="replace"), self.max_output_chars)
        return Observation(stdout, stderr, proc.returncode)

    def apply_patch(self, patch: Patch) -> PatchResult:
        try:
            tree = ast.parse(patch.source, filename="buggy.py")
            compile(tree, "buggy.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return PatchResult(False, str(exc))
        if not _defines_solve(tree):
            return PatchResult(False, "patch must define top-level 'solve'")
        self.source = patch.source
        return PatchResult(True)

    def verify_patch(self) -> TestResult:
        obs = self.run_experiment(Experiment(self.public.argument))
        ok = (not obs.timed_out) and obs.returncode == 0 and obs.stdout.strip() == str(self.public.expected)
        return TestResult(ok, 1, 1)
