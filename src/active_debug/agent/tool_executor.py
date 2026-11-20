"""Trusted local fixtures only. Probe directories are discarded after each call."""
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from active_debug.schemas import (Case, Experiment, Observation, Patch, PatchResult,
                                  TestResult)


class ToolExecutor:
    def __init__(self, source: str, public: Case, timeout: float):
        self.source = source
        self.public = public
        self.timeout = timeout

    def run_experiment(self, experiment: Experiment) -> Observation:
        with TemporaryDirectory() as tmp:
            script = Path(tmp, "buggy.py")
            script.write_text(self.source, encoding="utf-8")
            driver = ("import runpy, sys; "
                      "print(runpy.run_path('buggy.py')['solve'](int(sys.argv[1])))")
            with open(os.devnull, "rb") as stdin:
                try:
                    run = subprocess.run(
                        [sys.executable, "-I", "-B", "-c", driver, str(experiment.argument)],
                        cwd=tmp, env={}, stdin=stdin, capture_output=True,
                        timeout=self.timeout, encoding="utf-8", errors="replace",
                    )
                except subprocess.TimeoutExpired as exc:
                    return Observation((exc.stdout or b"").decode(errors="replace"),
                                       (exc.stderr or b"").decode(errors="replace"), -1, True)
        return Observation(run.stdout, run.stderr, run.returncode)

    def apply_patch(self, patch: Patch) -> PatchResult:
        try:
            compile(patch.source, "buggy.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return PatchResult(False, str(exc))
        self.source = patch.source
        return PatchResult(True)

    def verify_patch(self) -> TestResult:
        obs = self.run_experiment(Experiment(self.public.argument))
        ok = obs.returncode == 0 and obs.stdout.strip() == str(self.public.expected)
        return TestResult(ok, 1, 1)
