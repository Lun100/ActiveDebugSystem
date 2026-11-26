"""Manifest loader and Docker/local pytest sandbox executor for v3 benchmark bundles."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from active_debug.agent.tool_executor import _cap_output, _terminate_process_tree
from active_debug.schemas import Experiment, Observation, Patch, PatchResult, TestResult

DEFAULT_DOCKER_IMAGE = "active-debug-pysnooper-1"
_DEFAULT_MAX_OUTPUT_CHARS = 8192

# Zero-dependency fallback pytest-subset runner used locally when pytest is not installed.
# Supports top-level test_* functions and standard tmp_path: Path fixture injection.
_FALLBACK_SUITE_DRIVER = """\
import inspect
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import traceback

suite_dir = Path(sys.argv[1])
failures = 0
ran = 0

for test_file in sorted(suite_dir.glob("test_*.py")):
    ns = runpy.run_path(str(test_file))
    for name, fn in list(ns.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        ran += 1
        sig = inspect.signature(fn)
        with TemporaryDirectory() as tmp:
            kwargs = {}
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = Path(tmp)
            try:
                fn(**kwargs)
            except BaseException:
                failures += 1
                print(f"FAIL {test_file.name}::{name}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

if ran == 0:
    print("no tests collected", file=sys.stderr)
    sys.exit(5)
if failures:
    print(f"{failures} failed, {ran - failures} passed")
    sys.exit(1)
print(f"{ran} passed")
sys.exit(0)
"""


@dataclass(frozen=True)
class SuiteCase:
    """Single task entry loaded from a v3 bundle manifest.json."""

    case_id: str
    topic: str
    revision: str
    source_dir: Path
    public_suite_dir: Path
    initial_source: str
    public_test_source: str
    difficulty: str = "basic"
    functions: int = 1
    callable_symbols: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return self.case_id.replace("/", "-")


@dataclass(frozen=True)
class BundleManifest:
    """Parsed metadata and public cases from a v3 benchmark bundle."""

    name: str
    dataset_revision: str
    evaluation_python: str
    image: str
    root_dir: Path
    cases: tuple[SuiteCase, ...]


def extract_callable_symbols(source: str) -> tuple[str, ...]:
    """Extract top-level function names and Class.method names from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"{node.name}.{item.name}")
    return tuple(symbols)


def load_bundle(bundle_dir: Path) -> BundleManifest:
    """Load manifest.json and all public task sources/suites from bundle_dir."""
    root = bundle_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {root}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases: list[SuiteCase] = []
    for entry in raw.get("cases", []):
        source_dir = (root / entry["source"]).resolve()
        suite_dir = (root / entry["public_suite"]).resolve()
        buggy_path = source_dir / "buggy.py"
        if not buggy_path.is_file():
            raise FileNotFoundError(f"missing buggy.py for {entry['case_id']}: {buggy_path}")
        test_files = sorted(suite_dir.glob("test_*.py"))
        if not test_files:
            raise FileNotFoundError(f"no test_*.py files in {suite_dir}")

        initial_source = buggy_path.read_text(encoding="utf-8")
        public_tests = "\n\n".join(p.read_text(encoding="utf-8") for p in test_files)
        cases.append(
            SuiteCase(
                case_id=str(entry["case_id"]),
                topic=str(entry.get("topic", "")),
                revision=str(entry["revision"]),
                source_dir=source_dir,
                public_suite_dir=suite_dir,
                initial_source=initial_source,
                public_test_source=public_tests,
                difficulty=str(entry.get("difficulty", "basic")),
                functions=int(entry.get("functions", 1)),
                callable_symbols=extract_callable_symbols(initial_source),
            )
        )
    if not cases:
        raise ValueError(f"bundle at {root} contains no cases")

    return BundleManifest(
        name=str(raw.get("name", root.name)),
        dataset_revision=str(raw.get("dataset_revision", "")),
        evaluation_python=str(raw.get("evaluation_python", "")),
        image=str(raw.get("image", "")),
        root_dir=root,
        cases=tuple(cases),
    )


def _local_pytest_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", "import pytest"],
        capture_output=True,
    )
    return proc.returncode == 0


def docker_image_available(image: str = DEFAULT_DOCKER_IMAGE) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=3.0,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    max_output_chars: int,
) -> Observation:
    with open(os.devnull, "rb") as stdin:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(proc)
            raw_out, raw_err = proc.communicate()
            return Observation(
                _cap_output(raw_out.decode("utf-8", errors="replace"), max_output_chars),
                _cap_output(raw_err.decode("utf-8", errors="replace"), max_output_chars),
                -1,
                True,
            )

    return Observation(
        _cap_output(raw_out.decode("utf-8", errors="replace"), max_output_chars),
        _cap_output(raw_err.decode("utf-8", errors="replace"), max_output_chars),
        proc.returncode,
    )


class SuiteToolExecutor:
    """Execute public pytest suites and Python probe snippets against candidate buggy.py."""

    def __init__(
        self,
        source: str,
        public_suite_dir: Path,
        timeout: float,
        *,
        sandbox: str = "auto",
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if sandbox not in ("auto", "docker", "local"):
            raise ValueError(f"unsupported sandbox mode: {sandbox!r}")

        self.initial_source = source
        self.source = source
        self.public_suite_dir = public_suite_dir.resolve()
        self.timeout = timeout
        self.docker_image = docker_image
        self.max_output_chars = max_output_chars

        if sandbox == "docker":
            if not docker_image_available(docker_image):
                raise RuntimeError(
                    f"Docker daemon or image {docker_image!r} is not available"
                )
            self.backend = "docker"
        elif sandbox == "auto":
            self.backend = "docker" if docker_image_available(docker_image) else "local"
        else:
            self.backend = "local"

        self._has_local_pytest = _local_pytest_available() if self.backend == "local" else False
        self._probe_scripts: dict[int, str] = {}
        self._next_probe_id = 0

    @property
    def source_modified(self) -> bool:
        return self.source != self.initial_source

    def register_probe(
        self,
        code: str,
        predictions: dict[str, str] | None = None,
        cost: float = 1.0,
    ) -> Experiment:
        if not code or not code.strip():
            raise ValueError("probe code must be non-empty")
        probe_id = self._next_probe_id
        self._next_probe_id += 1
        self._probe_scripts[probe_id] = code
        return Experiment(probe_id, dict(predictions or {}), cost)

    def probe_code(self, experiment: Experiment) -> str:
        if experiment.argument not in self._probe_scripts:
            raise KeyError(f"unregistered probe id: {experiment.argument}")
        return self._probe_scripts[experiment.argument]

    def run_public_suite(self) -> Observation:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            work_dir = root / "work"
            candidate_dir.mkdir()
            work_dir.mkdir()
            (candidate_dir / "buggy.py").write_text(self.source, encoding="utf-8")

            if self.backend == "docker":
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--network",
                    "none",
                    "--read-only",
                    "--workdir",
                    "/tmp",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
                    "--mount",
                    f"type=bind,source={candidate_dir},target=/candidate,readonly",
                    "--mount",
                    f"type=bind,source={self.public_suite_dir},target=/suite,readonly",
                    "--env",
                    "PYTHONPATH=/candidate",
                    "--env",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    self.docker_image,
                    "python",
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-c",
                    "/dev/null",
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    "/suite",
                ]
                return _run_command(
                    cmd,
                    cwd=work_dir,
                    env=dict(os.environ),
                    timeout=self.timeout,
                    max_output_chars=self.max_output_chars,
                )

            env = {
                "PYTHONPATH": str(candidate_dir),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONUTF8": "0",
                "PYTHONIOENCODING": "utf-8",
            }
            if self._has_local_pytest:
                cmd = [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-c",
                    "/dev/null",
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    str(self.public_suite_dir),
                ]
            else:
                cmd = [
                    sys.executable,
                    "-B",
                    "-c",
                    _FALLBACK_SUITE_DRIVER,
                    str(self.public_suite_dir),
                ]
            return _run_command(
                cmd,
                cwd=work_dir,
                env=env,
                timeout=self.timeout,
                max_output_chars=self.max_output_chars,
            )

    def run_experiment(self, experiment: Experiment) -> Observation:
        script_code = self.probe_code(experiment)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            probe_dir = root / "probe"
            work_dir = root / "work"
            candidate_dir.mkdir()
            probe_dir.mkdir()
            work_dir.mkdir()
            (candidate_dir / "buggy.py").write_text(self.source, encoding="utf-8")
            probe_file = probe_dir / "probe_entry.py"
            probe_file.write_text(script_code, encoding="utf-8")

            if self.backend == "docker":
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--network",
                    "none",
                    "--read-only",
                    "--workdir",
                    "/tmp",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
                    "--mount",
                    f"type=bind,source={candidate_dir},target=/candidate,readonly",
                    "--mount",
                    f"type=bind,source={probe_dir},target=/probe,readonly",
                    "--env",
                    "PYTHONPATH=/candidate",
                    self.docker_image,
                    "python",
                    "-B",
                    "/probe/probe_entry.py",
                ]
                return _run_command(
                    cmd,
                    cwd=work_dir,
                    env=dict(os.environ),
                    timeout=self.timeout,
                    max_output_chars=self.max_output_chars,
                )

            env = {
                "PYTHONPATH": str(candidate_dir),
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONUTF8": "0",
                "PYTHONIOENCODING": "utf-8",
            }
            cmd = [sys.executable, "-B", str(probe_file)]
            return _run_command(
                cmd,
                cwd=work_dir,
                env=env,
                timeout=self.timeout,
                max_output_chars=self.max_output_chars,
            )

    def apply_patch(self, patch: Patch) -> PatchResult:
        try:
            tree = ast.parse(patch.source, filename="buggy.py")
            compile(tree, "buggy.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return PatchResult(False, str(exc))
        has_definition = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign))
            for node in tree.body
        )
        if not has_definition:
            return PatchResult(False, "patch contains no top-level definitions")
        self.source = patch.source
        return PatchResult(True)

    def verify_patch(self) -> TestResult:
        obs = self.run_public_suite()
        ok = (not obs.timed_out) and obs.returncode == 0
        return TestResult(ok, 1, 1)
