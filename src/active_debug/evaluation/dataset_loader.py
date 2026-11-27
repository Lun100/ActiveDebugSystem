"""Load local fixtures and prepare project tasks from BugsInPy."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from shutil import copytree
import subprocess
from tempfile import TemporaryDirectory
from typing import Protocol
from zipfile import ZipFile

from active_debug.evaluation.patch_validator import validate_project
from active_debug.evaluation.task_filter import qualifies_project_task
from active_debug.schemas import Case, DebugTask


class DatasetAdapter(Protocol):
    def load(self, root: Path) -> tuple[DebugTask, list[Case]]: ...


@dataclass(frozen=True)
class BugsInPyTask:
    """Public information needed to prepare a project task."""

    case_id: str
    repository_url: str
    buggy_revision: str
    python_version: str
    upstream_test_file: str


def _read_info_fields(path: Path, *keys: str) -> dict[str, str]:
    """Read selected quoted values without interpreting the dataset files as shell code."""
    fields = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, _, value = line.partition("=")
        key = key.strip()
        if key in keys:
            match = re.fullmatch(r'"([^"]+)"', value.strip())
            if match is None or key in fields:
                raise ValueError(f"invalid or repeated {key} in {path}")
            fields[key] = match[1]
    for key in keys:
        if key not in fields:
            raise ValueError(f"missing {key} in {path}")
    return fields


class BugsInPyAdapter:
    def load(self, root: Path, case_id: str) -> BugsInPyTask:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[0-9]+", case_id):
            raise ValueError("BugsInPy case must use Project/number")
        project, number = case_id.split("/")
        project_dir = root / "projects" / project
        project_fields = _read_info_fields(project_dir / "project.info", "github_url")
        bug_fields = _read_info_fields(project_dir / "bugs" / number / "bug.info",
                                       "buggy_commit_id", "python_version", "test_file")
        revision = bug_fields["buggy_commit_id"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"invalid buggy commit for {case_id}")
        return BugsInPyTask(case_id, project_fields["github_url"], revision,
                            bug_fields["python_version"], bug_fields["test_file"])


class PyBugHiveAdapter:
    def load(self, root: Path) -> tuple[DebugTask, list[Case]]:
        raise NotImplementedError("PyBugHive environment adapter pending")


def load(root: Path) -> tuple[DebugTask, list[Case]]:
    data = read_json(root / "task.json")
    hidden = [Case(*pair) for pair in data["held_out"]]
    if not hidden:
        raise ValueError("independent cases required")
    return DebugTask(data["name"], (root / "buggy.py").read_text(), Case(*data["public"])), hidden


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def export_revision(source_repo: Path, revision: str, destination: Path) -> None:
    """Extract a Git revision without checkout conversion or Git history."""
    archive = subprocess.run(
        ["git", "-C", str(source_repo), "-c", "core.autocrlf=false",
         "archive", "--format=zip", revision],
        capture_output=True, check=True, timeout=45,
    ).stdout
    with ZipFile(BytesIO(archive)) as snapshot:
        snapshot.extractall(destination)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _reference_files(source_repo: Path, buggy: str, fixed: str) -> list[str]:
    changed = subprocess.check_output(
        ["git", "-C", str(source_repo), "diff", "--name-only", buggy, fixed, "--"],
        text=True, encoding="utf-8", timeout=30,
    )
    files = []
    for name in changed.splitlines():
        parts = PurePosixPath(name).parts
        if (name.endswith(".py") and not {"test", "tests", "docs"}.intersection(parts)
                and not parts[-1].startswith("test_")):
            files.append(name)
    if not files:
        raise ValueError("reference repair has no production Python files")
    return sorted(set(files))


def _qualify_case(case: dict, *, manifest_dir: Path, data: Path, private: Path,
                 bundle: Path, references: Path, image: str) -> tuple[dict, dict]:
    task_id = case["id"]
    task = BugsInPyAdapter().load(data / "BugsInPy", task_id)
    project, number = task_id.split("/")
    slug = f"{project}-{number}"
    root = bundle / slug
    source_repo = data / project
    bug_info = data / "BugsInPy" / "projects" / project / "bugs" / number / "bug.info"
    fixed = _read_info_fields(bug_info, "fixed_commit_id")["fixed_commit_id"]
    if not re.fullmatch(r"[0-9a-f]{40}", fixed):
        raise ValueError(f"invalid fixed commit for {task_id}")

    copytree(manifest_dir / "public" / slug, root / "public")
    export_revision(source_repo, task.buggy_revision, root / "source")
    reference = references / slug
    export_revision(source_repo, fixed, reference)
    suites = (root / "public", private / "held-out" / slug)
    old = validate_project(root / "source", *suites, image=image)
    repaired = validate_project(reference, *suites, image=image)

    public = {
        "case_id": task_id,
        "topic": case["topic"],
        "revision": task.buggy_revision,
        "source": f"{slug}/source",
        "public_suite": f"{slug}/public",
    }
    evidence = {
        "case_id": task_id,
        "revision": task.buggy_revision,
        "reference_revision": fixed,
        "location_files": _reference_files(source_repo, task.buggy_revision, fixed),
        "buggy": asdict(old),
        "reference": asdict(repaired),
        "qualified": qualifies_project_task(old, repaired),
    }
    return public, evidence


def prepare_formal_tasks(manifest_path: Path, data: Path, private: Path, output: Path) -> dict:
    """Export a public bundle after checking the buggy and reference revisions."""
    manifest = read_json(manifest_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"bundle already exists: {output}")
    if private.is_relative_to(output) or output.is_relative_to(private):
        raise ValueError("public bundle and private evaluation paths must be separate")
    actual = subprocess.check_output(
        ["git", "-C", str(data / "BugsInPy"), "rev-parse", "HEAD"],
        text=True, timeout=30,
    ).strip()
    if actual != manifest["dataset_revision"]:
        raise ValueError("BugsInPy checkout does not match the frozen dataset revision")
    image = subprocess.check_output(
        ["docker", "image", "inspect", manifest["image"], "--format", "{{.Id}}"],
        text=True, timeout=30,
    ).strip()
    ids = [case["id"] for case in manifest["cases"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("task IDs must be nonempty and unique")

    with TemporaryDirectory(prefix="active-debug-qualify-", dir=output.parent) as temporary:
        temp = Path(temporary)
        bundle = temp / "bundle"
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(_qualify_case, case, manifest_dir=manifest_path.parent,
                                data=data, private=private, bundle=bundle,
                                references=temp / "reference", image=image)
                    for case in manifest["cases"]]
            qualified = [job.result() for job in jobs]
        public, evidence = zip(*qualified)
        all_ready = all(case["qualified"] for case in evidence)
        _write_json(private / "preflight.json", {
            "dataset_revision": actual, "image": image, "cases": evidence,
        })

        if all_ready:
            _write_json(bundle / "manifest.json", {
                "name": manifest["name"],
                "dataset_revision": actual,
                "evaluation_python": manifest["evaluation_python"],
                "image": image,
                "cases": public,
            })
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"bundle already exists: {output}")
            bundle.rename(output)
    return {
        "qualified": sum(case["qualified"] for case in evidence),
        "total": len(evidence),
        "bundle": str(output) if all_ready else None,
        "preflight": str(private / "preflight.json"),
    }
