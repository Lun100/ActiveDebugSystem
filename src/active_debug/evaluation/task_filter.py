"""Check that tasks use distinct test inputs and valid reference outcomes."""
from pathlib import Path
from typing import TYPE_CHECKING

from active_debug.schemas import Case, DebugTask

if TYPE_CHECKING:
    from active_debug.evaluation.patch_validator import ProjectVerdict


def eligible(task: DebugTask, hidden: list[Case]) -> bool:
    return bool(hidden) and all(c.argument != task.public.argument for c in hidden)


def project_validation_paths(candidate: Path, public_suite: Path,
                             held_out_suite: Path) -> tuple[Path, Path, Path]:
    paths = (candidate.resolve(), public_suite.resolve(), held_out_suite.resolve())
    for index, path in enumerate(paths):
        if not path.is_dir():
            raise ValueError(f"expected a directory: {path}")
        for other in paths[:index]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("candidate, public suite and held-out suite must be separate")
    return paths


def qualifies_project_task(buggy: "ProjectVerdict", reference: "ProjectVerdict") -> bool:
    return buggy.public == buggy.held_out == "failed" and reference.passed
