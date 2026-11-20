"""Local integer-function fixture format and extension contract."""
import json
from pathlib import Path
from typing import Protocol
from active_debug.schemas import Case, DebugTask


class DatasetAdapter(Protocol):
    def load(self, root: Path) -> tuple[DebugTask, list[Case]]: ...


class BugsInPyAdapter:
    def load(self, root: Path) -> tuple[DebugTask, list[Case]]:
        raise NotImplementedError("BugsInPy environment adapter pending")


class PyBugHiveAdapter:
    def load(self, root: Path) -> tuple[DebugTask, list[Case]]:
        raise NotImplementedError("PyBugHive environment adapter pending")


def load(root: Path) -> tuple[DebugTask, list[Case]]:
    data = json.loads((root / "task.json").read_text())
    hidden = [Case(*pair) for pair in data["held_out"]]
    if not hidden:
        raise ValueError("independent cases required")
    return DebugTask(data["name"], (root / "buggy.py").read_text(), Case(*data["public"])), hidden
