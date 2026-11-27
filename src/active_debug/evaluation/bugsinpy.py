"""Prepare and reproduce the public BugsInPy PySnooper/1 task."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

from active_debug.evaluation.dataset_loader import BugsInPyAdapter, export_revision
from active_debug.evaluation.patch_validator import run_suite


CASE_ID = "PySnooper/1"
IMAGE = "active-debug-pysnooper-1"
PUBLIC_SUITE = Path("examples/pysnooper_1/public")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", choices=[CASE_ID])
    parser.add_argument("--dataset-root", type=Path, default=Path("../evaluation-data/BugsInPy"))
    parser.add_argument("--source-repo", type=Path, default=Path("../evaluation-data/PySnooper"))
    parser.add_argument("--source-out", type=Path, help="save the reproduced source in a new directory")
    args = parser.parse_args()

    try:
        task = BugsInPyAdapter().load(args.dataset_root, args.case_id)
        if task.python_version != "3.8.1":
            raise ValueError(f"{CASE_ID} requires a matching image for Python {task.python_version}")
        suite = PUBLIC_SUITE.resolve()
        if not (suite / "test_chinese.py").is_file():
            raise ValueError("run from the ActiveDebugSystem repository root")
        output = args.source_out.absolute() if args.source_out is not None else None
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError(f"source output already exists: {output}")

        result = asdict(task)
        with TemporaryDirectory(prefix="active-debug-bugsinpy-",
                                dir=output.parent if output is not None else None) as temp:
            candidate = Path(temp).resolve() / "source"
            export_revision(args.source_repo, task.buggy_revision, candidate)
            public = run_suite(candidate, suite, IMAGE, timeout=30)
            if output is not None and public == "failed":
                if output.exists() or output.is_symlink():
                    raise FileExistsError(f"source output already exists: {output}")
                candidate.rename(output)
                result["source"] = str(output.resolve())
    except (OSError, ValueError, subprocess.SubprocessError, BadZipFile) as exc:
        parser.error(str(exc))

    print(json.dumps({"task": result, "public": public, "reproduced": public == "failed"}))
    if public in ("error", "timeout"):
        return 2
    return 0 if public == "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
