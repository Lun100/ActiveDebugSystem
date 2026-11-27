"""Command-line entry point for preparing a BugsInPy task bundle."""

import argparse
import json
from pathlib import Path
import subprocess
from zipfile import BadZipFile

from active_debug.evaluation.dataset_loader import prepare_formal_tasks


DEFAULT_MANIFEST = Path("examples/bugsinpy_formal/manifest.json")


def prepare(manifest_path: Path, data: Path, private: Path, output: Path) -> bool:
    summary = prepare_formal_tasks(manifest_path, data, private, output)
    print(json.dumps(summary))
    return summary["qualified"] == summary["total"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=Path("../evaluation-data"))
    parser.add_argument("--private-root", type=Path,
                        default=Path("../evaluation-private/bugsinpy-formal-v1"))
    parser.add_argument("--output", type=Path, default=Path("../evaluation-data/bugsinpy-formal-v1"))
    args = parser.parse_args()
    try:
        return 0 if prepare(args.manifest.resolve(), args.data_root.resolve(),
                            args.private_root.resolve(), args.output.resolve()) else 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
