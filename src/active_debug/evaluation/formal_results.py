"""Command-line entry point for reporting the frozen experiment."""

import argparse
import json
from pathlib import Path

from active_debug.evaluation.benchmark_runner import benchmark_project_runs as evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("../evaluation-data/bugsinpy-formal-v1"))
    parser.add_argument("--private-root", type=Path,
                        default=Path("../evaluation-private/bugsinpy-formal-v1"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.bundle.resolve(), args.private_root.resolve(),
                          args.plan.resolve(), args.runs.resolve())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
