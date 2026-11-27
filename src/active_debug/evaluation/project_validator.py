"""Command-line entry point for independent project validation."""

import argparse
import json
from pathlib import Path

from active_debug.evaluation.patch_validator import (
    ProjectVerdict,
    SuiteStatus,
    run_suite,
    validate_project,
)

__all__ = ["ProjectVerdict", "SuiteStatus", "run_suite", "validate_project"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--public-suite", type=Path, required=True)
    parser.add_argument("--held-out-suite", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        verdict = validate_project(args.candidate, args.public_suite, args.held_out_suite,
                                   image=args.image, timeout=args.timeout)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"passed": verdict.passed, "public": verdict.public,
                      "held_out": verdict.held_out}))
    if "error" in (verdict.public, verdict.held_out):
        return 2
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
