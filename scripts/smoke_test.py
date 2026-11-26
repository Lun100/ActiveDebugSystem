"""Run all four strategies from the repository root and verify CLI JSON verdicts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

STRATEGIES = ("direct_repair", "free_debug", "hypothesis_llm", "hypothesis_eig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--provider", type=str, default="mock")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}

    for strategy in STRATEGIES:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "active_debug",
                "--strategy",
                strategy,
                "--provider",
                args.provider,
                "--output",
                str(args.output),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        verdict = json.loads(proc.stdout.strip())
        if not verdict["validation"]["passed"]:
            raise SystemExit(f"smoke check failed for {strategy}: {verdict}")
        print(proc.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
