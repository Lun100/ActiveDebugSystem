"""Run from the repository root; installation is optional."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for strategy in ("direct_repair", "free_debug", "hypothesis_llm", "hypothesis_eig"):
    subprocess.run([sys.executable, "-m", "active_debug", "--strategy", strategy],
                   cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, check=True)
