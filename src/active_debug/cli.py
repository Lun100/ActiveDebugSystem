"""Repository-root CLI entry point."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tomllib

from active_debug.agent.baseline import run
from active_debug.agent.llm_client import MockProvider
from active_debug.evaluation.benchmark_runner import benchmark
from active_debug.evaluation.dataset_loader import load
from active_debug.schemas import RunConfig, Strategy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Strategy, choices=list(Strategy), default=Strategy.EIG)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--task", type=Path, default=Path("examples/simple_bug"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = RunConfig(**tomllib.loads(args.config.read_text()))
    task, hidden = load(args.task)
    result = benchmark(task, hidden, args.strategy, config, MockProvider(), run, args.output)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.validation.passed else 1
