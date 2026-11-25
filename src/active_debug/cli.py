"""Repository-root CLI entry point."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tomllib

from active_debug.agent.baseline import make_runner, run
from active_debug.agent.checkpoint import CheckpointStore, InterruptPolicy
from active_debug.agent.llm_client import create_provider, load_dotenv
from active_debug.evaluation.benchmark_runner import benchmark
from active_debug.evaluation.dataset_loader import load
from active_debug.schemas import RunConfig, Strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="active-debug")
    parser.add_argument(
        "--strategy",
        type=Strategy,
        choices=list(Strategy),
        default=Strategy.EIG,
        help="Debugging strategy to execute.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to TOML RunConfig file.",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("examples/simple_bug"),
        help="Path to task fixture directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for JSONL trace logs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file loaded automatically if present.",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "openai", "openai-compatible", "http"),
        default=None,
        help="LLM provider backend (defaults to ACTIVE_DEBUG_PROVIDER in .env/env or 'mock').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for real LLM providers.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds per LLM API call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum transient error retries per LLM API call.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Optional override for RunConfig.max_actions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to save/load atomic run checkpoint JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from --checkpoint file.",
    )
    parser.add_argument(
        "--interrupt-policy",
        type=InterruptPolicy,
        choices=list(InterruptPolicy),
        default=InterruptPolicy.CHARGE_IN_FLIGHT,
        help="Budget accounting rule when resuming an interrupted call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resume and args.checkpoint is None:
        print("error: --resume requires --checkpoint <path>", file=sys.stderr)
        return 2

    load_dotenv(args.env_file)
    provider_kind = args.provider or os.environ.get("ACTIVE_DEBUG_PROVIDER") or "mock"

    config = RunConfig(**tomllib.loads(args.config.read_text(encoding="utf-8")))
    if args.max_actions is not None:
        config = replace(config, max_actions=args.max_actions)

    task, hidden = load(args.task)
    provider = create_provider(
        provider_kind,
        model=args.model,
        base_url=args.base_url,
        timeout=args.provider_timeout,
        max_retries=args.max_retries,
    )

    runner = run
    if args.checkpoint is not None:
        store = CheckpointStore(args.checkpoint, interrupt_policy=args.interrupt_policy)
        runner = make_runner(checkpoint_store=store, resume=args.resume)

    result = benchmark(task, hidden, args.strategy, config, provider, runner, args.output)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.validation.passed else 1
