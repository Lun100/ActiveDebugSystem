"""Repository-root CLI entry point supporting single fixtures and v3 benchmark bundles."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tomllib

from active_debug.agent.baseline import make_runner, run
from active_debug.agent.batch_runner import ALL_STRATEGIES, run_bundle_experiment
from active_debug.agent.checkpoint import CheckpointStore, InterruptPolicy
from active_debug.agent.llm_client import create_provider, load_dotenv
from active_debug.evaluation.benchmark_runner import benchmark
from active_debug.evaluation.dataset_loader import load
from active_debug.schemas import RunConfig, Strategy


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one integer seed")
    return tuple(int(p) for p in parts)


def _parse_strategy_csv(raw: str) -> tuple[Strategy, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one strategy")
    return tuple(Strategy(p) for p in parts)


def _parse_str_csv(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected non-empty comma-separated list")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="active-debug")
    parser.add_argument(
        "--strategy",
        type=Strategy,
        choices=list(Strategy),
        default=Strategy.EIG,
        help="Debugging strategy to execute in single-task mode.",
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
        help="Path to single-task fixture directory.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to v3 benchmark bundle directory containing manifest.json.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated case_ids for --bundle (default: all in manifest).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Optional comma-separated strategies for --bundle (default: all four).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated integer seeds for --bundle (default: '0').",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional shared experiment_id recorded in plan.json and runs.jsonl.",
    )
    parser.add_argument(
        "--model-snapshot",
        type=str,
        default=None,
        help="Optional model snapshot identifier recorded in plan.json.",
    )
    parser.add_argument(
        "--sandbox",
        choices=("auto", "docker", "local"),
        default="auto",
        help="Execution sandbox for --bundle public suites and probes.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="active-debug-pysnooper-1",
        help="Docker image name used when --sandbox resolves to 'docker'.",
    )
    parser.add_argument(
        "--wall-clock-limit",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds per run in --bundle mode.",
    )
    parser.add_argument(
        "--resume-batch",
        action="store_true",
        help="Skip already-completed (seed, case_id, strategy) entries in runs.jsonl.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent isolated runs in --bundle mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for JSONL trace logs or bundle deliverables.",
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
        help="Optional path to save/load atomic run checkpoint JSON (single-task mode).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution from --checkpoint file (single-task mode).",
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

    provider = create_provider(
        provider_kind,
        model=args.model,
        base_url=args.base_url,
        timeout=args.provider_timeout,
        max_retries=args.max_retries,
    )

    if args.bundle is not None:
        try:
            seeds = _parse_int_csv(args.seeds)
            strategies = (
                _parse_strategy_csv(args.strategies) if args.strategies else ALL_STRATEGIES
            )
            case_ids = _parse_str_csv(args.cases) if args.cases else None
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        summary = run_bundle_experiment(
            args.bundle,
            args.output,
            config,
            provider,
            seeds=seeds,
            strategies=strategies,
            case_ids=case_ids,
            experiment_id=args.experiment_id,
            model_snapshot=args.model_snapshot,
            sandbox=args.sandbox,
            docker_image=args.docker_image,
            wall_clock_limit=args.wall_clock_limit,
            resume_batch=args.resume_batch,
            jobs=args.jobs,
        )
        print(json.dumps(asdict(summary), ensure_ascii=False))
        return 0 if summary.error_runs == 0 else 1

    task, hidden = load(args.task)
    runner = run
    if args.checkpoint is not None:
        store = CheckpointStore(args.checkpoint, interrupt_policy=args.interrupt_policy)
        runner = make_runner(checkpoint_store=store, resume=args.resume)

    result = benchmark(task, hidden, args.strategy, config, provider, runner, args.output)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.validation.passed else 1
