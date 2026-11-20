"""Inject a strategy runner; evaluate only its final candidate."""
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4
from active_debug.agent.agent import Actions
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.agent.trace_logger import TraceLogger
from active_debug.evaluation.patch_validator import validate
from active_debug.evaluation.task_filter import eligible
from active_debug.schemas import Case, DebugTask, RunConfig, RunResult, Strategy

Runner = Callable[[DebugTask, Strategy, RunConfig, Actions, ToolExecutor, Provider], str]


def benchmark(task: DebugTask, hidden: list[Case], strategy: Strategy, config: RunConfig,
              provider: Provider, runner: Runner, output: Path) -> RunResult:
    if not eligible(task, hidden):
        raise ValueError("need disjoint public and held-out inputs")
    log = TraceLogger(output / f"{strategy}-{uuid4().hex}.jsonl")
    log.log("start", task=task.name, strategy=strategy, config=asdict(config),
            provider=type(provider).__name__)
    start = monotonic()
    actions = Actions(config.max_actions, log)
    tools = ToolExecutor(task.source, task.public, config.timeout)
    try:
        reason = runner(task, strategy, config, actions, tools, provider)
        verdict = validate(tools.source, [task.public, *hidden], config.timeout)
    except Exception as exc:
        log.log("error", kind=type(exc).__name__, detail=str(exc))
        raise
    result = RunResult(task.name, strategy, verdict, actions.used, monotonic()-start, reason)
    log.log("finish", **asdict(result))
    return result
