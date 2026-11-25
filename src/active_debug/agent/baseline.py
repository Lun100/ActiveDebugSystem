"""Common reproduce -> (diagnose / free-probe) -> repair -> verify loop for all four strategies."""
from __future__ import annotations

from active_debug.agent.agent import Actions, BudgetExhausted, WallClockExhausted
from active_debug.agent.debug_controller import REPAIR_ACTION_RESERVE, diagnose
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import (
    DebugTask,
    Experiment,
    FailureTrace,
    Observation,
    RunConfig,
    Strategy,
)


def _is_passing_observation(obs: Observation, expected: int) -> bool:
    return (not obs.timed_out) and obs.returncode == 0 and obs.stdout.strip() == str(expected)


def _append_feedback(existing: str, line: str) -> str:
    return f"{existing}\n{line}" if existing else line


def _format_probe_feedback(probe: Experiment, obs: Observation) -> str:
    return (
        f"[probe] solve({probe.argument}) -> stdout={obs.stdout.strip()!r}, "
        f"stderr={obs.stderr.strip()!r}, returncode={obs.returncode}, timed_out={obs.timed_out}"
    )


def _format_failure_feedback(failure: FailureTrace) -> str:
    obs = failure.actual
    return (
        f"[verify_failed] solve({failure.case.argument}) expected {failure.case.expected}, "
        f"got stdout={obs.stdout.strip()!r}, stderr={obs.stderr.strip()!r}, "
        f"returncode={obs.returncode}, timed_out={obs.timed_out}"
    )


def run(
    task: DebugTask,
    strategy: Strategy,
    config: RunConfig,
    actions: Actions,
    tools: ToolExecutor,
    provider: Provider,
) -> str:
    try:
        actual = actions.call("reproduce", tools.run_experiment, Experiment(task.public.argument))
        failure = FailureTrace(task.public, actual)
        if _is_passing_observation(actual, task.public.expected):
            return "already_passing"

        state = None
        feedback = ""
        if strategy in (Strategy.LLM, Strategy.EIG):
            state = diagnose(failure, strategy, config, actions, tools, provider)

        while True:
            current = DebugTask(task.name, tools.source, task.public)
            if strategy == Strategy.FREE and actions.remaining >= REPAIR_ACTION_RESERVE + 2:
                probe = actions.call("free_choice", provider.free_probe, current, failure, feedback)
                if probe is not None:
                    obs = actions.call("probe", tools.run_experiment, probe)
                    feedback = _append_feedback(feedback, _format_probe_feedback(probe, obs))
                    continue

            patch = actions.call("repair", provider.repair, current, failure, state, feedback)
            applied = actions.call("apply", tools.apply_patch, patch)
            if applied.applied:
                if actions.call("public_verify", tools.verify_patch).passed:
                    return "public_pass"
                actual = actions.call(
                    "reproduce", tools.run_experiment, Experiment(task.public.argument)
                )
                failure = FailureTrace(task.public, actual)
                feedback = _append_feedback(feedback, _format_failure_feedback(failure))
            else:
                feedback = _append_feedback(feedback, f"[apply_failed] {applied.detail}")
    except WallClockExhausted:
        return "wall_clock_exhausted"
    except BudgetExhausted:
        return "budget_exhausted"
