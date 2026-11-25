"""Common reproduce -> (diagnose / free-probe) -> repair -> verify loop with checkpointing."""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

from active_debug.agent.agent import Actions, BudgetExhausted, WallClockExhausted
from active_debug.agent.checkpoint import CheckpointData, CheckpointError, CheckpointStore
from active_debug.agent.debug_controller import REPAIR_ACTION_RESERVE, diagnose
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import (
    DebugTask,
    DiagnosisState,
    Experiment,
    FailureTrace,
    Observation,
    Patch,
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
    *,
    checkpoint_store: CheckpointStore | None = None,
    resume: bool = False,
) -> str:
    phase = "init"
    failure: FailureTrace | None = None
    state: DiagnosisState | None = None
    feedback = ""
    pending_probe: Experiment | None = None
    pending_free_probe: Experiment | None = None
    free_probing_done = False
    pending_patch: str | None = None
    applied_unverified = False
    committed_used = actions.used

    def _save_snapshot(
        *,
        stop_reason: str | None = None,
        interrupted: bool = False,
        last_error: str | None = None,
    ) -> None:
        if checkpoint_store is None:
            return
        checkpoint_store.save(
            CheckpointData(
                task_name=task.name,
                public_case=task.public,
                strategy=strategy,
                phase=phase,
                source=tools.source,
                actions_used=actions.used,
                committed_actions_used=committed_used,
                actions_limit=actions.limit,
                failure=failure,
                diagnosis_state=state,
                feedback=feedback,
                pending_probe=pending_probe,
                pending_free_probe=pending_free_probe,
                free_probing_done=free_probing_done,
                pending_patch=pending_patch,
                applied_unverified=applied_unverified,
                stop_reason=stop_reason,
                interrupted=interrupted,
                last_error=last_error,
                trace_path=str(actions.logger.path),
            )
        )

    if resume:
        if checkpoint_store is None:
            raise CheckpointError("resume=True requires a checkpoint_store")
        ckpt = checkpoint_store.load()
        checkpoint_store.validate_compatibility(ckpt, task, strategy)
        restored_used = ckpt.resumed_actions_used(checkpoint_store.interrupt_policy)
        if restored_used > actions.limit:
            raise CheckpointError(
                f"checkpoint actions_used ({restored_used}) exceeds run limit ({actions.limit})"
            )
        tools.source = ckpt.source
        actions.used = restored_used
        committed_used = restored_used
        phase = ckpt.phase
        failure = ckpt.failure
        state = ckpt.diagnosis_state
        feedback = ckpt.feedback
        pending_probe = ckpt.pending_probe
        pending_free_probe = ckpt.pending_free_probe
        free_probing_done = ckpt.free_probing_done
        pending_patch = ckpt.pending_patch
        applied_unverified = ckpt.applied_unverified

        actions.logger.log(
            "resume",
            phase=phase,
            restored_actions_used=restored_used,
            interrupt_policy=str(checkpoint_store.interrupt_policy),
            previous_trace_path=ckpt.trace_path,
        )
        if phase == "completed" and ckpt.stop_reason in ("public_pass", "already_passing"):
            return ckpt.stop_reason

    try:
        if failure is None:
            actual = actions.call(
                "reproduce", tools.run_experiment, Experiment(task.public.argument)
            )
            committed_used = actions.used
            failure = FailureTrace(task.public, actual)
            if _is_passing_observation(actual, task.public.expected):
                phase = "completed"
                _save_snapshot(stop_reason="already_passing")
                return "already_passing"
            phase = "diagnosing" if strategy in (Strategy.LLM, Strategy.EIG) else "repairing"
            _save_snapshot()

        if strategy in (Strategy.LLM, Strategy.EIG) and phase == "diagnosing":
            def _on_diag_checkpoint(
                new_state: DiagnosisState, next_probe: Experiment | None
            ) -> None:
                nonlocal state, pending_probe, committed_used
                state = new_state
                pending_probe = next_probe
                committed_used = actions.used
                _save_snapshot()

            state = diagnose(
                failure,
                strategy,
                config,
                actions,
                tools,
                provider,
                initial_state=state,
                pending_probe=pending_probe,
                on_checkpoint=_on_diag_checkpoint,
            )
            pending_probe = None
            phase = "repairing"
            committed_used = actions.used
            _save_snapshot()

        while True:
            current = DebugTask(task.name, tools.source, task.public)

            if not applied_unverified and pending_patch is None:
                if strategy == Strategy.FREE and not free_probing_done:
                    if pending_free_probe is not None:
                        if actions.remaining >= REPAIR_ACTION_RESERVE + 1:
                            probe = pending_free_probe
                            pending_free_probe = None
                            obs = actions.call("probe", tools.run_experiment, probe)
                            feedback = _append_feedback(
                                feedback, _format_probe_feedback(probe, obs)
                            )
                            committed_used = actions.used
                            _save_snapshot()
                            continue
                        pending_free_probe = None
                    elif actions.remaining >= REPAIR_ACTION_RESERVE + 2:
                        probe = actions.call(
                            "free_choice", provider.free_probe, current, failure, feedback
                        )
                        committed_used = actions.used
                        if probe is not None:
                            pending_free_probe = probe
                            _save_snapshot()
                            obs = actions.call("probe", tools.run_experiment, probe)
                            pending_free_probe = None
                            feedback = _append_feedback(
                                feedback, _format_probe_feedback(probe, obs)
                            )
                            committed_used = actions.used
                            _save_snapshot()
                            continue
                        free_probing_done = True
                        _save_snapshot()

                patch = actions.call("repair", provider.repair, current, failure, state, feedback)
                pending_patch = patch.source
                committed_used = actions.used
                _save_snapshot()

            if not applied_unverified and pending_patch is not None:
                applied = actions.call("apply", tools.apply_patch, Patch(pending_patch))
                pending_patch = None
                committed_used = actions.used
                if not applied.applied:
                    feedback = _append_feedback(feedback, f"[apply_failed] {applied.detail}")
                    _save_snapshot()
                    continue
                applied_unverified = True
                _save_snapshot()

            if applied_unverified:
                verified = actions.call("public_verify", tools.verify_patch)
                applied_unverified = False
                committed_used = actions.used
                if verified.passed:
                    phase = "completed"
                    _save_snapshot(stop_reason="public_pass")
                    return "public_pass"
                _save_snapshot()
                actual = actions.call(
                    "reproduce", tools.run_experiment, Experiment(task.public.argument)
                )
                failure = FailureTrace(task.public, actual)
                feedback = _append_feedback(feedback, _format_failure_feedback(failure))
                committed_used = actions.used
                _save_snapshot()
    except WallClockExhausted:
        _save_snapshot(stop_reason="wall_clock_exhausted")
        return "wall_clock_exhausted"
    except BudgetExhausted:
        _save_snapshot(stop_reason="budget_exhausted")
        return "budget_exhausted"
    except BaseException as exc:
        _save_snapshot(interrupted=True, last_error=f"{type(exc).__name__}: {exc}")
        raise


def make_runner(
    *,
    checkpoint_store: CheckpointStore | None = None,
    resume: bool = False,
) -> Callable[[DebugTask, Strategy, RunConfig, Actions, ToolExecutor, Provider], str]:
    """Bind checkpoint/resume settings into a benchmark-compatible Runner callable."""
    return partial(run, checkpoint_store=checkpoint_store, resume=resume)
