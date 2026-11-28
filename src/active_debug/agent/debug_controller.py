"""Orchestrate B's pure diagnosis functions and A's budgeted tool executor."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from active_debug.agent.agent import Actions
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.diagnosis import (
    belief_updater,
    experiment_generator,
    experiment_selector,
    hypothesis_generator,
    hypothesis_manager,
    stopping_policy,
)
from active_debug.schemas import (
    DiagnosisState,
    Experiment,
    FailureTrace,
    RunConfig,
    Strategy,
)

REPAIR_ACTION_RESERVE = 3  # repair + apply + public_verify
DiagnosisCheckpointHook = Callable[[DiagnosisState, Experiment | None], None]


def _resolve_chosen_option(chosen: Experiment, options: list[Experiment]) -> Experiment:
    if chosen in options:
        return chosen
    if isinstance(chosen, Experiment) and not chosen.predictions:
        matches = [opt for opt in options if opt.argument == chosen.argument]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("provider selected an unknown experiment")


def diagnose(
    trace: FailureTrace,
    strategy: Strategy,
    config: RunConfig,
    actions: Actions,
    tools: ToolExecutor,
    provider: Provider,
    *,
    initial_state: DiagnosisState | None = None,
    pending_probe: Experiment | None = None,
    on_checkpoint: DiagnosisCheckpointHook | None = None,
) -> DiagnosisState:
    if strategy not in (Strategy.LLM, Strategy.EIG):
        raise ValueError(f"diagnose requires hypothesis strategy, got {strategy!r}")

    if initial_state is None:
        state = hypothesis_manager.initialize(hypothesis_generator.generate(trace))
        actions.logger.log("prior", state=asdict(state))
        if on_checkpoint is not None:
            on_checkpoint(state, None)
    else:
        state = initial_state

    queued_probe = pending_probe
    while not stopping_policy.should_stop(state, config.confidence):
        if queued_probe is not None:
            if actions.remaining < REPAIR_ACTION_RESERVE + 1:
                break
            probe = queued_probe
            queued_probe = None
        else:
            options = experiment_selector.worthwhile(
                experiment_generator.generate(state), state, config.cost_weight
            )
            required = 2 if strategy == Strategy.LLM else 1
            if not options or actions.remaining < REPAIR_ACTION_RESERVE + required:
                break
            if strategy == Strategy.LLM:
                raw_choice = actions.call("choose", provider.choose, state, options)
                probe = _resolve_chosen_option(raw_choice, options)
                if on_checkpoint is not None:
                    on_checkpoint(state, probe)
            else:
                probe = experiment_selector.select(options, state, config.cost_weight)

        obs = actions.call("probe", tools.run_experiment, probe)
        unexplained = belief_updater.is_unexplained(probe, obs)
        state = belief_updater.update(state, probe, obs)
        actions.logger.log(
            "update",
            probe=asdict(probe),
            observation=asdict(obs),
            unexplained=unexplained,
            beliefs=[asdict(h) for h in state.hypotheses],
        )
        if unexplained:
            # Without a provider-side re-plan hook here, stop probing: running
            # more probes against an unchanged hypothesis set cannot help.
            actions.logger.log(
                "anomaly",
                probe=asdict(probe),
                observation=asdict(obs),
                beliefs=[asdict(h) for h in state.hypotheses],
            )
            if on_checkpoint is not None:
                on_checkpoint(state, None)
            break
        if on_checkpoint is not None:
            on_checkpoint(state, None)
    return state
