"""Orchestrate B's pure functions and A's budgeted executor."""
from dataclasses import asdict
from active_debug.agent.agent import Actions
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.diagnosis import (belief_updater, experiment_generator, experiment_selector,
                                    hypothesis_generator, hypothesis_manager, stopping_policy)
from active_debug.schemas import DiagnosisState, FailureTrace, RunConfig, Strategy


def diagnose(trace: FailureTrace, strategy: Strategy, config: RunConfig, actions: Actions,
             tools: ToolExecutor, provider: Provider) -> DiagnosisState:
    state = hypothesis_manager.initialize(hypothesis_generator.generate(trace))
    actions.logger.log("prior", state=asdict(state))
    reserve = 3  # repair, apply, public verification
    while not stopping_policy.should_stop(state, config.confidence):
        options = experiment_generator.generate(state)
        required = 2 if strategy == Strategy.LLM else 1
        if not options or actions.limit - actions.used < reserve + required:
            break
        if strategy == Strategy.LLM:
            probe = actions.call("choose", provider.choose, state, options)
            if probe not in options:
                raise ValueError("provider selected an unknown experiment")
        else:
            probe = experiment_selector.select(options, state, config.cost_weight)
        obs = actions.call("probe", tools.run_experiment, probe)
        state = belief_updater.update(state, probe, obs)
        actions.logger.log("update", probe=asdict(probe), beliefs=[asdict(h) for h in state.hypotheses])
    return state
