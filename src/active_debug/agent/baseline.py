"""Common repair/verify loop for all four strategies."""
from active_debug.agent.agent import Actions, BudgetExhausted
from active_debug.agent.debug_controller import diagnose
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.schemas import DebugTask, Experiment, FailureTrace, RunConfig, Strategy


def run(task: DebugTask, strategy: Strategy, config: RunConfig, actions: Actions,
        tools: ToolExecutor, provider: Provider) -> str:
    try:
        actual = actions.call("reproduce", tools.run_experiment, Experiment(task.public.argument))
        failure = FailureTrace(task.public, actual)
        if actual.returncode == 0 and actual.stdout.strip() == str(task.public.expected):
            return "already_passing"
        state = None
        feedback = ""
        if strategy in (Strategy.LLM, Strategy.EIG):
            state = diagnose(failure, strategy, config, actions, tools, provider)
        while True:
            current = DebugTask(task.name, tools.source, task.public)
            if strategy == Strategy.FREE:
                probe = actions.call("free_choice", provider.free_probe, current, failure, feedback)
                if probe is not None:
                    obs = actions.call("probe", tools.run_experiment, probe)
                    feedback += repr((probe, obs))
                    continue
            patch = actions.call("repair", provider.repair, current, failure, state, feedback)
            applied = actions.call("apply", tools.apply_patch, patch)
            if applied.applied:
                if actions.call("public_verify", tools.verify_patch).passed:
                    return "public_pass"
                actual = actions.call("reproduce", tools.run_experiment,
                                      Experiment(task.public.argument))
                failure = FailureTrace(task.public, actual)
                feedback += repr(failure)
            else:
                feedback += applied.detail
    except BudgetExhausted:
        return "budget_exhausted"
