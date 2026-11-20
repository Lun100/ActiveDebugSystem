from active_debug.schemas import Case, DebugTask


def eligible(task: DebugTask, hidden: list[Case]) -> bool:
    return bool(hidden) and all(c.argument != task.public.argument for c in hidden)
