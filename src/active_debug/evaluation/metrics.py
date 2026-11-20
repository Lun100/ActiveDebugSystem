from active_debug.schemas import RunResult


def summarize(results: list[RunResult]) -> dict[str, float]:
    if not results:
        raise ValueError("no completed runs")
    n = len(results)
    return {"success_rate": sum(r.validation.passed for r in results) / n,
            "mean_actions": sum(r.actions for r in results) / n,
            "mean_seconds": sum(r.seconds for r in results) / n}
