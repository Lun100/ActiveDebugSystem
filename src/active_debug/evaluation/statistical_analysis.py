from dataclasses import replace
from active_debug.schemas import RunConfig, RunResult


def paired_analysis(left: list[RunResult], right: list[RunResult]) -> dict[str, float]:
    """Future paired comparison with task/seed matching and uncertainty estimates."""
    raise NotImplementedError("Define paired sampling protocol before inference")


def ablation_configs(base: RunConfig, weights: list[float]) -> list[RunConfig]:
    """Vary only the EIG cost penalty; keep all other run settings fixed."""
    return [replace(base, cost_weight=weight) for weight in weights]
