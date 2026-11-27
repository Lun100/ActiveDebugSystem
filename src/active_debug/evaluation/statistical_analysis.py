from collections import defaultdict
from dataclasses import replace
from random import Random
from statistics import mean

from active_debug.evaluation.metrics import COST_FIELDS, RunKey, ScoredRun, summarize_strategy
from active_debug.schemas import RunConfig, Strategy


def _bootstrap_by_defect(groups: list[list[int]]) -> list[float]:
    """Resample defects with their seeds together to preserve paired observations."""
    rng = Random(0)
    samples = []
    for _ in range(2000):
        picked = rng.choices(groups, k=len(groups))
        samples.append(mean(delta for defect in picked for delta in defect))
    samples.sort()
    return [samples[49], samples[1949]]


def paired_analysis(keys: list[RunKey], left: dict[RunKey, ScoredRun],
                    right: dict[RunKey, ScoredRun]) -> dict:
    """Compare two strategies on the same scheduled revisions and seeds."""
    pairs = [(left.get(key), right.get(key)) for key in keys]
    comparisons = [(bool(a and a.passed), bool(b and b.passed)) for a, b in pairs]
    groups = defaultdict(list)
    for key, (a, b) in zip(keys, comparisons, strict=True):
        groups[(key.case_id, key.revision)].append(int(a) - int(b))
    observed = sum(a is not None and b is not None for a, b in pairs)
    costs = {}
    for field in COST_FIELDS:
        differences = []
        for left_run, right_run in pairs:
            a = getattr(left_run, field, None)
            b = getattr(right_run, field, None)
            if a is not None and b is not None:
                differences.append(a - b)
        costs[field] = {"pairs": len(differences),
                        "mean_difference": mean(differences) if differences else None}
    return {
        "planned_pairs": len(keys),
        "observed_pairs": observed,
        "left_only_passed": comparisons.count((True, False)),
        "right_only_passed": comparisons.count((False, True)),
        "both_passed": comparisons.count((True, True)),
        "neither_passed": comparisons.count((False, False)),
        "repair_rate_difference": mean(int(a) - int(b) for a, b in comparisons),
        "repair_rate_difference_ci95": (
            _bootstrap_by_defect(list(groups.values()))
            if observed == len(keys) and len(groups) > 1 else None
        ),
        "cost_differences": costs,
    }


def summarize_experiment(planned: set[RunKey], runs: list[ScoredRun]) -> dict:
    if not planned:
        raise ValueError("expected at least one scheduled task")
    indexed: dict[Strategy, dict[RunKey, ScoredRun]] = {strategy: {} for strategy in Strategy}
    for run in runs:
        if run.key not in planned or run.key in indexed[run.strategy]:
            raise ValueError(f"unexpected or duplicate run: {run.strategy} {run.key}")
        indexed[run.strategy][run.key] = run

    n = len(planned)
    summaries = {strategy: summarize_strategy(list(by_key.values()), n)
                 for strategy, by_key in indexed.items()}
    keys = sorted(planned)
    return {
        "complete": len(runs) == n * len(Strategy),
        "planned_runs": n * len(Strategy),
        "received_runs": len(runs),
        "strategies": summaries,
        "eig_comparisons": {
            strategy: paired_analysis(keys, indexed[Strategy.EIG], indexed[strategy])
            for strategy in Strategy if strategy != Strategy.EIG
        },
    }


def ablation_configs(base: RunConfig, weights: list[float]) -> list[RunConfig]:
    """Vary only the EIG cost penalty; keep all other run settings fixed."""
    return [replace(base, cost_weight=weight) for weight in weights]
