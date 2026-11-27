"""Evaluate final candidates from the demo or recorded project runs."""
from collections.abc import Callable
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path, PurePosixPath
from time import monotonic
from uuid import uuid4

from active_debug.agent.agent import Actions
from active_debug.agent.llm_client import Provider
from active_debug.agent.tool_executor import ToolExecutor
from active_debug.agent.trace_logger import TraceLogger
from active_debug.evaluation.dataset_loader import read_json
from active_debug.evaluation.metrics import RunKey, ScoredRun
from active_debug.evaluation.patch_validator import validate, validate_project
from active_debug.evaluation.statistical_analysis import summarize_experiment
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


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _optional_cost(row: dict, name: str, *, integer: bool = True) -> int | float | None:
    value = row.get(name)
    if value is not None:
        allowed = (int,) if integer else (int, float)
        if type(value) not in allowed or not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a nonnegative {'integer' if integer else 'number'}")
    return value


def _parse_rows(path: Path, experiment: str,
                planned: set[RunKey]) -> list[tuple[RunKey, Strategy, dict]]:
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["experiment_id"] != experiment:
            raise ValueError(f"wrong experiment on line {line_number}")
        if type(row["seed"]) is not int:
            raise ValueError(f"seed must be an integer on line {line_number}")
        key = RunKey(row["case_id"], row["revision"], row["seed"])
        strategy = Strategy(row["strategy"])
        identity = (key, strategy)
        if key not in planned or identity in seen:
            raise ValueError(f"unplanned or duplicate run on line {line_number}")
        seen.add(identity)
        records.append((key, strategy, row))
    return records


def _score_project_run(key: RunKey, strategy: Strategy, row: dict, *, case: dict, label: dict,
                       bundle: Path, private: Path, image: str, runs_dir: Path,
                       max_actions: int) -> tuple[ScoredRun, dict | None]:
    status = row["status"]
    if status not in ("completed", "timeout", "error", "missing"):
        raise ValueError(f"unknown run status: {status}")
    actions = _optional_cost(row, "actions")
    if actions is not None and actions > max_actions:
        raise ValueError(f"recorded actions exceed the experiment budget for {key}")
    seconds = _optional_cost(row, "seconds", integer=False)
    input_tokens = _optional_cost(row, "input_tokens")
    output_tokens = _optional_cost(row, "output_tokens")
    tokens = None if input_tokens is None or output_tokens is None else input_tokens + output_tokens
    location_correct = None
    if "location_functions" in label:
        prediction = row.get("predicted_function")
        if prediction is not None:
            location_correct = prediction in label["location_functions"]
    else:
        prediction = row.get("predicted_file")
        if prediction is not None:
            location_correct = PurePosixPath(prediction.replace("\\", "/")).as_posix() in label["location_files"]

    passed = False
    validation = None
    if status == "completed":
        supplied = row["candidate_dir"]
        candidate = bundle / case["source"] if supplied is None else runs_dir / supplied
        public = bundle / case["public_suite"]
        held_out = private / "held-out" / case["public_suite"].split("/")[0]
        started = monotonic()
        try:
            verdict = validate_project(candidate, public, held_out, image=image)
        except (OSError, ValueError):
            verdict = None
        if seconds is not None:
            seconds += monotonic() - started
        if verdict is None:
            status = "error"
        else:
            validation = asdict(verdict)
            passed = verdict.passed
            statuses = (verdict.public, verdict.held_out)
            if "error" in statuses:
                status = "error"
            elif "timeout" in statuses:
                status = "timeout"

    return ScoredRun(key, strategy, status, passed, row.get("stop_reason", status),
                     actions, seconds, tokens, location_correct), validation


def benchmark_project_runs(bundle: Path, private: Path, plan_path: Path, runs_path: Path) -> dict:
    """Score recorded candidates and summarize their scheduled comparisons."""
    manifest = read_json(bundle / "manifest.json")
    preflight = read_json(private / "preflight.json")
    plan = read_json(plan_path)
    if not plan["experiment_id"] or not plan["model_snapshot"] or not _positive_integer(plan["max_actions"]):
        raise ValueError("the plan needs an experiment ID, model snapshot and positive action budget")
    seeds = plan["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("the plan needs unique integer seeds")
    if any(preflight[key] != manifest[key] for key in ("dataset_revision", "image")):
        raise ValueError("public bundle and private preflight use different dataset or image versions")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    labels = {row["case_id"]: row for row in preflight["cases"] if row["qualified"]}
    if cases.keys() != labels.keys() or any(
        case["revision"] != labels[case_id]["revision"] for case_id, case in cases.items()
    ):
        raise ValueError("the private preflight does not cover this public bundle")

    planned = {RunKey(case_id, case["revision"], seed)
               for case_id, case in cases.items() for seed in seeds}
    runs = []
    details = []
    for key, strategy, row in _parse_rows(runs_path, plan["experiment_id"], planned):
        run, verdict = _score_project_run(
            key, strategy, row, case=cases[key.case_id], label=labels[key.case_id],
            bundle=bundle, private=private, image=manifest["image"],
            runs_dir=runs_path.parent, max_actions=plan["max_actions"],
        )
        runs.append(run)
        details.append({**asdict(run), "validation": verdict})

    return {"plan": plan, "benchmark": manifest["name"], "dataset_revision": manifest["dataset_revision"],
            **summarize_experiment(planned, runs), "runs": details}
