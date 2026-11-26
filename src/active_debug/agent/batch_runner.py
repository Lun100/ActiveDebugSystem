"""Batch experiment orchestrator producing plan.json, runs.jsonl, and candidate directories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, TextIO
from uuid import uuid4

from active_debug.agent.llm_client import OpenAICompatibleProvider
from active_debug.agent.suite_executor import BundleManifest, SuiteCase, load_bundle
from active_debug.agent.suite_runner import SuiteRunOutcome, run_suite_case
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import RunConfig, Strategy

ALL_STRATEGIES: tuple[Strategy, ...] = (
    Strategy.DIRECT,
    Strategy.FREE,
    Strategy.LLM,
    Strategy.EIG,
)


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate execution counts for a v3 bundle batch run."""

    experiment_id: str
    output_dir: str
    planned_runs: int
    executed_runs: int
    skipped_completed_runs: int
    completed_runs: int
    public_pass_runs: int
    budget_exhausted_runs: int
    timeout_runs: int
    error_runs: int


def _infer_model_snapshot(provider: Any, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if isinstance(provider, OpenAICompatibleProvider):
        return provider.model
    model_attr = getattr(provider, "model", None)
    if isinstance(model_attr, str) and model_attr.strip():
        return model_attr.strip()
    return type(provider).__name__


def _run_slug(seed: int, case: SuiteCase, strategy: Strategy) -> str:
    return f"seed{seed}-{case.slug}-{strategy}"


def _load_existing_runs(runs_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not runs_path.is_file():
        return {}
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw_line in runs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = (int(row["seed"]), str(row["case_id"]), str(row["strategy"]))
        records[key] = row
    return records


def _write_runs_atomic(runs_path: Path, rows: list[dict[str, Any]]) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = runs_path.with_name(f".{runs_path.name}.tmp")
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(runs_path)


def run_bundle_experiment(
    bundle_dir: Path,
    output_dir: Path,
    config: RunConfig,
    provider: Any,
    *,
    seeds: tuple[int, ...] = (0,),
    strategies: tuple[Strategy, ...] = ALL_STRATEGIES,
    case_ids: tuple[str, ...] | None = None,
    experiment_id: str | None = None,
    model_snapshot: str | None = None,
    sandbox: str = "auto",
    docker_image: str = "active-debug-pysnooper-1",
    wall_clock_limit: float | None = None,
    resume_batch: bool = False,
    progress_stream: TextIO | None = sys.stderr,
) -> BatchSummary:
    """Execute scheduled (seed, case, strategy) runs and emit C's 3 evaluation deliverables."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if not strategies:
        raise ValueError("at least one strategy is required")

    manifest: BundleManifest = load_bundle(bundle_dir)
    if case_ids:
        wanted = set(case_ids)
        selected_cases = tuple(c for c in manifest.cases if c.case_id in wanted)
        missing = wanted - {c.case_id for c in selected_cases}
        if missing:
            raise ValueError(f"unknown case_id(s) in bundle: {sorted(missing)}")
    else:
        selected_cases = manifest.cases

    out_root = output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    candidates_root = out_root / "candidates"
    traces_root = out_root / "traces"
    plan_path = out_root / "plan.json"
    runs_path = out_root / "runs.jsonl"

    existing_records: dict[tuple[int, str, str], dict[str, Any]] = {}
    resolved_exp_id = experiment_id
    if resume_batch and plan_path.is_file():
        old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not resolved_exp_id and isinstance(old_plan.get("experiment_id"), str):
            resolved_exp_id = old_plan["experiment_id"]
        existing_records = _load_existing_runs(runs_path)

    if not resolved_exp_id:
        resolved_exp_id = f"{manifest.name}-{uuid4().hex[:8]}"
    resolved_model = _infer_model_snapshot(provider, model_snapshot)

    planned_total = len(seeds) * len(selected_cases) * len(strategies)
    plan_payload = {
        "experiment_id": resolved_exp_id,
        "model_snapshot": resolved_model,
        "max_actions": config.max_actions,
        "seeds": list(seeds),
        "dataset_name": manifest.name,
        "dataset_revision": manifest.dataset_revision,
        "strategies": [str(s) for s in strategies],
        "cases": [c.case_id for c in selected_cases],
        "total_planned_runs": planned_total,
        "sandbox": sandbox,
        "config": asdict(config),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    plan_path.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ordered_keys: list[tuple[int, SuiteCase, Strategy]] = [
        (seed, case, strategy)
        for seed in seeds
        for case in selected_cases
        for strategy in strategies
    ]

    records_by_key = dict(existing_records)
    executed_count = 0
    skipped_count = 0

    for idx, (seed, case, strategy) in enumerate(ordered_keys, start=1):
        key = (seed, case.case_id, str(strategy))
        prev = records_by_key.get(key)
        if resume_batch and prev is not None and prev.get("status") == "completed":
            skipped_count += 1
            if progress_stream is not None:
                progress_stream.write(
                    f"[{idx}/{planned_total}] SKIP completed seed={seed} "
                    f"case={case.case_id} strategy={strategy}\n"
                )
                progress_stream.flush()
            continue

        slug = _run_slug(seed, case, strategy)
        trace_path = traces_root / f"{slug}.jsonl"
        logger = TraceLogger(trace_path, allow_append=resume_batch or trace_path.exists())

        outcome: SuiteRunOutcome = run_suite_case(
            case,
            strategy,
            config,
            provider,
            logger,
            seed=seed,
            sandbox=sandbox,
            docker_image=docker_image,
            wall_clock_limit=wall_clock_limit,
        )
        executed_count += 1

        cand_dir = candidates_root / slug
        rel_candidate_dir: str | None = None
        if outcome.source_modified:
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / "buggy.py").write_text(outcome.final_source, encoding="utf-8")
            rel_candidate_dir = f"candidates/{slug}"
        elif cand_dir.exists():
            shutil.rmtree(cand_dir)

        row: dict[str, Any] = {
            "experiment_id": resolved_exp_id,
            "case_id": outcome.case_id,
            "revision": outcome.revision,
            "seed": outcome.seed,
            "strategy": str(outcome.strategy),
            "status": outcome.status,
            "actions": outcome.actions,
            "seconds": round(outcome.seconds, 6),
            "stop_reason": outcome.stop_reason,
            "candidate_dir": rel_candidate_dir,
            "predicted_function": outcome.predicted_function,
        }
        if outcome.error_detail is not None:
            row["error_detail"] = outcome.error_detail

        records_by_key[key] = row
        # Persist runs.jsonl incrementally after every single run so progress is crash-safe
        ordered_rows = [
            records_by_key[(s, c.case_id, str(st))]
            for s, c, st in ordered_keys
            if (s, c.case_id, str(st)) in records_by_key
        ]
        _write_runs_atomic(runs_path, ordered_rows)

        if progress_stream is not None:
            progress_stream.write(
                f"[{idx}/{planned_total}] seed={seed} case={case.case_id} "
                f"strategy={strategy} -> status={outcome.status} "
                f"stop={outcome.stop_reason} actions={outcome.actions} "
                f"({outcome.seconds:.2f}s)\n"
            )
            progress_stream.flush()

    final_rows = [
        records_by_key[(s, c.case_id, str(st))]
        for s, c, st in ordered_keys
        if (s, c.case_id, str(st)) in records_by_key
    ]
    _write_runs_atomic(runs_path, final_rows)

    completed_runs = sum(1 for r in final_rows if r["status"] == "completed")
    public_pass_runs = sum(1 for r in final_rows if r["stop_reason"] == "public_pass")
    budget_exhausted_runs = sum(1 for r in final_rows if r["stop_reason"] == "budget_exhausted")
    timeout_runs = sum(1 for r in final_rows if r["status"] == "timeout")
    error_runs = sum(1 for r in final_rows if r["status"] == "error")

    return BatchSummary(
        experiment_id=resolved_exp_id,
        output_dir=str(out_root),
        planned_runs=planned_total,
        executed_runs=executed_count,
        skipped_completed_runs=skipped_count,
        completed_runs=completed_runs,
        public_pass_runs=public_pass_runs,
        budget_exhausted_runs=budget_exhausted_runs,
        timeout_runs=timeout_runs,
        error_runs=error_runs,
    )
