"""Atomic JSON checkpointing and state recovery for agent runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import os
from pathlib import Path
from typing import Any

from active_debug.schemas import (
    Case,
    DebugTask,
    DiagnosisState,
    Experiment,
    FailureTrace,
    Hypothesis,
    Observation,
    Strategy,
)

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base error for checkpoint serialization or recovery problems."""


class CheckpointMismatchError(CheckpointError):
    """Raised when resuming a checkpoint against a different task or strategy."""


class InterruptPolicy(StrEnum):
    """Accounting rule when an in-flight action raises before completing."""

    CHARGE_IN_FLIGHT = "charge_in_flight"
    ROLLBACK_IN_FLIGHT = "rollback_in_flight"


@dataclass
class CheckpointData:
    """Complete recoverable snapshot of an active debugging session."""

    task_name: str
    public_case: Case
    strategy: Strategy
    phase: str  # "init" | "diagnosing" | "repairing" | "completed"
    source: str
    actions_used: int
    committed_actions_used: int
    actions_limit: int
    failure: FailureTrace | None = None
    diagnosis_state: DiagnosisState | None = None
    feedback: str = ""
    pending_probe: Experiment | None = None
    pending_free_probe: Experiment | None = None
    free_probing_done: bool = False
    pending_patch: str | None = None
    applied_unverified: bool = False
    stop_reason: str | None = None
    interrupted: bool = False
    last_error: str | None = None
    trace_path: str | None = None

    def resumed_actions_used(self, policy: InterruptPolicy) -> int:
        if self.interrupted and policy == InterruptPolicy.ROLLBACK_IN_FLIGHT:
            return self.committed_actions_used
        return self.actions_used


def _obs_to_dict(obs: Observation) -> dict[str, Any]:
    return asdict(obs)


def _obs_from_dict(data: dict[str, Any]) -> Observation:
    return Observation(
        stdout=str(data["stdout"]),
        stderr=str(data["stderr"]),
        returncode=int(data["returncode"]),
        timed_out=bool(data.get("timed_out", False)),
    )


def _exp_to_dict(exp: Experiment) -> dict[str, Any]:
    return asdict(exp)


def _exp_from_dict(data: dict[str, Any]) -> Experiment:
    preds = {str(k): str(v) for k, v in dict(data.get("predictions", {})).items()}
    return Experiment(
        argument=int(data["argument"]),
        predictions=preds,
        cost=float(data.get("cost", 1.0)),
    )


def _failure_from_dict(data: dict[str, Any]) -> FailureTrace:
    case_raw = data["case"]
    return FailureTrace(
        case=Case(int(case_raw["argument"]), int(case_raw["expected"])),
        actual=_obs_from_dict(data["actual"]),
    )


def _diagnosis_to_dict(state: DiagnosisState) -> dict[str, Any]:
    return {
        "hypotheses": [asdict(h) for h in state.hypotheses],
        "history": [
            {"experiment": _exp_to_dict(exp), "observation": _obs_to_dict(obs)}
            for exp, obs in state.history
        ],
    }


def _diagnosis_from_dict(data: dict[str, Any]) -> DiagnosisState:
    hypotheses = [
        Hypothesis(str(h["name"]), float(h["belief"])) for h in data["hypotheses"]
    ]
    history = [
        (_exp_from_dict(item["experiment"]), _obs_from_dict(item["observation"]))
        for item in data.get("history", [])
    ]
    return DiagnosisState(hypotheses=hypotheses, history=history)


class CheckpointStore:
    """Persist and load CheckpointData atomically via temp-file rename."""

    def __init__(
        self,
        path: Path,
        *,
        interrupt_policy: InterruptPolicy = InterruptPolicy.CHARGE_IN_FLIGHT,
    ):
        self.path = path
        self.interrupt_policy = interrupt_policy

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, data: CheckpointData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "task_name": data.task_name,
            "public_case": asdict(data.public_case),
            "strategy": str(data.strategy),
            "phase": data.phase,
            "source": data.source,
            "actions_used": data.actions_used,
            "committed_actions_used": data.committed_actions_used,
            "actions_limit": data.actions_limit,
            "failure": asdict(data.failure) if data.failure is not None else None,
            "diagnosis_state": (
                _diagnosis_to_dict(data.diagnosis_state)
                if data.diagnosis_state is not None
                else None
            ),
            "feedback": data.feedback,
            "pending_probe": (
                _exp_to_dict(data.pending_probe) if data.pending_probe is not None else None
            ),
            "pending_free_probe": (
                _exp_to_dict(data.pending_free_probe)
                if data.pending_free_probe is not None
                else None
            ),
            "free_probing_done": data.free_probing_done,
            "pending_patch": data.pending_patch,
            "applied_unverified": data.applied_unverified,
            "stop_reason": data.stop_reason,
            "interrupted": data.interrupted,
            "last_error": data.last_error,
            "trace_path": data.trace_path,
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def load(self) -> CheckpointData:
        if not self.path.exists():
            raise CheckpointError(f"checkpoint file not found: {self.path}")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointError(f"corrupted checkpoint JSON: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint schema version")
        try:
            case_raw = raw["public_case"]
            return CheckpointData(
                task_name=str(raw["task_name"]),
                public_case=Case(int(case_raw["argument"]), int(case_raw["expected"])),
                strategy=Strategy(raw["strategy"]),
                phase=str(raw["phase"]),
                source=str(raw["source"]),
                actions_used=int(raw["actions_used"]),
                committed_actions_used=int(raw["committed_actions_used"]),
                actions_limit=int(raw["actions_limit"]),
                failure=(
                    _failure_from_dict(raw["failure"]) if raw.get("failure") is not None else None
                ),
                diagnosis_state=(
                    _diagnosis_from_dict(raw["diagnosis_state"])
                    if raw.get("diagnosis_state") is not None
                    else None
                ),
                feedback=str(raw.get("feedback", "")),
                pending_probe=(
                    _exp_from_dict(raw["pending_probe"])
                    if raw.get("pending_probe") is not None
                    else None
                ),
                pending_free_probe=(
                    _exp_from_dict(raw["pending_free_probe"])
                    if raw.get("pending_free_probe") is not None
                    else None
                ),
                free_probing_done=bool(raw.get("free_probing_done", False)),
                pending_patch=(
                    str(raw["pending_patch"]) if raw.get("pending_patch") is not None else None
                ),
                applied_unverified=bool(raw.get("applied_unverified", False)),
                stop_reason=(
                    str(raw["stop_reason"]) if raw.get("stop_reason") is not None else None
                ),
                interrupted=bool(raw.get("interrupted", False)),
                last_error=(
                    str(raw["last_error"]) if raw.get("last_error") is not None else None
                ),
                trace_path=(
                    str(raw["trace_path"]) if raw.get("trace_path") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint payload: {exc}") from exc

    def validate_compatibility(
        self, data: CheckpointData, task: DebugTask, strategy: Strategy
    ) -> None:
        if data.task_name != task.name or data.public_case != task.public:
            raise CheckpointMismatchError(
                f"checkpoint task ({data.task_name}, {data.public_case}) does not match "
                f"target task ({task.name}, {task.public})"
            )
        if data.strategy != strategy:
            raise CheckpointMismatchError(
                f"checkpoint strategy {data.strategy!r} does not match {strategy!r}"
            )
