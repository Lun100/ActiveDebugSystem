"""Meter every provider/tool action at the common dispatch boundary."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
import time
from typing import Any, TypeVar

from active_debug.agent.llm_client import CallUsage
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import Experiment

T = TypeVar("T")


class BudgetExhausted(Exception):
    """No further agent action allowed."""


class WallClockExhausted(BudgetExhausted):
    """Wall-clock deadline reached before dispatching next action."""


class Actions:
    def __init__(
        self,
        limit: int,
        logger: TraceLogger,
        *,
        start_used: int = 0,
        wall_clock_limit: float | None = None,
    ):
        if limit < 1:
            raise ValueError("action limit must be positive")
        if start_used < 0 or start_used > limit:
            raise ValueError("start_used out of bounds")
        if wall_clock_limit is not None and wall_clock_limit <= 0:
            raise ValueError("wall_clock_limit must be positive when set")

        self.limit = limit
        self.used = start_used
        self.logger = logger
        self.wall_clock_limit = wall_clock_limit
        self._started_at = time.monotonic()

        self.total_duration_ms = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_retries = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def usage_summary(self) -> dict[str, int | float]:
        return {
            "actions_used": self.used,
            "actions_limit": self.limit,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_retries": self.total_retries,
        }

    def call(self, name: str, operation: Callable[..., T], *args: object) -> T:
        if self.used >= self.limit:
            raise BudgetExhausted("action_budget_exhausted")
        if self.wall_clock_limit is not None:
            if time.monotonic() - self._started_at >= self.wall_clock_limit:
                raise WallClockExhausted("wall_clock_exhausted")

        self.used += 1
        details: dict[str, Any] = (
            {"probe": asdict(args[0])} if args and isinstance(args[0], Experiment) else {}
        )
        self.logger.log("action", index=self.used, name=name, **details)

        owner = getattr(operation, "__self__", None)
        if owner is not None and hasattr(owner, "last_usage"):
            owner.last_usage = None

        t0 = time.monotonic()
        try:
            result = operation(*args)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self.total_duration_ms += elapsed_ms
            self.logger.log(
                "action_error",
                index=self.used,
                name=name,
                duration_ms=round(elapsed_ms, 3),
                kind=type(exc).__name__,
                detail=str(exc),
            )
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self.total_duration_ms += elapsed_ms

        extra: dict[str, Any] = {"duration_ms": round(elapsed_ms, 3)}
        usage = getattr(owner, "last_usage", None) if owner is not None else None
        if isinstance(usage, CallUsage):
            extra["usage"] = usage.to_dict()
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.total_retries += usage.retries

        self.logger.log(
            "response",
            index=self.used,
            value=asdict(result) if is_dataclass(result) else result,
            **extra,
        )
        return result
