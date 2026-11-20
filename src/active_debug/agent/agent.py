"""Meter every provider/tool action at the common dispatch boundary."""
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import TypeVar
from active_debug.agent.trace_logger import TraceLogger
from active_debug.schemas import Experiment

T = TypeVar("T")


class BudgetExhausted(Exception):
    """No further agent action allowed."""


class Actions:
    def __init__(self, limit: int, logger: TraceLogger):
        self.limit = limit
        self.used = 0
        self.logger = logger

    def call(self, name: str, operation: Callable[..., T], *args: object) -> T:
        if self.used >= self.limit:
            raise BudgetExhausted
        self.used += 1
        details = {"probe": asdict(args[0])} if args and isinstance(args[0], Experiment) else {}
        self.logger.log("action", index=self.used, name=name, **details)
        result = operation(*args)
        self.logger.log("response", index=self.used,
                        value=asdict(result) if is_dataclass(result) else result)
        return result
