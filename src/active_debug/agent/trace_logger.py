"""Structured JSONL trace logger with secret redaction and replay reader."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

_SECRET_ENV_VARS = ("ACTIVE_DEBUG_API_KEY", "OPENAI_API_KEY")


class TraceLogger:
    def __init__(
        self,
        path: Path,
        *,
        allow_append: bool = False,
        redact_secrets: tuple[str, ...] = (),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if allow_append:
            path.touch(exist_ok=True)
        else:
            path.touch(exist_ok=False)
        self.path = path
        self._extra_secrets = tuple(s for s in redact_secrets if s)

    def _collect_secrets(self) -> list[str]:
        secrets = list(self._extra_secrets)
        for env_key in _SECRET_ENV_VARS:
            val = os.environ.get(env_key, "").strip()
            if val:
                secrets.append(val)
        return secrets

    def log(self, event: str, **fields: object) -> None:
        row = dict(time=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        serialized = json.dumps(row, ensure_ascii=False)
        for secret in self._collect_secrets():
            serialized = serialized.replace(secret, "***")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                events.append(json.loads(stripped))
        return events
