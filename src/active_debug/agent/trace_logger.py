"""One exclusively created JSONL file per run."""
import json
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)
        self.path = path

    def log(self, event: str, **fields: object) -> None:
        row = dict(time=datetime.now(timezone.utc).isoformat(), event=event, **fields)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
