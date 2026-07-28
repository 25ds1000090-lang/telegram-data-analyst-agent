from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIRECTORY = Path("logs")
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()

# Retain logs in memory as well as on disk. This helps /run.jsonl continue
# working while the Render instance remains active.
_memory_logs: dict[str, list[str]] = {}
_latest_run_id: str | None = None

# Prevent unbounded memory growth.
MAX_MEMORY_LOGS = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_json_safe(value: Any) -> Any:
    """
    Convert values into JSON-compatible structures without exposing objects
    that json.dumps cannot serialise.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    return str(value)


class RunLogger:
    def __init__(
        self,
        *,
        chat_id: int,
        update_id: int,
        message_id: int,
    ) -> None:
        global _latest_run_id

        self.run_id = uuid.uuid4().hex
        self.path = LOG_DIRECTORY / f"{self.run_id}.jsonl"
        self.sequence = 0

        with _lock:
            _memory_logs[self.run_id] = []
            _latest_run_id = self.run_id
            self._prune_memory_logs()

        self.write(
            "run_started",
            {
                "run_id": self.run_id,
                "chat_id": chat_id,
                "update_id": update_id,
                "message_id": message_id,
            },
        )

    @staticmethod
    def _prune_memory_logs() -> None:
        while len(_memory_logs) > MAX_MEMORY_LOGS:
            oldest_run_id = next(iter(_memory_logs))
            del _memory_logs[oldest_run_id]

    def write(
        self,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.sequence += 1

        record = {
            "timestamp": utc_now(),
            "sequence": self.sequence,
            "event": event,
            "data": make_json_safe(data or {}),
        }

        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

        with _lock:
            _memory_logs.setdefault(self.run_id, []).append(line)

            try:
                with self.path.open("a", encoding="utf-8") as log_file:
                    log_file.write(line)
                    log_file.write("\n")
            except OSError:
                # The public log can still be served from memory.
                pass


def get_log_text(run_id: str | None) -> str | None:
    """
    Return one complete JSONL log.

    When run_id is None, return the latest run.
    """
    with _lock:
        selected_run_id = run_id or _latest_run_id

        if selected_run_id is None:
            return None

        memory_lines = _memory_logs.get(selected_run_id)

        if memory_lines is not None:
            return "\n".join(memory_lines) + ("\n" if memory_lines else "")

    path = LOG_DIRECTORY / f"{selected_run_id}.jsonl"

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
