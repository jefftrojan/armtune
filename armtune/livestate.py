"""Thread-safe progress state shared between the sweep loop (writer, on the
main thread) and the live HTTP server (reader, polled by the browser) when
`armtune sweep --serve` is used."""

from __future__ import annotations

import threading
import time

_MAX_PROGRESS_LINES = 60


class SweepState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "starting"
        self.message = "Starting up..."
        self.progress_lines: list[str] = []
        self.total_steps = 0
        self.steps_done = 0
        self.error: str | None = None
        self.updated_at = time.time()

    def set_status(self, status: str, message: str) -> None:
        with self._lock:
            self.status = status
            self.message = message
            self.updated_at = time.time()

    def set_total_steps(self, n: int) -> None:
        with self._lock:
            self.total_steps = n
            self.updated_at = time.time()

    def advance(self, message: str | None = None) -> None:
        with self._lock:
            self.steps_done += 1
            if message:
                self.message = message
            self.updated_at = time.time()

    def add_progress_line(self, line: str) -> None:
        with self._lock:
            self.progress_lines.append(line)
            self.progress_lines = self.progress_lines[-_MAX_PROGRESS_LINES:]
            self.updated_at = time.time()

    def set_error(self, message: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = message
            self.updated_at = time.time()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "message": self.message,
                "progress_lines": list(self.progress_lines),
                "total_steps": self.total_steps,
                "steps_done": self.steps_done,
                "error": self.error,
                "updated_at": self.updated_at,
            }
