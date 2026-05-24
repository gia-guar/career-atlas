"""Thread-safe progress emitter shared between pipeline nodes and the SSE layer.

The pipelines run synchronously on a worker thread. The FastAPI app owns an
``asyncio.Queue`` consumed by the SSE response. ``ProgressEmitter.emit`` is
called from the worker thread; it hops the value onto the event loop via
``loop.call_soon_threadsafe``. When unbound (CLI use, or before a UI run
starts), ``emit`` is a no-op so existing tests/CLI runs are unaffected.
"""

from __future__ import annotations

import asyncio
from typing import Any


class ProgressEmitter:
    def __init__(self) -> None:
        self._queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._counters: dict[str, int] = {}

    def bind(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def unbind(self) -> None:
        self._queue = None
        self._loop = None

    def reset(self) -> None:
        self._counters.clear()

    @property
    def bound(self) -> bool:
        return self._queue is not None and self._loop is not None

    def emit(self, event_type: str, value: Any) -> None:
        """Emit an absolute event value (used for non-counter events like
        'done', 'error', or counters set directly)."""
        if isinstance(value, int):
            self._counters[event_type] = value
        self._dispatch({"type": event_type, "value": value})

    def incr(self, event_type: str, by: int = 1) -> int:
        """Increment a named counter and emit the new total. Returns the new
        total. Multiple producers (Adzuna + JobSpy + skill extraction) can
        share a counter without coordinating on absolute values."""
        new_total = self._counters.get(event_type, 0) + int(by)
        self._counters[event_type] = new_total
        self._dispatch({"type": event_type, "value": new_total})
        return new_total

    def _dispatch(self, event: dict[str, Any]) -> None:
        if self._queue is None or self._loop is None:
            return
        if self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            # Loop is closing or already closed mid-emit.
            return


# Module-level slot read by ``ProgressHook``. The web runner sets this to the
# active emitter before invoking ``KedroSession.run`` and clears it in
# ``finally``. CLI runs never touch this and see ``None``.
CURRENT: ProgressEmitter | None = None
