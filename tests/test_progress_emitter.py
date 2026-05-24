"""ProgressEmitter thread-safety + bind/unbind semantics."""

from __future__ import annotations

import asyncio
import threading

import pytest

from job_universe.web.progress import ProgressEmitter


class TestUnbound:
    def test_emit_is_a_noop_when_unbound(self):
        emitter = ProgressEmitter()
        # Must not raise even though nothing is listening.
        emitter.emit("postings_count", 5)
        emitter.incr("skills_count", by=3)
        assert emitter.bound is False

    def test_counters_still_track_when_unbound(self):
        emitter = ProgressEmitter()
        emitter.incr("postings_count", by=10)
        emitter.incr("postings_count", by=5)
        # Counter state is internal but observable via the next emit value.
        assert emitter._counters["postings_count"] == 15


class TestBound:
    def test_emit_delivers_event_to_queue(self):
        async def run():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            emitter = ProgressEmitter()
            emitter.bind(queue, loop)

            # Emit from a worker thread, just like the pipeline does.
            t = threading.Thread(target=emitter.emit, args=("postings_count", 42))
            t.start()
            t.join()

            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            assert event == {"type": "postings_count", "value": 42}

        asyncio.run(run())

    def test_incr_emits_cumulative_total(self):
        async def run():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            emitter = ProgressEmitter()
            emitter.bind(queue, loop)

            def producer():
                emitter.incr("postings_count", by=5)
                emitter.incr("postings_count", by=3)

            t = threading.Thread(target=producer)
            t.start()
            t.join()

            first = await asyncio.wait_for(queue.get(), timeout=1.0)
            second = await asyncio.wait_for(queue.get(), timeout=1.0)
            assert first["value"] == 5
            assert second["value"] == 8  # cumulative

        asyncio.run(run())

    def test_two_event_types_track_independently(self):
        async def run():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            emitter = ProgressEmitter()
            emitter.bind(queue, loop)

            def producer():
                emitter.incr("postings_count", by=10)
                emitter.incr("skills_count", by=2)
                emitter.incr("postings_count", by=5)

            t = threading.Thread(target=producer)
            t.start()
            t.join()

            events = []
            for _ in range(3):
                events.append(await asyncio.wait_for(queue.get(), timeout=1.0))
            postings = [e for e in events if e["type"] == "postings_count"]
            skills = [e for e in events if e["type"] == "skills_count"]
            assert [e["value"] for e in postings] == [10, 15]
            assert [e["value"] for e in skills] == [2]

        asyncio.run(run())

    def test_reset_clears_counters(self):
        emitter = ProgressEmitter()
        emitter.incr("postings_count", by=10)
        emitter.reset()
        assert emitter._counters == {}

    def test_unbind_makes_emit_a_noop_again(self):
        async def run():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            emitter = ProgressEmitter()
            emitter.bind(queue, loop)
            emitter.unbind()
            # Should not raise nor enqueue.
            emitter.emit("postings_count", 100)
            await asyncio.sleep(0)  # let any spurious scheduled callbacks fire
            assert queue.empty()

        asyncio.run(run())


def test_emit_survives_a_closed_loop():
    """If the FastAPI loop has closed while a worker thread is still emitting,
    ``ProgressEmitter.emit`` must absorb the RuntimeError silently."""
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    emitter = ProgressEmitter()
    emitter.bind(queue, loop)
    loop.close()
    # The loop is closed; emit should not crash the worker thread.
    emitter.emit("postings_count", 99)
