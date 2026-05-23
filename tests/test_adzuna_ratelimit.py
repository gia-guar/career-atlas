"""Adzuna client must respect the 25 req / 60s ceiling.

We avoid `freezegun` because it does not advance frozen time inside
`time.sleep()` by default — the throttle would deadlock. Instead we inject a
custom `time_func` and `sleep_func` into the client so the rate limiter
operates on a fully controllable virtual clock.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from job_universe.clients.adzuna import AdzunaClient


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        # Track requested sleeps and advance the virtual clock.
        self.sleeps.append(seconds)
        self.t += max(seconds, 0.0)


class StubHTTPClient:
    """Minimal stub matching the httpx.Client.get interface used by AdzunaClient."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        # Record the wall-clock time at which the client believed it was sending.
        self.calls.append(self.clock.time())
        return httpx.Response(
            status_code=200,
            json={"results": []},
            request=httpx.Request("GET", url, params=params),
        )

    def close(self) -> None:
        pass


def _make_client(rpm: int = 20) -> tuple[AdzunaClient, FakeClock, StubHTTPClient]:
    clock = FakeClock()
    stub = StubHTTPClient(clock)
    client = AdzunaClient(
        app_id="x",
        app_key="y",
        requests_per_minute=rpm,
        client=stub,  # type: ignore[arg-type]
        time_func=clock.time,
        sleep_func=clock.sleep,
    )
    return client, clock, stub


def _no_window_exceeds(calls: list[float], limit: int = 25) -> bool:
    for i, t in enumerate(calls):
        window_count = sum(1 for u in calls[: i + 1] if u > t - 60.0)
        if window_count > limit:
            return False
    return True


def test_throttle_respects_25_per_60s():
    client, clock, stub = _make_client(rpm=20)
    for _ in range(30):
        client.search("de", "machine learning engineer", page=1)
    assert len(stub.calls) == 30
    assert _no_window_exceeds(stub.calls, limit=25)


def test_throttle_invokes_sleep_when_bucket_full():
    client, clock, stub = _make_client(rpm=5)
    for _ in range(10):
        client.search("de", "ml", page=1)
    assert any(s > 0 for s in clock.sleeps), "throttle should sleep once the bucket fills"
    assert _no_window_exceeds(stub.calls, limit=25)


def test_throttle_does_not_sleep_under_limit():
    client, clock, stub = _make_client(rpm=20)
    for _ in range(5):
        client.search("de", "ml", page=1)
    # 5 requests well under 20/min should never trigger a sleep
    assert clock.sleeps == []


def test_throttle_first_request_is_immediate():
    client, clock, stub = _make_client(rpm=20)
    client.search("de", "ml", page=1)
    assert clock.sleeps == []
    assert stub.calls == [0.0]
