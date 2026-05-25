"""Adzuna search API client with token-bucket throttling and retry."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs"
# Adzuna's documented hard ceiling is 25 req/60s; default to a safety margin.
DEFAULT_RPM = 20


class AdzunaRateLimitError(Exception):
    """Raised when Adzuna returns 429."""


class AdzunaClient:
    """Thin wrapper around Adzuna's job-search endpoint.

    `time_func` and `sleep_func` are indirected so the rate-limit test can drive
    a fake clock without monkey-patching the `time` module globally.
    """

    def __init__(
        self,
        app_id: str,
        app_key: str,
        requests_per_minute: int = DEFAULT_RPM,
        results_per_page: int = 50,
        max_days_old: int = 30,
        client: httpx.Client | None = None,
        time_func=time.monotonic,
        sleep_func=time.sleep,
    ):
        self.app_id = app_id
        self.app_key = app_key
        self.requests_per_minute = requests_per_minute
        self.results_per_page = results_per_page
        self.max_days_old = max_days_old
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._time = time_func
        self._sleep = sleep_func
        self._request_times: deque[float] = deque()

    def __enter__(self) -> "AdzunaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _throttle(self) -> None:
        now = self._time()
        window_start = now - 60.0
        while self._request_times and self._request_times[0] <= window_start:
            self._request_times.popleft()
        if len(self._request_times) >= self.requests_per_minute:
            wait = 60.0 - (now - self._request_times[0]) + 0.05
            if wait > 0:
                logger.debug("adzuna throttle sleeping %.2fs", wait)
                self._sleep(wait)
                now = self._time()
                window_start = now - 60.0
                while self._request_times and self._request_times[0] <= window_start:
                    self._request_times.popleft()
        self._request_times.append(self._time())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.TransportError, AdzunaRateLimitError, httpx.HTTPStatusError)
        ),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        resp = self._client.get(url, params=params)
        if resp.status_code == 429:
            raise AdzunaRateLimitError("Adzuna returned 429")
        if 500 <= resp.status_code < 600:
            raise httpx.HTTPStatusError(
                f"server error {resp.status_code}", request=resp.request, response=resp
            )
        resp.raise_for_status()
        return resp.json()

    def search(self, country: str, query: str, page: int = 1) -> dict[str, Any]:
        url = f"{BASE_URL}/{country}/search/{page}"
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query,
            "results_per_page": self.results_per_page,
            "max_days_old": self.max_days_old,
            "content-type": "application/json",
        }
        return self._get(url, params)

    def search_all(
        self, country: str, query: str, max_pages: int = 5
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            try:
                payload = self.search(country, query, page)
            except httpx.HTTPStatusError as exc:
                # 4xx other than 429 means the query is bad — stop early instead of retry.
                logger.warning(
                    "adzuna search aborted for %s/%r page %d: %s",
                    country,
                    query,
                    page,
                    exc,
                )
                break
            page_results = payload.get("results") or []
            if not page_results:
                break
            results.extend(page_results)
            if len(page_results) < self.results_per_page:
                break
        return results
