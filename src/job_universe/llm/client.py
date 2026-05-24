"""Ollama HTTP client with JSON-schema-constrained output and retry.

Ollama exposes a `format` field on `/api/chat` that — when set to a JSON
schema — constrains the model's output to validate against that schema. We
pass the schema derived from a Pydantic model (`SomeModel.model_json_schema()`)
so the LLM contract has a single source of truth.

Mirrors the injectable-client pattern of `AdzunaClient`: pass a real
`httpx.Client` in production, a `respx`-mocked one in tests.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 180.0


class OllamaError(Exception):
    """Raised for non-retryable Ollama failures (bad model, malformed response, ...)."""


class OllamaServerError(Exception):
    """Raised on 5xx — retryable; distinct from 4xx so 'model not found' fails fast."""


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.Client | None = None,
    ):
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self._client = http_client or httpx.Client(timeout=timeout_s)
        self._owns_client = http_client is None

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.TransportError, OllamaServerError, json.JSONDecodeError)
        ),
        reraise=True,
    )
    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /api/chat with format=<json_schema>, return parsed JSON content.

        Retries on transport errors, 5xx, and JSON decode failures (a glitched
        token stream can produce malformed JSON). Validation errors raised by
        callers (Pydantic) are NOT retried — those indicate prompt drift.
        """
        url = f"{self.host}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": json_schema,
            "stream": False,
            "options": options or {},
        }
        resp = self._client.post(url, json=payload)
        if 500 <= resp.status_code < 600:
            raise OllamaServerError(f"ollama server error {resp.status_code}")
        resp.raise_for_status()
        body = resp.json()
        content = (body.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            # Include done_reason + eval_count so the caller can tell
            # "num_predict exhausted" ('length') from a genuine empty
            # generation ('stop') without inspecting the server logs.
            raise OllamaError(
                f"ollama response missing message.content; "
                f"done_reason={body.get('done_reason')!r} "
                f"eval_count={body.get('eval_count')!r} "
                f"keys={list(body)}"
            )
        return json.loads(content)
