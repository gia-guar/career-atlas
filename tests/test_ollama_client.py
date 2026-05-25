"""OllamaClient HTTP behavior — mocked via respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from career_atlas.llm.client import OllamaClient, OllamaError

OLLAMA_URL = "http://localhost:11434/api/chat"


def _chat_payload(content: dict) -> dict:
    return {
        "model": "gemma4:e2b",
        "message": {"role": "assistant", "content": json.dumps(content)},
        "done": True,
    }


@respx.mock
def test_chat_json_success_returns_parsed_dict():
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(200, json=_chat_payload({"queries": ["ml engineer"]}))
    )
    with OllamaClient() as client:
        out = client.chat_json(
            model="gemma4:e2b",
            system="sys",
            user="usr",
            json_schema={"type": "object"},
        )
    assert out == {"queries": ["ml engineer"]}
    assert route.call_count == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "gemma4:e2b"
    assert sent["stream"] is False
    assert sent["format"] == {"type": "object"}
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1] == {"role": "user", "content": "usr"}


@respx.mock
def test_chat_json_retries_on_5xx_then_succeeds():
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_chat_payload({"ok": True})),
        ]
    )
    with OllamaClient() as client:
        out = client.chat_json(
            model="gemma4:e2b", system="s", user="u", json_schema={}
        )
    assert out == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_chat_json_retries_on_transport_error():
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.ConnectError("nope"),
            httpx.Response(200, json=_chat_payload({"ok": 1})),
        ]
    )
    with OllamaClient() as client:
        out = client.chat_json(
            model="gemma4:e2b", system="s", user="u", json_schema={}
        )
    assert out == {"ok": 1}
    assert route.call_count == 2


@respx.mock
def test_chat_json_retries_on_malformed_json_then_gives_up():
    bad = {
        "model": "gemma4:e2b",
        "message": {"role": "assistant", "content": "not json {{{"},
        "done": True,
    }
    route = respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=bad))
    with OllamaClient() as client, pytest.raises(json.JSONDecodeError):
        client.chat_json(
            model="gemma4:e2b", system="s", user="u", json_schema={}
        )
    assert route.call_count == 3  # stop_after_attempt(3)


@respx.mock
def test_chat_json_does_not_retry_on_4xx():
    route = respx.post(OLLAMA_URL).mock(return_value=httpx.Response(404))
    with OllamaClient() as client, pytest.raises(httpx.HTTPStatusError):
        client.chat_json(
            model="gemma4:nope", system="s", user="u", json_schema={}
        )
    assert route.call_count == 1


@respx.mock
def test_chat_json_raises_on_empty_assistant_content():
    bad = {
        "model": "gemma4:e2b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=bad))
    with OllamaClient() as client, pytest.raises(OllamaError):
        client.chat_json(
            model="gemma4:e2b", system="s", user="u", json_schema={}
        )


@respx.mock
def test_chat_json_forwards_options():
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(200, json=_chat_payload({"x": 1}))
    )
    opts = {"temperature": 0.1, "num_predict": 4096, "num_ctx": 16384}
    with OllamaClient() as client:
        client.chat_json(
            model="gemma4:e2b",
            system="s",
            user="u",
            json_schema={},
            options=opts,
        )
    sent = json.loads(route.calls[0].request.content)
    assert sent["options"] == opts
