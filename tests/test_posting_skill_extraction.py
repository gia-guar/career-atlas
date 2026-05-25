"""Stage 3 extract_posting_skills node behavior with a stub OllamaClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from career_atlas.pipelines.skill_map import nodes as sg_nodes
from career_atlas.schemas import POSTING_SKILLS_COLUMNS

FIXTURES = Path(__file__).parent / "fixtures"


class StubOllama:
    def __init__(self, responses: list[dict] | None = None):
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self._exception_factory = None

    def queue(self, response: dict):
        self._responses.append(response)

    def raise_next(self, exc: Exception):
        self._exception_factory = exc

    def chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self._exception_factory is not None:
            exc = self._exception_factory
            self._exception_factory = None
            raise exc
        if not self._responses:
            raise AssertionError("StubOllama ran out of queued responses")
        return self._responses.pop(0)


@pytest.fixture
def cv_params() -> dict:
    return {
        "hardware_tier": "mid",
        "model_registry": {
            "low": {"ollama_tag": "gemma4:e2b"},
            "mid": {"ollama_tag": "gemma4:e4b"},
            "high": {"ollama_tag": "gemma4:26b"},
        },
    }


@pytest.fixture
def skill_map_params() -> dict:
    return {"extraction": {"generation": {"temperature": 0.1, "num_predict": 1024}}}


@pytest.fixture
def posting_response() -> dict:
    return json.loads(
        (FIXTURES / "llm" / "posting_skills_response.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def postings_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "p1", "description": "Old posting with a long description."},
            {"id": "p2", "description": "New posting with skills to extract."},
            {"id": "p3", "description": ""},  # no description → skipped
        ]
    )


@pytest.fixture
def cached_skills() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"posting_id": "p1", "name": "ExistingTool", "kind": "tool"},
        ],
        columns=list(POSTING_SKILLS_COLUMNS),
    )


class TestExtractPostingSkills:
    def test_only_new_postings_call_the_llm(
        self,
        postings_df,
        cached_skills,
        cv_params,
        skill_map_params,
        posting_response,
    ):
        stub = StubOllama([posting_response])
        out = sg_nodes.extract_posting_skills(
            postings_df, cached_skills, cv_params, skill_map_params, stub
        )
        # Only p2 is new and has a description, so exactly one LLM call.
        assert len(stub.calls) == 1
        # Cached row preserved, new rows appended.
        assert set(out["posting_id"].unique()) == {"p1", "p2"}
        # The new posting yields one row per skill in the fixture.
        n_new = len(posting_response["skills"])
        assert (out["posting_id"] == "p2").sum() == n_new

    def test_columns_match_schema(
        self, postings_df, cached_skills, cv_params, skill_map_params, posting_response
    ):
        stub = StubOllama([posting_response])
        out = sg_nodes.extract_posting_skills(
            postings_df, cached_skills, cv_params, skill_map_params, stub
        )
        assert list(out.columns) == list(POSTING_SKILLS_COLUMNS)

    def test_model_tag_picked_from_tier(
        self, postings_df, cached_skills, cv_params, skill_map_params, posting_response
    ):
        cv_params["hardware_tier"] = "high"
        stub = StubOllama([posting_response])
        sg_nodes.extract_posting_skills(
            postings_df, cached_skills, cv_params, skill_map_params, stub
        )
        assert stub.calls[0]["model"] == "gemma4:26b"

    def test_forwards_generation_options(
        self, postings_df, cached_skills, cv_params, skill_map_params, posting_response
    ):
        stub = StubOllama([posting_response])
        sg_nodes.extract_posting_skills(
            postings_df, cached_skills, cv_params, skill_map_params, stub
        )
        assert (
            stub.calls[0]["options"]
            == skill_map_params["extraction"]["generation"]
        )

    def test_uses_postingskills_schema(
        self, postings_df, cached_skills, cv_params, skill_map_params, posting_response
    ):
        stub = StubOllama([posting_response])
        sg_nodes.extract_posting_skills(
            postings_df, cached_skills, cv_params, skill_map_params, stub
        )
        schema = stub.calls[0]["json_schema"]
        assert schema["properties"].get("skills") is not None

    def test_per_posting_failure_is_isolated(
        self, postings_df, cached_skills, cv_params, skill_map_params
    ):
        # Two new postings, the first one errors. The second should still run.
        postings = pd.DataFrame(
            [
                {"id": "px", "description": "first new posting"},
                {"id": "py", "description": "second new posting"},
            ]
        )
        stub = StubOllama()
        stub.raise_next(RuntimeError("ollama transient blow-up"))
        stub.queue({"skills": [{"name": "Recovered", "kind": "tool"}]})
        out = sg_nodes.extract_posting_skills(
            postings,
            pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS)),
            cv_params,
            skill_map_params,
            stub,
        )
        # px failed, py succeeded.
        assert set(out["posting_id"].unique()) == {"py"}
        assert out.iloc[0]["name"] == "Recovered"

    def test_no_postings_returns_cache_unchanged(
        self, cached_skills, cv_params, skill_map_params
    ):
        stub = StubOllama()
        out = sg_nodes.extract_posting_skills(
            pd.DataFrame(columns=["id", "description"]),
            cached_skills,
            cv_params,
            skill_map_params,
            stub,
        )
        assert stub.calls == []
        pd.testing.assert_frame_equal(out.reset_index(drop=True), cached_skills.reset_index(drop=True))

    def test_bare_list_response_is_coerced(
        self, cv_params, skill_map_params
    ):
        # Gemma sometimes flattens {"skills": [...]} → just [...]. The node
        # must wrap before validating so we don't lose the row.
        postings = pd.DataFrame([{"id": "pflat", "description": "desc"}])
        bare_list = [
            {"name": "Python", "kind": "tool"},
            {"name": "MLOps", "kind": "skill"},
        ]
        stub = StubOllama([bare_list])  # type: ignore[arg-type] — passes through chat_json
        out = sg_nodes.extract_posting_skills(
            postings,
            pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS)),
            cv_params,
            skill_map_params,
            stub,
        )
        assert (out["posting_id"] == "pflat").sum() == 2
        assert set(out["name"]) == {"Python", "MLOps"}

    def test_unknown_kind_is_coerced_to_skill(
        self, cv_params, skill_map_params
    ):
        # Gemma occasionally emits a kind outside the Literal set (German
        # equivalents, "language", "certification", etc). The node coerces
        # the row instead of dropping the whole posting.
        postings = pd.DataFrame([{"id": "pkind", "description": "desc"}])
        response = {
            "skills": [
                {"name": "Python", "kind": "tool"},
                {"name": "Deutsch", "kind": "sprache"},  # not in Literal
                {"name": "PhD", "kind": "requirement"},
            ]
        }
        stub = StubOllama([response])
        out = sg_nodes.extract_posting_skills(
            postings,
            pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS)),
            cv_params,
            skill_map_params,
            stub,
        )
        assert (out["posting_id"] == "pkind").sum() == 3
        deutsch_row = out[out["name"] == "Deutsch"].iloc[0]
        assert deutsch_row["kind"] == "skill"
        # Valid kinds pass through unchanged.
        assert out[out["name"] == "Python"].iloc[0]["kind"] == "tool"
        assert out[out["name"] == "PhD"].iloc[0]["kind"] == "requirement"

    def test_postings_without_description_are_skipped(
        self, cv_params, skill_map_params
    ):
        postings = pd.DataFrame(
            [
                {"id": "empty1", "description": None},
                {"id": "empty2", "description": "   "},
            ]
        )
        stub = StubOllama()
        out = sg_nodes.extract_posting_skills(
            postings,
            pd.DataFrame(columns=list(POSTING_SKILLS_COLUMNS)),
            cv_params,
            skill_map_params,
            stub,
        )
        assert stub.calls == []
        assert out.empty
        assert list(out.columns) == list(POSTING_SKILLS_COLUMNS)
