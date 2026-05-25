"""cv_extraction node behavior with a stub OllamaClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from career_atlas.pipelines.cv_extraction import nodes as cv_nodes
from career_atlas.schemas import CVProfile

FIXTURES = Path(__file__).parent / "fixtures"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


class StubOllama:
    """Returns a queued response per call; records each request for assertions."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("StubOllama ran out of queued responses")
        return self._responses.pop(0)


@pytest.fixture
def cv_text() -> str:
    return (FIXTURES / "cv" / "sample_cv.md").read_text(encoding="utf-8")


@pytest.fixture
def extraction_response() -> dict:
    return _load_json(FIXTURES / "llm" / "skill_extraction_response.json")


@pytest.fixture
def targeting_response() -> dict:
    return _load_json(FIXTURES / "llm" / "query_derivation_response.json")


@pytest.fixture
def cv_params() -> dict:
    return {
        "hardware_tier": "mid",
        "generation": {"temperature": 0.1, "num_predict": 4096},
        "model_registry": {
            "low": {"ollama_tag": "gemma4:e2b"},
            "mid": {"ollama_tag": "gemma4:e4b"},
            "high": {"ollama_tag": "gemma4:26b"},
            "max": {"ollama_tag": "gemma4:31b"},
        },
    }


@pytest.fixture
def base_scraping_params() -> dict:
    """Tech knobs only — the LLM provides queries/countries/locations."""
    return {
        "adzuna": {
            "results_per_page": 50,
            "max_pages": 5,
            "max_days_old": 30,
            "requests_per_minute": 20,
        },
        "jobspy": {
            "sites": ["linkedin", "indeed"],
            "results_wanted": 100,
            "hours_old": 720,
        },
    }


class TestExtractCvProfile:
    def test_returns_validated_profile_dict(
        self, cv_text, cv_params, extraction_response
    ):
        stub = StubOllama([extraction_response])
        out = cv_nodes.extract_cv_profile(cv_text, cv_params, stub)
        CVProfile.model_validate(out)
        assert out["role_titles"]
        assert any(s["name"] == "PyTorch" for s in out["skills"])

    def test_picks_model_tag_from_tier(
        self, cv_text, cv_params, extraction_response
    ):
        cv_params["hardware_tier"] = "high"
        stub = StubOllama([extraction_response])
        cv_nodes.extract_cv_profile(cv_text, cv_params, stub)
        assert stub.calls[0]["model"] == "gemma4:26b"

    def test_forwards_generation_options(
        self, cv_text, cv_params, extraction_response
    ):
        stub = StubOllama([extraction_response])
        cv_nodes.extract_cv_profile(cv_text, cv_params, stub)
        assert stub.calls[0]["options"] == cv_params["generation"]

    def test_uses_cvprofile_schema(
        self, cv_text, cv_params, extraction_response
    ):
        stub = StubOllama([extraction_response])
        cv_nodes.extract_cv_profile(cv_text, cv_params, stub)
        schema = stub.calls[0]["json_schema"]
        assert "properties" in schema
        assert "skills" in schema["properties"]
        assert "role_titles" in schema["properties"]

    def test_empty_cv_text_raises(self, cv_params, extraction_response):
        stub = StubOllama([extraction_response])
        with pytest.raises(ValueError, match="cv_raw_text is empty"):
            cv_nodes.extract_cv_profile("   ", cv_params, stub)

    def test_unknown_tier_raises(self, cv_text, cv_params, extraction_response):
        cv_params["hardware_tier"] = "ultra"
        stub = StubOllama([extraction_response])
        with pytest.raises(ValueError, match="model_registry"):
            cv_nodes.extract_cv_profile(cv_text, cv_params, stub)


class TestDeriveTargetedScrapingParams:
    def test_preserves_tech_knobs(
        self, extraction_response, cv_params, base_scraping_params, targeting_response
    ):
        stub = StubOllama([targeting_response])
        out = cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        # Tech knobs from base must survive unchanged.
        assert out["adzuna"]["requests_per_minute"] == 20
        assert out["adzuna"]["results_per_page"] == 50
        assert out["adzuna"]["max_pages"] == 5
        assert out["jobspy"]["sites"] == ["linkedin", "indeed"]
        assert out["jobspy"]["results_wanted"] == 100

    def test_llm_fills_countries_and_locations(
        self, extraction_response, cv_params, base_scraping_params, targeting_response
    ):
        stub = StubOllama([targeting_response])
        out = cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        assert out["adzuna"]["countries"] == ["de", "gb", "us"]
        assert out["jobspy"]["locations"] == [
            {"name": "Berlin", "country_indeed": "germany"},
            {"name": "Munich", "country_indeed": "germany"},
            {"name": "United Kingdom", "country_indeed": "uk"},
        ]

    def test_queries_combine_roles_and_extras(
        self, extraction_response, cv_params, base_scraping_params, targeting_response
    ):
        stub = StubOllama([targeting_response])
        out = cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        queries = out["adzuna"]["queries"]
        # role_titles come first verbatim
        assert queries[: len(extraction_response["role_titles"])] == extraction_response["role_titles"]
        # LLM extras follow
        assert "mlops engineer aws kubernetes" in queries
        assert out["jobspy"]["queries"] == queries

    def test_invalid_adzuna_country_rejected(
        self, extraction_response, cv_params, base_scraping_params
    ):
        bad = {
            "queries": ["ml engineer"],
            "adzuna_countries": ["xx"],  # not in the enum
            "jobspy_locations": [{"name": "Berlin", "country_indeed": "germany"}],
        }
        stub = StubOllama([bad])
        with pytest.raises(Exception):  # noqa: B017 - any ValidationError flavor
            cv_nodes.derive_targeted_scraping_params(
                extraction_response, cv_params, base_scraping_params, stub
            )

    def test_invalid_jobspy_country_rejected(
        self, extraction_response, cv_params, base_scraping_params
    ):
        bad = {
            "queries": ["ml engineer"],
            "adzuna_countries": ["de"],
            "jobspy_locations": [{"name": "Berlin", "country_indeed": "narnia"}],
        }
        stub = StubOllama([bad])
        with pytest.raises(Exception):  # noqa: B017
            cv_nodes.derive_targeted_scraping_params(
                extraction_response, cv_params, base_scraping_params, stub
            )

    def test_dedup_is_case_insensitive(
        self, extraction_response, cv_params, base_scraping_params
    ):
        dup = {
            "queries": [
                "senior machine learning engineer",  # case-dup of role_titles[0]
                "MLOps Engineer Cloud",
            ],
            "adzuna_countries": ["de"],
            "jobspy_locations": [{"name": "Berlin", "country_indeed": "germany"}],
        }
        stub = StubOllama([dup])
        out = cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        lowered = [q.lower() for q in out["adzuna"]["queries"]]
        assert lowered.count("senior machine learning engineer") == 1

    def test_does_not_mutate_base_params(
        self, extraction_response, cv_params, base_scraping_params, targeting_response
    ):
        snapshot = json.dumps(base_scraping_params, sort_keys=True)
        stub = StubOllama([targeting_response])
        cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        assert json.dumps(base_scraping_params, sort_keys=True) == snapshot

    def test_targeting_uses_jobsearchtargeting_schema(
        self, extraction_response, cv_params, base_scraping_params, targeting_response
    ):
        stub = StubOllama([targeting_response])
        cv_nodes.derive_targeted_scraping_params(
            extraction_response, cv_params, base_scraping_params, stub
        )
        schema = stub.calls[0]["json_schema"]
        assert "queries" in schema["properties"]
        assert "adzuna_countries" in schema["properties"]
        assert "jobspy_locations" in schema["properties"]
