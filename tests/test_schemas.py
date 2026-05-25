"""Schema and identity-hash invariants."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from career_atlas.schemas import (
    JOB_POSTING_COLUMNS,
    CVProfile,
    CVSkill,
    JobPosting,
    JobSearchTargeting,
    JobspyLocation,
    _content_hash,
    _normalize_text,
)


def _valid_payload(**overrides):
    base = {
        "id": "abcdef0123456789",
        "source": "adzuna",
        "source_id": "12345",
        "url": "https://example.com/job/12345",
        "title": "Machine Learning Engineer",
        "title_normalized": "machine learning engineer",
        "company": "Acme",
        "company_normalized": "acme",
        "location": "Berlin",
        "country": "de",
        "is_remote": False,
        "description": "Some role.",
        "posted_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "salary_min": 80000.0,
        "salary_max": 120000.0,
        "salary_currency": "EUR",
        "job_type": "permanent",
        "first_seen_at": datetime(2026, 5, 20, tzinfo=timezone.utc),
        "last_seen_at": datetime(2026, 5, 20, tzinfo=timezone.utc),
        "raw": {"adzuna": {"id": "12345"}},
    }
    base.update(overrides)
    return base


class TestJobPosting:
    def test_valid_roundtrip(self):
        post = JobPosting(**_valid_payload())
        assert post.id == "abcdef0123456789"
        assert post.title_normalized == "machine learning engineer"

    def test_url_accepts_malformed_strings(self):
        post = JobPosting(**_valid_payload(url="not a url at all"))
        assert post.url == "not a url at all"

    def test_url_empty_string_becomes_none(self):
        post = JobPosting(**_valid_payload(url="   "))
        assert post.url is None

    def test_url_none_passes(self):
        post = JobPosting(**_valid_payload(url=None))
        assert post.url is None

    def test_empty_title_normalized_rejected(self):
        with pytest.raises(ValidationError):
            JobPosting(**_valid_payload(title_normalized=""))

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            JobPosting(**_valid_payload(extra="nope"))

    def test_raw_is_dict_keyed_by_source(self):
        post = JobPosting(
            **_valid_payload(raw={"adzuna": {"a": 1}, "linkedin": {"b": 2}})
        )
        assert isinstance(post.raw, dict)
        assert set(post.raw.keys()) == {"adzuna", "linkedin"}


class TestNormalizeText:
    def test_basic_lowercase_strip(self):
        assert _normalize_text("  Machine Learning  ") == "machine learning"

    def test_punctuation_removed(self):
        assert _normalize_text("Sr. ML Engineer (Remote)") == "sr ml engineer remote"

    def test_unicode_normalised(self):
        assert _normalize_text("Café Engineer") == "cafe engineer"

    def test_none_returns_empty(self):
        assert _normalize_text(None) == ""

    def test_whitespace_collapsed(self):
        assert _normalize_text("ML\t\nEngineer\n") == "ml engineer"


class TestContentHash:
    def test_stable_across_calls(self):
        h1 = _content_hash("ml engineer", "acme", "berlin")
        h2 = _content_hash("ml engineer", "acme", "berlin")
        assert h1 == h2
        assert len(h1) == 16

    def test_case_insensitive_via_location(self):
        h_lower = _content_hash("ml engineer", "acme", "berlin")
        h_upper = _content_hash("ml engineer", "acme", "BERLIN")
        assert h_lower == h_upper

    def test_distinct_locations_differ(self):
        h_berlin = _content_hash("ml engineer", "acme", "berlin")
        h_munich = _content_hash("ml engineer", "acme", "munich")
        assert h_berlin != h_munich

    def test_company_change_changes_hash(self):
        h1 = _content_hash("ml engineer", "acme", "berlin")
        h2 = _content_hash("ml engineer", "globex", "berlin")
        assert h1 != h2

    def test_normalized_inputs_are_what_matter(self):
        # Caller normalises before hashing; the helper itself doesn't re-normalise
        # but `_normalize_text` makes raw "Sr. ML Engineer" and "sr ml engineer"
        # collapse to the same string, so identity holds end-to-end.
        a = _normalize_text("Sr. ML Engineer")
        b = _normalize_text("sr ml engineer")
        assert _content_hash(a, "acme", "berlin") == _content_hash(b, "acme", "berlin")


def test_columns_list_matches_model():
    assert set(JOB_POSTING_COLUMNS) == set(JobPosting.model_fields.keys())


class TestCVProfile:
    def _valid(self, **overrides):
        base = {
            "skills": [
                {"name": "PyTorch", "kind": "tool", "proficiency": "expert"},
                {"name": "MLOps", "kind": "skill", "proficiency": "used"},
            ],
            "role_titles": ["ML Engineer"],
            "seniority": "senior",
            "years_experience": 6.0,
            "locations_preferred": ["Berlin"],
            "summary": "ML engineer.",
        }
        base.update(overrides)
        return base

    def test_valid_roundtrip(self):
        p = CVProfile(**self._valid())
        assert p.role_titles == ["ML Engineer"]
        assert p.skills[0].kind == "tool"

    def test_empty_role_titles_rejected(self):
        with pytest.raises(ValidationError):
            CVProfile(**self._valid(role_titles=[]))

    def test_skills_kind_constrained(self):
        with pytest.raises(ValidationError):
            CVSkill(name="x", kind="invalid")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CVProfile(**self._valid(extra="nope"))

    def test_seniority_optional(self):
        p = CVProfile(**self._valid(seniority=None))
        assert p.seniority is None

    def test_locations_default_empty(self):
        payload = self._valid()
        payload.pop("locations_preferred")
        p = CVProfile(**payload)
        assert p.locations_preferred == []

    def test_json_schema_is_well_formed(self):
        schema = CVProfile.model_json_schema()
        assert schema["type"] == "object"
        assert "skills" in schema["properties"]
        assert "role_titles" in schema["properties"]


class TestJobSearchTargeting:
    def _valid(self, **overrides):
        base = {
            "queries": ["ml engineer", "data scientist"],
            "adzuna_countries": ["de", "gb"],
            "jobspy_locations": [
                {"name": "Berlin", "country_indeed": "germany"},
            ],
        }
        base.update(overrides)
        return base

    def test_valid_roundtrip(self):
        t = JobSearchTargeting(**self._valid())
        assert t.adzuna_countries == ["de", "gb"]
        assert t.jobspy_locations[0].country_indeed == "germany"

    def test_empty_queries_rejected(self):
        with pytest.raises(ValidationError):
            JobSearchTargeting(**self._valid(queries=[]))

    def test_unsupported_adzuna_country_rejected(self):
        with pytest.raises(ValidationError):
            JobSearchTargeting(**self._valid(adzuna_countries=["zz"]))

    def test_unsupported_jobspy_country_rejected(self):
        with pytest.raises(ValidationError):
            JobspyLocation(name="Atlantis", country_indeed="atlantis")  # type: ignore[arg-type]

    def test_json_schema_contains_country_enums(self):
        schema = JobSearchTargeting.model_json_schema()
        text = str(schema)
        # ISO-style Adzuna codes appear as Literal enum values in the schema
        assert "de" in text and "gb" in text and "us" in text
        # JobSpy country_indeed values too
        assert "germany" in text and "uk" in text and "usa" in text
