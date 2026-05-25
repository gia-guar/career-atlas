"""Canonical job-posting schema and identity helpers.

`JobPosting` is the unified shape every source normalises to. Identity is the
sha256 prefix of `title_normalized | company_normalized | location_lower`, which
lets us deduplicate across Adzuna and JobSpy without trusting either source's id.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


Source = Literal["adzuna", "linkedin", "indeed", "glassdoor"]

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(value))
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", no_punct).strip()


def _content_hash(
    title_normalized: str,
    company_normalized: str | None,
    location: str | None,
) -> str:
    company = company_normalized or ""
    loc = (location or "").lower().strip()
    payload = f"{title_normalized}|{company}|{loc}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: Source
    source_id: str | None = None
    url: str | None = None
    title: str
    title_normalized: str
    company: str | None = None
    company_normalized: str | None = None
    location: str | None = None
    country: str | None = None
    is_remote: bool = False
    description: str | None = None
    posted_at: datetime | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    job_type: str | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    raw: dict[str, Any]

    @field_validator("url", mode="before")
    @classmethod
    def _soft_url(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("title_normalized")
    @classmethod
    def _require_title_norm(cls, v: str) -> str:
        if not v:
            raise ValueError("title_normalized must be non-empty")
        return v


JOB_POSTING_COLUMNS: tuple[str, ...] = tuple(JobPosting.model_fields.keys())


# ---------------------------------------------------------------------------
# Stage 2: CV profile (skills / tools extracted from a user CV by the LLM)
# ---------------------------------------------------------------------------

SkillKind = Literal["skill", "tool"]
Proficiency = Literal["mentioned", "used", "expert"]
Seniority = Literal["junior", "mid", "senior", "staff", "principal"]


class CVSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: SkillKind
    proficiency: Proficiency | None = None
    evidence: str | None = None


class CVProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[CVSkill]
    role_titles: list[str]
    seniority: Seniority | None = None
    years_experience: float | None = None
    locations_preferred: list[str] = []
    summary: str | None = None

    @field_validator("role_titles")
    @classmethod
    def _require_at_least_one_role(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("role_titles must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Job-search targeting (LLM-derived from CVProfile, drives the scrape)
# ---------------------------------------------------------------------------
#
# Both enums are constrained to what the providers actually accept — this
# means the LLM cannot hallucinate an unsupported country, because Ollama's
# JSON-schema `format` rejects values outside the Literal set.

AdzunaCountry = Literal[
    "gb", "us", "at", "au", "be", "br", "ca", "ch", "de", "es",
    "fr", "in", "it", "mx", "nl", "nz", "pl", "sg", "za",
]

JobspyCountry = Literal[
    "uk", "usa", "austria", "australia", "belgium", "brazil", "canada",
    "switzerland", "germany", "spain", "france", "india", "italy",
    "mexico", "netherlands", "new zealand", "poland", "singapore",
    "south africa",
]


class JobspyLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    country_indeed: JobspyCountry


class JobSearchTargeting(BaseModel):
    """LLM-derived search plan: where and what to look for.

    Built from a `CVProfile` by `derive_targeted_scraping_params`. Combined
    with the technical knobs in `params:scraping` (rate limits, sites,
    max_pages, etc.) to produce the final scraping-params dict consumed by
    the existing fetch/normalize/dedupe nodes.
    """

    model_config = ConfigDict(extra="forbid")

    queries: list[str]
    adzuna_countries: list[AdzunaCountry]
    jobspy_locations: list[JobspyLocation]

    @field_validator("queries")
    @classmethod
    def _require_queries(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("queries must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Stage 3: per-posting skill extraction (LLM reads each posting description)
# ---------------------------------------------------------------------------

SkillRequirementKind = Literal["skill", "tool", "requirement"]

POSTING_SKILLS_COLUMNS: tuple[str, ...] = ("posting_id", "name", "kind")


class PostingSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: SkillRequirementKind


class PostingSkills(BaseModel):
    """Container for the LLM response when extracting skills from a posting."""

    model_config = ConfigDict(extra="forbid")

    skills: list[PostingSkill]
