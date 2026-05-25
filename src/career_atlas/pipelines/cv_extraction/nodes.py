"""CV extraction nodes: parse CV text into a CVProfile, derive scraping params."""

from __future__ import annotations

import json
import logging
from typing import Any

from career_atlas.llm import OllamaClient, prompts
from career_atlas.schemas import CVProfile, JobSearchTargeting

logger = logging.getLogger(__name__)


def _model_tag(params: dict[str, Any]) -> str:
    tier = params.get("hardware_tier", "mid")
    registry = params.get("model_registry") or {}
    tier_entry = registry.get(tier)
    if not tier_entry or "ollama_tag" not in tier_entry:
        raise ValueError(
            f"cv_extraction.model_registry missing entry for tier {tier!r}"
        )
    return tier_entry["ollama_tag"]


def extract_cv_profile(
    cv_text: str,
    params: dict[str, Any],
    ollama_client: OllamaClient,
) -> dict[str, Any]:
    """Run the LLM extraction prompt and return a CVProfile dump (dict)."""
    if not cv_text or not cv_text.strip():
        raise ValueError("cv_raw_text is empty — drop a CV at data/01_raw/cv/cv.md")

    model = _model_tag(params)
    options = params.get("generation") or {}
    logger.info("extracting CV profile with %s", model)

    raw = ollama_client.chat_json(
        model=model,
        system=prompts.SKILL_EXTRACTION_SYSTEM,
        user=cv_text,
        json_schema=CVProfile.model_json_schema(),
        options=options,
    )
    profile = CVProfile.model_validate(raw)
    logger.info(
        "extracted %d skills, roles=%s, seniority=%s",
        len(profile.skills),
        profile.role_titles,
        profile.seniority,
    )
    return profile.model_dump(mode="json")


def derive_targeted_scraping_params(
    cv_profile: dict[str, Any],
    cv_params: dict[str, Any],
    base_scraping_params: dict[str, Any],
    ollama_client: OllamaClient,
) -> dict[str, Any]:
    """Build the scraping-params dict from the CVProfile via one LLM call.

    The LLM produces a `JobSearchTargeting` — queries, Adzuna country codes,
    and JobSpy locations — constrained by the JSON schema so it cannot
    hallucinate invalid country codes. We combine that with the technical
    knobs from `params:scraping` (rate limits, sites, max_pages, etc.) to
    produce the dict consumed by `fetch_adzuna` / `fetch_jobspy`.

    Nothing in this pipeline is profession-specific — same code path works
    for any CV.
    """
    profile = CVProfile.model_validate(cv_profile)
    model = _model_tag(cv_params)
    options = cv_params.get("generation") or {}

    targeting = ollama_client.chat_json(
        model=model,
        system=prompts.TARGETING_SYSTEM,
        user=json.dumps(cv_profile),
        json_schema=JobSearchTargeting.model_json_schema(),
        options=options,
    )
    target = JobSearchTargeting.model_validate(targeting)

    # role_titles get prepended verbatim so the most obvious searches always
    # run; LLM extras follow. Dedup case-insensitively while preserving order.
    queries: list[str] = []
    seen: set[str] = set()
    for q in [*profile.role_titles, *target.queries]:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            queries.append(q.strip())

    # Optional overrides from conf/local/parameters/cv_extraction.yml. They
    # are applied AFTER the LLM proposes its plan so the user can broaden
    # (e.g. Europe-wide) or replace targeting without re-prompting the model.
    overrides = (cv_params or {}).get("targeting_overrides") or {}
    if "extra_queries" in overrides:
        for q in overrides["extra_queries"] or []:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                queries.append(q.strip())

    out = json.loads(json.dumps(base_scraping_params))  # deep copy via JSON
    out.setdefault("adzuna", {})
    out["adzuna"]["queries"] = queries
    out["adzuna"]["countries"] = (
        list(overrides["adzuna_countries"])
        if "adzuna_countries" in overrides
        else list(target.adzuna_countries)
    )
    out.setdefault("jobspy", {})
    out["jobspy"]["queries"] = queries
    out["jobspy"]["locations"] = (
        list(overrides["jobspy_locations"])
        if "jobspy_locations" in overrides
        else [loc.model_dump() for loc in target.jobspy_locations]
    )

    logger.info(
        "scraping plan: %d queries × %d Adzuna countries × %d JobSpy locations%s",
        len(queries),
        len(out["adzuna"]["countries"]),
        len(out["jobspy"]["locations"]),
        " (overrides applied)" if overrides else "",
    )
    return out
