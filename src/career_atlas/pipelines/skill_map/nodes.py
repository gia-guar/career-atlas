"""Stage 3 nodes — thin wrappers over the helpers in ``skill_map.py`` and
``canonicalize.py``. The Kedro layer here is intentionally dumb: catalog
glue, parameter unpacking, light logging.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from career_atlas import canonicalize, skill_map
from career_atlas.canonicalize import normalize_skill
from career_atlas.llm import OllamaClient, prompts
from career_atlas.schemas import (
    POSTING_SKILLS_COLUMNS,
    CVProfile,
    PostingSkills,
)

logger = logging.getLogger(__name__)

_VALID_POSTING_SKILL_KINDS = {"skill", "tool", "requirement"}


def _model_tag(cv_params: dict[str, Any]) -> str:
    tier = cv_params.get("hardware_tier", "mid")
    registry = cv_params.get("model_registry") or {}
    tier_entry = registry.get(tier)
    if not tier_entry or "ollama_tag" not in tier_entry:
        raise ValueError(
            f"cv_extraction.model_registry missing entry for tier {tier!r}"
        )
    return tier_entry["ollama_tag"]


def _empty_posting_skills() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=object) for col in POSTING_SKILLS_COLUMNS})


def extract_posting_skills(
    postings_cumulative_existing: pd.DataFrame,
    posting_skills_existing: pd.DataFrame,
    cv_params: dict[str, Any],
    skill_map_params: dict[str, Any],
    ollama_client: OllamaClient,
    progress_emitter: Any | None = None,
) -> pd.DataFrame:
    """For each posting not yet in the cache, ask Ollama for its skills.

    Cached posting IDs are passed through unchanged. New IDs without a
    description are skipped (no LLM call). Failures on individual postings
    are logged and that posting yields no rows — it'll be retried on the
    next run.
    """
    if postings_cumulative_existing is None or postings_cumulative_existing.empty:
        logger.info("no postings available; skill cache unchanged")
        if posting_skills_existing is None or posting_skills_existing.empty:
            return _empty_posting_skills()
        return posting_skills_existing.copy()

    cached_ids: set[str] = set()
    if posting_skills_existing is not None and not posting_skills_existing.empty:
        cached_ids = set(posting_skills_existing["posting_id"].astype(str).unique())

    model = _model_tag(cv_params)
    options = (skill_map_params or {}).get("extraction", {}).get("generation") or {}
    json_schema = PostingSkills.model_json_schema()

    new_rows: list[dict[str, Any]] = []
    n_called = 0
    n_skipped = 0
    n_failed = 0

    for _, posting in postings_cumulative_existing.iterrows():
        pid = str(posting["id"])
        if pid in cached_ids:
            continue
        description = posting.get("description")
        if not description or not str(description).strip():
            n_skipped += 1
            continue
        try:
            raw = ollama_client.chat_json(
                model=model,
                system=prompts.POSTING_SKILL_EXTRACTION_SYSTEM,
                user=str(description),
                json_schema=json_schema,
                options=options,
            )
            # Gemma occasionally flattens a single-property object schema and
            # returns the bare inner array (`[{...}, ...]`) instead of the
            # wrapper (`{"skills": [...]}`). Coerce before validating.
            if isinstance(raw, list):
                raw = {"skills": raw}
            # Gemma also drifts on `kind` despite the schema constraint —
            # e.g. emits German equivalents or alternate enums like
            # "language" / "certification". Coerce unknowns to "skill"
            # rather than losing the whole posting.
            skills_field = raw.get("skills") if isinstance(raw, dict) else None
            if isinstance(skills_field, list):
                for s in skills_field:
                    if isinstance(s, dict) and s.get("kind") not in _VALID_POSTING_SKILL_KINDS:
                        s["kind"] = "skill"
            parsed = PostingSkills.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 — per-posting isolation
            logger.warning("skill extraction failed for posting %s: %s", pid, exc)
            n_failed += 1
            continue
        n_called += 1
        for s in parsed.skills:
            new_rows.append(
                {"posting_id": pid, "name": s.name, "kind": s.kind}
            )
        if progress_emitter is not None and parsed.skills:
            progress_emitter.incr("skills_count", by=len(parsed.skills))

    logger.info(
        "skill extraction: called=%d, skipped=%d, failed=%d, rows_added=%d",
        n_called,
        n_skipped,
        n_failed,
        len(new_rows),
    )

    if not new_rows:
        if posting_skills_existing is None or posting_skills_existing.empty:
            return _empty_posting_skills()
        return posting_skills_existing.copy()

    new_df = pd.DataFrame(new_rows, columns=list(POSTING_SKILLS_COLUMNS))
    if posting_skills_existing is None or posting_skills_existing.empty:
        return new_df
    return pd.concat(
        [posting_skills_existing[list(POSTING_SKILLS_COLUMNS)], new_df],
        ignore_index=True,
    )


def canonicalize_skills(
    posting_skills: pd.DataFrame,
    cv_profile: dict[str, Any],
    skill_map_params: dict[str, Any],
    skill_embedder,
) -> dict[str, str]:
    """Cluster raw skill names into canonical buckets."""
    profile = CVProfile.model_validate(cv_profile)

    posting_names = [] if posting_skills.empty else posting_skills["name"].tolist()
    cv_names = [s.name for s in profile.skills]

    raw_counts = canonicalize.collect_raw_counts(posting_names, cv_names)
    cfg = (skill_map_params or {}).get("canonicalize", {})
    mapping = canonicalize.build_canonical_map(
        raw_counts,
        embedder=skill_embedder,
        encode_prefix=cfg.get("encode_prefix", "clustering: "),
        distance_threshold=float(cfg.get("distance_threshold", 0.15)),
    )
    return mapping


def assign_skill_categories(
    canonical_skill_map: dict[str, str],
    skill_map_params: dict[str, Any],
    skill_embedder,
) -> dict[str, str]:
    """Second-pass clustering on the canonical vocabulary → {canonical: category}.

    Operates on the canonical names (values of ``canonical_skill_map``) — not
    the raw vocabulary — at a coarser distance threshold than canonicalize.
    Cheap: the embedder is already in memory from the canonicalize step.

    Per-canonical popularity (number of raw variants mapped to it) is passed
    as ``counts`` so the cluster-label tie-break picks the most-popular
    member instead of the absolute shortest string (a single-letter "c"
    sneaks into the vocab via "C the programming language" and wins
    otherwise).
    """
    if not canonical_skill_map:
        return {}
    from collections import Counter

    canonical_counts = Counter(canonical_skill_map.values())
    canonical_names = sorted(canonical_counts)
    cfg = (skill_map_params or {}).get("categorize", {})
    return canonicalize.build_category_map(
        canonical_names,
        embedder=skill_embedder,
        encode_prefix=cfg.get(
            "encode_prefix",
            (skill_map_params or {})
            .get("canonicalize", {})
            .get("encode_prefix", "clustering: "),
        ),
        distance_threshold=float(cfg.get("distance_threshold", 0.25)),
        counts=dict(canonical_counts),
    )


def build_skill_map(
    posting_skills: pd.DataFrame,
    canonical_skill_map: dict[str, str],
    skill_category_map: dict[str, str],
    cv_profile: dict[str, Any],
    skill_map_params: dict[str, Any],
    skill_embedder=None,
) -> dict[str, Any]:
    profile = CVProfile.model_validate(cv_profile)
    cv_names = [s.name for s in profile.skills]
    cfg = (skill_map_params or {}).get("map", {})
    viz = (skill_map_params or {}).get("viz", {})
    canonicalize_cfg = (skill_map_params or {}).get("canonicalize", {})
    return skill_map.build_map(
        posting_skills,
        canonical_map=canonical_skill_map,
        cv_skill_names=cv_names,
        category_map=skill_category_map or None,
        embedder=skill_embedder,
        min_node_count=int(cfg.get("min_node_count", 3)),
        tsne_perplexity=float(viz.get("tsne_perplexity", 30.0)),
        tsne_random_state=int(viz.get("tsne_random_state", 42)),
        encode_prefix=canonicalize_cfg.get("encode_prefix", "clustering: "),
    )


def render_skill_map_png(
    skill_map_data: dict[str, Any],
    skill_map_params: dict[str, Any],
):
    """Return (figure, nodes_dataframe) for the MatplotlibWriter + CSVDataset."""
    fig = skill_map.render_skill_map_figure(
        skill_map_data,
        viz_params=(skill_map_params or {}).get("viz") or {},
    )
    nodes_df = skill_map.nodes_dataframe(skill_map_data)
    return fig, nodes_df


__all__ = (
    "extract_posting_skills",
    "canonicalize_skills",
    "assign_skill_categories",
    "build_skill_map",
    "render_skill_map_png",
    "normalize_skill",
)
