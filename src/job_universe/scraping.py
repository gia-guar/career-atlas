"""Reusable scraping helpers: fetch / normalize / dedupe / merge / append.

These functions are provider-agnostic w.r.t. *what* is scraped — the
caller passes the search config (queries, countries, locations) as a dict.
The CV-driven pipeline derives that config from the user's CV via the LLM;
nothing here is specific to ML/AI or any other profession.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from job_universe.clients.adzuna import AdzunaClient
from job_universe.clients.jobspy_wrapper import fetch_jobspy as _scrape_jobspy
from job_universe.schemas import (
    JOB_POSTING_COLUMNS,
    JobPosting,
    _content_hash,
    _normalize_text,
)

logger = logging.getLogger(__name__)

_SOURCE_PRIORITY = {"linkedin": 0, "indeed": 1, "glassdoor": 2, "adzuna": 3}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def empty_postings_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical JobPosting columns."""
    df = pd.DataFrame({col: pd.Series(dtype=object) for col in JOB_POSTING_COLUMNS})
    return df


# ---------------------------------------------------------------------------
# Fetch nodes
# ---------------------------------------------------------------------------


def fetch_adzuna(
    params: dict[str, Any], credentials: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fetch postings from Adzuna across the configured queries × countries."""
    creds = credentials.get("adzuna") or {}
    app_id = creds.get("app_id")
    app_key = creds.get("app_key")
    if not (app_id and app_key):
        logger.warning("adzuna credentials missing — skipping fetch")
        return []

    cfg = params.get("adzuna", {})
    countries = cfg.get("countries", [])
    queries = cfg.get("queries", [])
    if not countries or not queries:
        logger.info("adzuna fetch skipped — no countries or queries configured")
        return []
    rpm = cfg.get("requests_per_minute", 20)
    results_per_page = cfg.get("results_per_page", 50)
    max_pages = cfg.get("max_pages", 5)
    max_days_old = cfg.get("max_days_old", 30)

    out: list[dict[str, Any]] = []
    with AdzunaClient(
        app_id=app_id,
        app_key=app_key,
        requests_per_minute=rpm,
        results_per_page=results_per_page,
        max_days_old=max_days_old,
    ) as client:
        for country in countries:
            for query in queries:
                try:
                    rows = client.search_all(country, query, max_pages=max_pages)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "adzuna query failed (%s / %r): %s", country, query, exc
                    )
                    continue
                for row in rows:
                    row["_search_country"] = country
                    row["_search_query"] = query
                out.extend(rows)
    logger.info("adzuna fetched %d rows", len(out))
    return out


def fetch_jobspy(params: dict[str, Any]) -> pd.DataFrame:
    """Fetch postings from JobSpy across the configured queries × locations × sites."""
    cfg = params.get("jobspy", {})
    sites = cfg.get("sites", ["linkedin", "indeed"])
    queries = cfg.get("queries", [])
    results_wanted = cfg.get("results_wanted", 100)
    hours_old = cfg.get("hours_old", 720)

    locations = cfg.get("locations") or []
    if not locations or not queries:
        logger.info("jobspy fetch skipped — no locations or queries configured")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for loc in locations:
        location = loc.get("name") if isinstance(loc, dict) else loc
        country_indeed = (
            loc.get("country_indeed") if isinstance(loc, dict) else None
        ) or location
        for query in queries:
            df = _scrape_jobspy(
                query=query,
                sites=sites,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                country_indeed=country_indeed,
            )
            if not df.empty:
                df = df.copy()
                df["_search_location"] = location
                frames.append(df)
    if not frames:
        logger.info("jobspy returned 0 rows across all queries × locations")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Normalize nodes
# ---------------------------------------------------------------------------


def _build_row(
    *,
    source: str,
    source_id: str | None,
    url: str | None,
    title: str | None,
    company: str | None,
    location: str | None,
    country: str | None,
    is_remote: bool,
    description: str | None,
    posted_at: datetime | None,
    salary_min: float | None,
    salary_max: float | None,
    salary_currency: str | None,
    job_type: str | None,
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    title = (title or "").strip()
    if not title:
        return None
    title_norm = _normalize_text(title)
    if not title_norm:
        return None
    company_norm = _normalize_text(company) if company else None
    posting_id = _content_hash(title_norm, company_norm, location)
    now = _now()
    return {
        "id": posting_id,
        "source": source,
        "source_id": source_id,
        "url": (url.strip() if isinstance(url, str) and url.strip() else None),
        "title": title,
        "title_normalized": title_norm,
        "company": company,
        "company_normalized": company_norm,
        "location": location,
        "country": country,
        "is_remote": bool(is_remote),
        "description": description,
        "posted_at": posted_at,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
        "job_type": job_type,
        "first_seen_at": now,
        "last_seen_at": now,
        "raw": {source: raw},
    }


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def normalize_adzuna(raw: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in raw or []:
        company_obj = record.get("company") or {}
        location_obj = record.get("location") or {}
        category_obj = record.get("category") or {}
        location_str = location_obj.get("display_name") or None
        country = (record.get("_search_country") or "").lower() or None
        contract_time = record.get("contract_time")
        contract_type = record.get("contract_type")
        job_type = contract_time or contract_type
        row = _build_row(
            source="adzuna",
            source_id=str(record.get("id")) if record.get("id") is not None else None,
            url=record.get("redirect_url"),
            title=record.get("title"),
            company=company_obj.get("display_name"),
            location=location_str,
            country=country,
            is_remote=False,  # Adzuna doesn't expose this reliably
            description=record.get("description"),
            posted_at=_parse_dt(record.get("created")),
            salary_min=_coerce_float(record.get("salary_min")),
            salary_max=_coerce_float(record.get("salary_max")),
            salary_currency=None,
            job_type=job_type or category_obj.get("label"),
            raw=record,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return empty_postings_frame()
    return pd.DataFrame(rows, columns=list(JOB_POSTING_COLUMNS))


def normalize_jobspy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_postings_frame()
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        site = record.get("source") or "linkedin"
        title = record.get("title")
        company = record.get("company")
        location = record.get("location")
        country = record.get("country") or None
        if isinstance(country, str):
            country = country.lower() or None
        is_remote = bool(record.get("is_remote") or False)
        description = record.get("description")
        posted_at = _parse_dt(record.get("date_posted"))
        salary_min = _coerce_float(record.get("min_amount"))
        salary_max = _coerce_float(record.get("max_amount"))
        salary_currency = record.get("currency")
        job_type = record.get("job_type")
        url = record.get("job_url") or record.get("job_url_direct")
        source_id = record.get("id") or record.get("job_url")
        row = _build_row(
            source=site,
            source_id=str(source_id) if source_id is not None else None,
            url=url,
            title=title,
            company=company,
            location=location,
            country=country,
            is_remote=is_remote,
            description=description,
            posted_at=posted_at,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            job_type=job_type,
            raw=record,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return empty_postings_frame()
    return pd.DataFrame(rows, columns=list(JOB_POSTING_COLUMNS))


# ---------------------------------------------------------------------------
# Merge / dedupe / cumulative
# ---------------------------------------------------------------------------


def _merge_raw_series(series: pd.Series) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in series:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def merge_and_dedupe(
    df_adzuna: pd.DataFrame, df_jobspy: pd.DataFrame
) -> pd.DataFrame:
    """Combine two source frames, drop duplicate ids, keep best description.

    Tie-breaker order per `id`:
      1. LinkedIn row with non-empty description (longest if multiple)
      2. Longest non-empty description from any other source
      3. Source-priority order (linkedin < indeed < glassdoor < adzuna)
    `raw` dicts from every row in the group are merged by source key, and
    `is_remote` is OR'd across rows.
    """
    frames = [df for df in (df_adzuna, df_jobspy) if df is not None and not df.empty]
    if not frames:
        return empty_postings_frame()

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return empty_postings_frame()

    work = combined.copy()
    desc_len = work["description"].fillna("").astype(str).str.len()
    is_linkedin_with_desc = ((work["source"] == "linkedin") & (desc_len > 0)).astype(int)
    src_priority = work["source"].map(_SOURCE_PRIORITY).fillna(99)
    work = work.assign(
        _lk=-is_linkedin_with_desc,
        _dl=-desc_len,
        _sp=src_priority,
    ).sort_values(by=["id", "_lk", "_dl", "_sp"], kind="stable")

    merged_raw: dict[str, dict[str, Any]] = {}
    is_remote_any: dict[str, bool] = {}
    for gid, group in work.groupby("id", sort=False):
        merged_raw[gid] = _merge_raw_series(group["raw"])
        is_remote_any[gid] = bool(group["is_remote"].fillna(False).any())

    winners = work.drop_duplicates(subset="id", keep="first").copy()
    winners["raw"] = winners["id"].map(merged_raw)
    winners["is_remote"] = winners["id"].map(is_remote_any)

    return winners[list(JOB_POSTING_COLUMNS)].reset_index(drop=True)


def update_cumulative(
    df_new: pd.DataFrame, df_existing: pd.DataFrame
) -> pd.DataFrame:
    """Append-only merge that preserves `first_seen_at` and refreshes `last_seen_at`."""
    if df_existing is None or df_existing.empty:
        if df_new is None or df_new.empty:
            return empty_postings_frame()
        return df_new[list(JOB_POSTING_COLUMNS)].copy()
    if df_new is None or df_new.empty:
        return df_existing[list(JOB_POSTING_COLUMNS)].copy()

    now = _now()
    existing_ids = set(df_existing["id"].tolist())
    new_ids = set(df_new["id"].tolist())

    refreshed_ids = existing_ids & new_ids
    brand_new_ids = new_ids - existing_ids

    existing_updated = df_existing.copy()
    if refreshed_ids:
        mask = existing_updated["id"].isin(refreshed_ids)
        existing_updated.loc[mask, "last_seen_at"] = now
        new_indexed = df_new.set_index("id")
        for col in (
            "description",
            "url",
            "salary_min",
            "salary_max",
            "salary_currency",
            "job_type",
            "is_remote",
            "posted_at",
            "raw",
        ):
            if col not in new_indexed.columns:
                continue
            for idx in existing_updated.index[mask]:
                pid = existing_updated.at[idx, "id"]
                if pid in new_indexed.index:
                    val = new_indexed.at[pid, col]
                    if col == "raw" and isinstance(val, dict):
                        merged = dict(existing_updated.at[idx, "raw"] or {})
                        merged.update(val)
                        existing_updated.at[idx, "raw"] = merged
                    elif col != "raw" and val is not None and not pd.isna(val):
                        existing_updated.at[idx, col] = val

    additions = df_new[df_new["id"].isin(brand_new_ids)].copy()
    out = pd.concat([existing_updated, additions], ignore_index=True)
    return out[list(JOB_POSTING_COLUMNS)]


def validate_sample(df: pd.DataFrame, sample_size: int = 5) -> pd.DataFrame:
    """Validate up to `sample_size` rows through Pydantic; raise on first failure."""
    if df is None or df.empty:
        return df
    sample = df.head(sample_size).to_dict(orient="records")
    for row in sample:
        JobPosting.model_validate(row)
    return df
