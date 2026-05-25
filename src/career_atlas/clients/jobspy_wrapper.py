"""Per-site isolated wrapper around python-jobspy.

JobSpy can blow up on individual sites (rate limits, layout changes, network
hiccups). We loop sites one at a time so a broken site only loses its own rows
instead of the whole run.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_jobspy(
    query: str,
    sites: Iterable[str],
    location: str | None = None,
    results_wanted: int = 100,
    hours_old: int = 720,
    country_indeed: str = "Germany",
) -> pd.DataFrame:
    """Scrape postings from each site independently and concatenate the results.

    Failure in one site is logged and skipped; the call always returns a
    DataFrame (possibly empty). A `source` column is added per site so
    downstream normalisation can branch on it.
    """
    try:
        from jobspy import scrape_jobs
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "python-jobspy is not installed; run `uv pip install python-jobspy`"
        ) from exc

    frames: list[pd.DataFrame] = []
    for site in sites:
        try:
            df = scrape_jobs(
                site_name=[site],
                search_term=query,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                country_indeed=country_indeed,
            )
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            logger.warning(
                "jobspy site %s failed for query=%r: %s", site, query, exc
            )
            continue
        if df is None or df.empty:
            logger.info("jobspy site %s returned no rows for query=%r", site, query)
            continue
        df = df.copy()
        df["source"] = site
        df["search_query"] = query
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
