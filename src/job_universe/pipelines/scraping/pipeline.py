"""Scraping pipeline wiring.

The single scraping flow in this project. There is no hard-coded list of
queries, countries, or locations — everything is derived from the user's
CV by the `cv_extraction` pipeline and handed in via
`cv_derived_scraping_params`. The actual fetch / normalize / dedupe /
append-only-cumulative logic lives in `job_universe.scraping`.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from job_universe import scraping


def create_pipeline(**_kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=scraping.fetch_adzuna,
                inputs=["cv_derived_scraping_params", "credentials", "progress_emitter"],
                outputs="adzuna_raw",
                name="fetch_adzuna",
            ),
            node(
                func=scraping.fetch_jobspy,
                inputs=["cv_derived_scraping_params", "progress_emitter"],
                outputs="jobspy_raw",
                name="fetch_jobspy",
            ),
            node(
                func=scraping.normalize_adzuna,
                inputs="adzuna_raw",
                outputs="adzuna_normalized",
                name="normalize_adzuna",
            ),
            node(
                func=scraping.normalize_jobspy,
                inputs="jobspy_raw",
                outputs="jobspy_normalized",
                name="normalize_jobspy",
            ),
            node(
                func=scraping.merge_and_dedupe,
                inputs=["adzuna_normalized", "jobspy_normalized"],
                outputs="postings_normalized",
                name="merge_and_dedupe",
            ),
            node(
                func=scraping.update_cumulative,
                inputs=["postings_normalized", "postings_cumulative_existing"],
                outputs="postings_cumulative",
                name="update_cumulative",
            ),
        ]
    )
