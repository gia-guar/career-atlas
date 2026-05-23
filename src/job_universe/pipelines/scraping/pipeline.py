"""Scraping pipeline wiring."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from . import nodes


def create_pipeline(**_kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=nodes.fetch_adzuna,
                inputs=["params:scraping", "credentials"],
                outputs="adzuna_raw",
                name="fetch_adzuna",
            ),
            node(
                func=nodes.fetch_jobspy,
                inputs="params:scraping",
                outputs="jobspy_raw",
                name="fetch_jobspy",
            ),
            node(
                func=nodes.normalize_adzuna,
                inputs="adzuna_raw",
                outputs="adzuna_normalized",
                name="normalize_adzuna",
            ),
            node(
                func=nodes.normalize_jobspy,
                inputs="jobspy_raw",
                outputs="jobspy_normalized",
                name="normalize_jobspy",
            ),
            node(
                func=nodes.merge_and_dedupe,
                inputs=["adzuna_normalized", "jobspy_normalized"],
                outputs="postings_normalized",
                name="merge_and_dedupe",
            ),
            node(
                func=nodes.update_cumulative,
                inputs=["postings_normalized", "postings_cumulative_existing"],
                outputs="postings_cumulative",
                name="update_cumulative",
            ),
        ]
    )
