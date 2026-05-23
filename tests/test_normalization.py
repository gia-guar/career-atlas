"""Normalization from raw Adzuna payloads and JobSpy frames."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from job_universe.scraping import (
    normalize_adzuna,
    normalize_jobspy,
)
from job_universe.schemas import JOB_POSTING_COLUMNS, JobPosting


ADZUNA_SAMPLE = {
    "id": "987654",
    "title": "Senior Machine Learning Engineer",
    "company": {"display_name": "Acme GmbH"},
    "location": {"display_name": "Berlin, Germany"},
    "category": {"label": "IT"},
    "description": "Build ML systems.",
    "redirect_url": "https://www.adzuna.de/land/ad/987654",
    "created": "2026-05-15T10:30:00Z",
    "salary_min": 80000,
    "salary_max": 120000,
    "contract_time": "full_time",
    "_search_country": "de",
    "_search_query": "machine learning engineer",
}


JOBSPY_SAMPLE = pd.DataFrame(
    [
        {
            "title": "ML Engineer",
            "company": "Globex",
            "location": "Munich, DE",
            "country": "Germany",
            "is_remote": True,
            "description": "Train models.",
            "job_url": "https://linkedin.com/jobs/view/111",
            "job_url_direct": "https://example.com/apply",
            "date_posted": "2026-05-10",
            "min_amount": 70000,
            "max_amount": 100000,
            "currency": "EUR",
            "job_type": "fulltime",
            "source": "linkedin",
            "search_query": "ml engineer",
        }
    ]
)


class TestNormalizeAdzuna:
    def test_columns_match_schema(self):
        df = normalize_adzuna([ADZUNA_SAMPLE])
        assert list(df.columns) == list(JOB_POSTING_COLUMNS)

    def test_basic_fields_populated(self):
        df = normalize_adzuna([ADZUNA_SAMPLE])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["source"] == "adzuna"
        assert row["title"] == "Senior Machine Learning Engineer"
        assert row["title_normalized"] == "senior machine learning engineer"
        assert row["company"] == "Acme GmbH"
        assert row["company_normalized"] == "acme gmbh"
        assert row["country"] == "de"

    def test_raw_keyed_by_source(self):
        df = normalize_adzuna([ADZUNA_SAMPLE])
        raw = df.iloc[0]["raw"]
        assert isinstance(raw, dict)
        assert "adzuna" in raw
        assert raw["adzuna"]["id"] == "987654"

    def test_pydantic_validates(self):
        df = normalize_adzuna([ADZUNA_SAMPLE])
        JobPosting.model_validate(df.iloc[0].to_dict())

    def test_empty_input_returns_empty_frame_with_columns(self):
        df = normalize_adzuna([])
        assert df.empty
        assert list(df.columns) == list(JOB_POSTING_COLUMNS)

    def test_missing_title_row_skipped(self):
        bad = dict(ADZUNA_SAMPLE)
        bad["title"] = ""
        df = normalize_adzuna([bad, ADZUNA_SAMPLE])
        assert len(df) == 1


class TestNormalizeJobspy:
    def test_columns_match_schema(self):
        df = normalize_jobspy(JOBSPY_SAMPLE)
        assert list(df.columns) == list(JOB_POSTING_COLUMNS)

    def test_basic_fields_populated(self):
        df = normalize_jobspy(JOBSPY_SAMPLE)
        row = df.iloc[0]
        assert row["source"] == "linkedin"
        assert row["title_normalized"] == "ml engineer"
        assert bool(row["is_remote"]) is True
        assert row["salary_currency"] == "EUR"

    def test_raw_keyed_by_source(self):
        df = normalize_jobspy(JOBSPY_SAMPLE)
        raw = df.iloc[0]["raw"]
        assert "linkedin" in raw

    def test_pydantic_validates(self):
        df = normalize_jobspy(JOBSPY_SAMPLE)
        JobPosting.model_validate(df.iloc[0].to_dict())

    def test_empty_input_returns_empty_frame_with_columns(self):
        df = normalize_jobspy(pd.DataFrame())
        assert df.empty
        assert list(df.columns) == list(JOB_POSTING_COLUMNS)

    def test_malformed_url_accepted(self):
        bad = JOBSPY_SAMPLE.copy()
        bad.loc[0, "job_url"] = ""
        bad.loc[0, "job_url_direct"] = "  "
        df = normalize_jobspy(bad)
        row = df.iloc[0]
        assert row["url"] is None


def test_adzuna_and_jobspy_can_produce_overlapping_ids():
    """Both sources, given the same title+company+location, must hash to the same id."""
    adzuna_payload = dict(ADZUNA_SAMPLE)
    adzuna_payload["title"] = "ML Engineer"
    adzuna_payload["company"] = {"display_name": "Globex"}
    adzuna_payload["location"] = {"display_name": "Munich, DE"}

    df_a = normalize_adzuna([adzuna_payload])
    df_j = normalize_jobspy(JOBSPY_SAMPLE)
    assert df_a.iloc[0]["id"] == df_j.iloc[0]["id"]
