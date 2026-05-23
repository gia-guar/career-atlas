"""merge_and_dedupe + update_cumulative behaviour."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from job_universe import scraping as nodes
from job_universe.scraping import (
    empty_postings_frame,
    merge_and_dedupe,
    update_cumulative,
)
from job_universe.schemas import JOB_POSTING_COLUMNS


def _row(
    *,
    pid: str,
    source: str,
    description: str | None = None,
    title: str = "ML Engineer",
    raw: dict | None = None,
    first_seen_at: datetime | None = None,
    last_seen_at: datetime | None = None,
) -> dict:
    now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "id": pid,
        "source": source,
        "source_id": f"{source}-{pid}",
        "url": f"https://{source}.example/{pid}",
        "title": title,
        "title_normalized": title.lower(),
        "company": "Acme",
        "company_normalized": "acme",
        "location": "Berlin",
        "country": "de",
        "is_remote": False,
        "description": description,
        "posted_at": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "job_type": None,
        "first_seen_at": first_seen_at or now,
        "last_seen_at": last_seen_at or now,
        "raw": raw or {source: {"id": pid}},
    }


class TestMergeAndDedupe:
    def test_distinct_ids_kept(self):
        df_a = pd.DataFrame([_row(pid="aaa", source="adzuna")])
        df_b = pd.DataFrame([_row(pid="bbb", source="linkedin")])
        out = merge_and_dedupe(df_a, df_b)
        assert len(out) == 2
        assert set(out["id"]) == {"aaa", "bbb"}

    def test_collision_prefers_linkedin_description(self):
        df_a = pd.DataFrame(
            [_row(pid="x", source="adzuna", description="adzuna short")]
        )
        df_b = pd.DataFrame(
            [_row(pid="x", source="linkedin", description="linkedin much longer text")]
        )
        out = merge_and_dedupe(df_a, df_b)
        assert len(out) == 1
        assert out.iloc[0]["description"] == "linkedin much longer text"

    def test_collision_no_linkedin_picks_longest(self):
        df_a = pd.DataFrame(
            [_row(pid="x", source="adzuna", description="short")]
        )
        df_b = pd.DataFrame(
            [
                _row(pid="x", source="indeed", description="medium length here"),
                _row(pid="x", source="glassdoor", description="this one is the longest of the three"),
            ]
        )
        out = merge_and_dedupe(df_a, df_b)
        assert len(out) == 1
        assert "longest" in out.iloc[0]["description"]

    def test_collision_merges_raw_by_source(self):
        df_a = pd.DataFrame([_row(pid="x", source="adzuna", raw={"adzuna": {"k": 1}})])
        df_b = pd.DataFrame([_row(pid="x", source="linkedin", raw={"linkedin": {"k": 2}})])
        out = merge_and_dedupe(df_a, df_b)
        merged = out.iloc[0]["raw"]
        assert merged == {"adzuna": {"k": 1}, "linkedin": {"k": 2}}
        assert not isinstance(merged, list)

    def test_empty_inputs(self):
        out = merge_and_dedupe(empty_postings_frame(), empty_postings_frame())
        assert out.empty
        assert list(out.columns) == list(JOB_POSTING_COLUMNS)


class TestUpdateCumulative:
    def test_first_run_returns_new(self):
        df_new = pd.DataFrame([_row(pid="a", source="adzuna")])
        out = update_cumulative(df_new, empty_postings_frame())
        assert len(out) == 1
        assert out.iloc[0]["id"] == "a"

    def test_new_id_appended(self):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df_existing = pd.DataFrame(
            [_row(pid="a", source="adzuna", first_seen_at=t0, last_seen_at=t0)]
        )
        df_new = pd.DataFrame([_row(pid="b", source="linkedin")])
        out = update_cumulative(df_new, df_existing)
        assert len(out) == 2
        assert set(out["id"]) == {"a", "b"}

    def test_refreshed_id_updates_last_seen_only(self, monkeypatch):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(nodes, "_now", lambda: t1)

        df_existing = pd.DataFrame(
            [_row(pid="a", source="adzuna", first_seen_at=t0, last_seen_at=t0)]
        )
        df_new = pd.DataFrame(
            [_row(pid="a", source="adzuna", description="updated desc")]
        )
        out = update_cumulative(df_new, df_existing)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["first_seen_at"] == t0
        assert row["last_seen_at"] == t1
        assert row["description"] == "updated desc"

    def test_no_duplicate_rows_on_rerun(self, monkeypatch):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t1 = datetime(2026, 5, 23, tzinfo=timezone.utc)
        monkeypatch.setattr(nodes, "_now", lambda: t1)

        df_existing = pd.DataFrame(
            [
                _row(pid="a", source="adzuna", first_seen_at=t0, last_seen_at=t0),
                _row(pid="b", source="linkedin", first_seen_at=t0, last_seen_at=t0),
            ]
        )
        df_new = pd.DataFrame(
            [
                _row(pid="a", source="adzuna"),
                _row(pid="b", source="linkedin"),
                _row(pid="c", source="indeed"),
            ]
        )
        out = update_cumulative(df_new, df_existing)
        assert len(out) == 3
        assert set(out["id"]) == {"a", "b", "c"}

    def test_raw_merged_for_refreshed_id(self, monkeypatch):
        t1 = datetime(2026, 5, 23, tzinfo=timezone.utc)
        monkeypatch.setattr(nodes, "_now", lambda: t1)

        df_existing = pd.DataFrame(
            [_row(pid="a", source="adzuna", raw={"adzuna": {"v": 1}})]
        )
        df_new = pd.DataFrame(
            [_row(pid="a", source="linkedin", raw={"linkedin": {"v": 2}})]
        )
        out = update_cumulative(df_new, df_existing)
        merged_raw = out.iloc[0]["raw"]
        assert merged_raw == {"adzuna": {"v": 1}, "linkedin": {"v": 2}}
