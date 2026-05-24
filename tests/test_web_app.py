"""FastAPI endpoint smoke tests.

The pipeline-running internals are stubbed so we don't need Ollama, the
network, or the embedding model. We exercise:

* ``/api/status`` — file-existence flags
* ``/api/cv``    — writes the CV, runs the pipeline (stubbed), returns parsed JSON
* ``/api/build`` — kicks off a job, the second concurrent call returns 409
* ``/api/build/events`` — SSE stream emits queued events and terminates on done
* ``/api/graph`` — returns the on-disk JSON
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from job_universe.web import app as web_app
from job_universe.web import runner


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with the project data dir pointed at a temp directory and
    pipeline-running functions replaced with stubs."""
    # Reset the singleton job registry between tests.
    web_app._JOBS.clear()

    data_root = tmp_path / "data"
    (data_root / "01_raw" / "cv").mkdir(parents=True)
    (data_root / "02_intermediate").mkdir(parents=True)
    (data_root / "03_primary").mkdir(parents=True)

    # Patch all the path constants the app reads.
    monkeypatch.setattr(runner, "_CV_PATH", data_root / "01_raw" / "cv" / "cv.md")
    monkeypatch.setattr(
        runner, "_CV_PROFILE_PATH", data_root / "02_intermediate" / "cv_profile.json"
    )
    monkeypatch.setattr(
        runner,
        "_TARGETING_PATH",
        data_root / "02_intermediate" / "cv_derived_scraping_params.json",
    )
    monkeypatch.setattr(
        runner, "_POSTINGS_PATH", data_root / "03_primary" / "job_postings.parquet"
    )
    monkeypatch.setattr(
        runner, "_GRAPH_JSON_PATH", data_root / "02_intermediate" / "skill_graph.json"
    )
    monkeypatch.setattr(
        runner,
        "_POSTING_SKILLS_PATH",
        data_root / "02_intermediate" / "posting_skills.parquet",
    )
    monkeypatch.setattr(
        runner,
        "_CANONICAL_MAP_PATH",
        data_root / "02_intermediate" / "canonical_skill_map.json",
    )
    runner._LOOKUP_CACHE.update(
        {"mtimes": None, "posting_meta_by_id": None, "raw_to_canonical": None, "canonical_to_postings": None}
    )

    # Stub cv_extraction: write the expected outputs without invoking Kedro.
    def fake_cv_extraction():
        runner._CV_PROFILE_PATH.write_text(
            json.dumps(
                {
                    "skills": [
                        {"name": "Python", "kind": "tool", "proficiency": "expert"},
                        {"name": "Comm", "kind": "skill", "proficiency": "used"},
                    ],
                    "role_titles": ["data engineer"],
                    "seniority": "senior",
                    "years_of_experience": 5,
                    "preferred_locations": ["Berlin"],
                    "summary": "test",
                }
            )
        )
        # Merged shape: matches what derive_targeted_scraping_params actually writes.
        runner._TARGETING_PATH.write_text(
            json.dumps(
                {
                    "adzuna": {
                        "queries": ["data engineer"],
                        "countries": ["de"],
                    },
                    "jobspy": {
                        "queries": ["data engineer"],
                        "locations": [{"name": "Berlin", "country_indeed": "germany"}],
                    },
                }
            )
        )

    monkeypatch.setattr(runner, "run_cv_extraction", fake_cv_extraction)

    yield TestClient(web_app.app)


def test_status_reports_missing_artifacts(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.json() == {
        "has_cv_profile": False,
        "has_postings": False,
        "has_graph": False,
    }


def test_cv_endpoint_writes_file_and_returns_profile(client):
    r = client.post("/api/cv", json={"cv_text": "I work in data"})
    assert r.status_code == 200
    body = r.json()
    assert "profile" in body and "targeting" in body
    assert body["profile"]["skills"][0]["name"] == "Python"
    assert body["targeting"]["adzuna"]["queries"] == ["data engineer"]
    assert body["targeting"]["adzuna"]["countries"] == ["de"]
    assert body["targeting"]["jobspy"]["locations"][0]["name"] == "Berlin"
    # File was written.
    assert runner._CV_PATH.read_text() == "I work in data"


def test_cv_endpoint_rejects_empty_body(client):
    r = client.post("/api/cv", json={"cv_text": ""})
    assert r.status_code == 422  # pydantic min_length


def test_graph_endpoint_404_until_built(client):
    r = client.get("/api/graph")
    assert r.status_code == 404


def test_graph_endpoint_returns_on_disk_json(client):
    graph = {"nodes": [{"id": "python", "count": 5, "user_has": True}], "edges": []}
    runner._GRAPH_JSON_PATH.write_text(json.dumps(graph))
    r = client.get("/api/graph")
    assert r.status_code == 200
    assert r.json() == graph


def test_build_returns_job_id_and_409_when_in_flight(client, monkeypatch):
    """First /api/build call returns a job_id and 200; the second returns 409
    while the first is still in 'running' state."""
    started = threading.Event()
    release = threading.Event()

    def fake_start(self):
        # Mark the job as running but don't actually do anything until released.
        self.status = "running"
        started.set()
        threading.Thread(
            target=lambda: (release.wait(), setattr(self, "status", "done")),
            daemon=True,
        ).start()

    monkeypatch.setattr(runner.BuildJob, "start", fake_start)

    r1 = client.post("/api/build")
    assert r1.status_code == 200
    job_id = r1.json()["job_id"]
    assert job_id

    started.wait(timeout=2.0)

    r2 = client.post("/api/build")
    assert r2.status_code == 409

    # Let the first job complete so other tests are not affected.
    release.set()


def test_build_events_stream_emits_queued_events(client, monkeypatch):
    """Push a few events into the job's queue and verify the SSE response
    delivers them in order, terminating on the 'done' event."""

    def fake_start(self):
        self.status = "running"

        async def producer():
            await self.queue.put({"type": "postings_count", "value": 10})
            await self.queue.put({"type": "skills_count", "value": 5})
            await self.queue.put({"type": "done", "value": True})

        asyncio.run_coroutine_threadsafe(producer(), self.loop)

    monkeypatch.setattr(runner.BuildJob, "start", fake_start)

    r = client.post("/api/build")
    job_id = r.json()["job_id"]

    with client.stream("GET", f"/api/build/events?job_id={job_id}") as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if not line:
                continue
            assert line.startswith("data:")
            payload = json.loads(line[len("data:"):].strip())
            events.append(payload)
            if payload["type"] == "done":
                break

    assert [e["type"] for e in events] == ["postings_count", "skills_count", "done"]


def test_build_events_unknown_job_id(client):
    r = client.get("/api/build/events", params={"job_id": "deadbeef"})
    assert r.status_code == 404


def test_index_html_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SKILL-GRAPH" in r.text


def test_static_assets_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "--accent" in r.text


def test_skill_postings_endpoint(client, tmp_path):
    """End-to-end: write minimal job_postings.parquet + posting_skills.parquet
    + canonical_skill_map.json, then ask /api/skill/pytorch/postings."""
    import pandas as pd

    # job_postings
    postings_df = pd.DataFrame(
        [
            {
                "id": "p1",
                "title": "Senior ML Engineer",
                "company": "DeepMind",
                "location": "London",
                "url": "https://example.com/p1",
                "source": "linkedin",
            },
            {
                "id": "p2",
                "title": "Junior Data Scientist",
                "company": "Acme",
                "location": "Berlin",
                "url": "https://example.com/p2",
                "source": "indeed",
            },
        ]
    )
    postings_df.to_parquet(runner._POSTINGS_PATH, index=False)

    # posting_skills (raw names) — note one row maps via canonical_map below.
    skills_df = pd.DataFrame(
        [
            {"posting_id": "p1", "name": "PyTorch", "kind": "tool"},
            {"posting_id": "p1", "name": "CUDA", "kind": "tool"},
            {"posting_id": "p2", "name": "pytorch", "kind": "tool"},
        ]
    )
    posting_skills_path = (
        tmp_path / "data" / "02_intermediate" / "posting_skills.parquet"
    )
    skills_df.to_parquet(posting_skills_path, index=False)

    canonical_map = {"pytorch": "pytorch", "cuda": "cuda"}
    canonical_path = (
        tmp_path / "data" / "02_intermediate" / "canonical_skill_map.json"
    )
    canonical_path.write_text(json.dumps(canonical_map))

    # Patch the cache paths the runner uses.
    runner._POSTING_SKILLS_PATH = posting_skills_path
    runner._CANONICAL_MAP_PATH = canonical_path
    # Force cache reload.
    runner._LOOKUP_CACHE["mtimes"] = None

    r = client.get("/api/skill/pytorch/postings")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "pytorch"
    assert body["total"] == 2
    titles = [p["title"] for p in body["postings"]]
    assert "Senior ML Engineer" in titles
    assert "Junior Data Scientist" in titles
    # LinkedIn entry sorts before Indeed by source priority.
    assert body["postings"][0]["source"] == "linkedin"


def test_skill_postings_unknown_skill_returns_empty(client, tmp_path):
    import pandas as pd

    pd.DataFrame(
        [
            {
                "id": "p1",
                "title": "x",
                "company": "y",
                "location": "z",
                "url": "u",
                "source": "indeed",
            }
        ]
    ).to_parquet(runner._POSTINGS_PATH, index=False)
    posting_skills_path = (
        tmp_path / "data" / "02_intermediate" / "posting_skills.parquet"
    )
    pd.DataFrame(
        [{"posting_id": "p1", "name": "Python", "kind": "tool"}]
    ).to_parquet(posting_skills_path, index=False)
    canonical_path = (
        tmp_path / "data" / "02_intermediate" / "canonical_skill_map.json"
    )
    canonical_path.write_text(json.dumps({"python": "python"}))
    runner._POSTING_SKILLS_PATH = posting_skills_path
    runner._CANONICAL_MAP_PATH = canonical_path
    runner._LOOKUP_CACHE["mtimes"] = None

    r = client.get("/api/skill/notathing/postings")
    assert r.status_code == 200
    assert r.json() == {"name": "notathing", "total": 0, "postings": []}


def test_skill_postings_when_artifacts_missing(client, monkeypatch):
    # Point the path constants at nonexistent files.
    monkeypatch.setattr(
        runner, "_POSTING_SKILLS_PATH", runner._POSTING_SKILLS_PATH.parent / "missing.parquet"
    )
    monkeypatch.setattr(
        runner, "_CANONICAL_MAP_PATH", runner._CANONICAL_MAP_PATH.parent / "missing.json"
    )
    runner._LOOKUP_CACHE["mtimes"] = None

    r = client.get("/api/skill/pytorch/postings")
    assert r.status_code == 200
    assert r.json()["postings"] == []
