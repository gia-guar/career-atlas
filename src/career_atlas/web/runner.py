"""Threaded driver that runs scraping + skill_map for the web UI.

Each /api/build request spawns one ``BuildJob``. The job binds a fresh
``ProgressEmitter`` to an ``asyncio.Queue`` owned by the FastAPI event loop,
sets the module-level ``progress.CURRENT`` slot so ``ProgressHook`` picks it
up, then runs the two pipelines back-to-back on a worker thread. SSE
consumers read events off the queue until a terminal ``done`` / ``error``.

Only one job at a time per process — a second concurrent /api/build returns
409 from the app layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from career_atlas.web import progress
from career_atlas.web.progress import ProgressEmitter

logger = logging.getLogger(__name__)

# src/career_atlas/web/runner.py → parents[3] is the kedro project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


_BOOTSTRAPPED = False


def ensure_bootstrapped() -> None:
    """Call ``bootstrap_project`` once per process. Safe to call repeatedly."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    bootstrap_project(PROJECT_ROOT)
    _BOOTSTRAPPED = True


class BuildJob:
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.job_id = uuid.uuid4().hex
        self.queue = queue
        self.loop = loop
        self.emitter = ProgressEmitter()
        self.status = "pending"
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.emitter.bind(self.queue, self.loop)
        progress.CURRENT = self.emitter
        self.status = "running"
        self._thread = threading.Thread(
            target=self._run, name=f"build-{self.job_id}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            ensure_bootstrapped()
            with KedroSession.create(project_path=PROJECT_ROOT) as session:
                session.run(pipeline_name="scraping")
            with KedroSession.create(project_path=PROJECT_ROOT) as session:
                session.run(pipeline_name="skill_map")
            self.status = "done"
            self.emitter.emit("done", True)
        except Exception as exc:  # noqa: BLE001 — surfaced to the SSE client
            logger.exception("build job %s failed", self.job_id)
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            self.emitter.emit("error", self.error)
        finally:
            progress.CURRENT = None
            self.emitter.unbind()


def run_cv_extraction() -> None:
    """Run the cv_extraction pipeline once. Blocking — call via ``to_thread``."""
    ensure_bootstrapped()
    with KedroSession.create(project_path=PROJECT_ROOT) as session:
        session.run(pipeline_name="cv_extraction")


def project_data_dir() -> Path:
    return PROJECT_ROOT / "data"


_CV_PATH = project_data_dir() / "01_raw" / "cv" / "cv.md"
_CV_PROFILE_PATH = project_data_dir() / "02_intermediate" / "cv_profile.json"
_TARGETING_PATH = (
    project_data_dir() / "02_intermediate" / "cv_derived_scraping_params.json"
)
_POSTINGS_PATH = project_data_dir() / "03_primary" / "job_postings.parquet"
_MAP_JSON_PATH = project_data_dir() / "02_intermediate" / "skill_map.json"


def paths() -> dict[str, Path]:
    return {
        "cv": _CV_PATH,
        "cv_profile": _CV_PROFILE_PATH,
        "targeting": _TARGETING_PATH,
        "postings": _POSTINGS_PATH,
        "map": _MAP_JSON_PATH,
    }


def write_cv_text(text: str) -> Path:
    _CV_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CV_PATH.write_text(text, encoding="utf-8")
    return _CV_PATH


def status_flags() -> dict[str, Any]:
    return {
        "has_cv_profile": _CV_PROFILE_PATH.exists(),
        "has_postings": _POSTINGS_PATH.exists(),
        "has_map": _MAP_JSON_PATH.exists(),
    }


# Posting-lookup cache: load the three artefacts lazily and invalidate on
# mtime change. Cheap enough to re-read entirely when the parquet rebuilds.

_POSTING_SKILLS_PATH = (
    project_data_dir() / "02_intermediate" / "posting_skills.parquet"
)
_CANONICAL_MAP_PATH = (
    project_data_dir() / "02_intermediate" / "canonical_skill_map.json"
)
_LOOKUP_CACHE: dict[str, Any] = {
    "mtimes": None,
    "posting_meta_by_id": None,  # dict[str, dict]
    "raw_to_canonical": None,    # dict[str, str]
    "canonical_to_postings": None,  # dict[str, list[str]]
}
_LOOKUP_LOCK = threading.Lock()
_SOURCE_PRIORITY = {"linkedin": 0, "indeed": 1, "glassdoor": 2, "adzuna": 3}


def _current_mtimes() -> tuple[float, float, float] | None:
    try:
        return (
            _POSTINGS_PATH.stat().st_mtime,
            _POSTING_SKILLS_PATH.stat().st_mtime,
            _CANONICAL_MAP_PATH.stat().st_mtime,
        )
    except FileNotFoundError:
        return None


def _reload_lookup() -> None:
    import pandas as pd

    canonical_map: dict[str, str] = json.loads(
        _CANONICAL_MAP_PATH.read_text(encoding="utf-8")
    )
    postings = pd.read_parquet(_POSTINGS_PATH)
    posting_skills = pd.read_parquet(_POSTING_SKILLS_PATH)

    meta_cols = [
        c
        for c in ("id", "title", "company", "location", "url", "source")
        if c in postings.columns
    ]
    meta_records = postings[meta_cols].to_dict(orient="records")
    posting_meta_by_id = {str(rec["id"]): rec for rec in meta_records}

    # Build canonical → posting_ids index. posting_skills.name is the RAW
    # name; normalise + map through canonical_map.
    from career_atlas.canonicalize import normalize_skill

    canonical_to_postings: dict[str, set[str]] = {}
    for row in posting_skills.itertuples(index=False):
        canonical = canonical_map.get(normalize_skill(row.name))
        if canonical is None:
            continue
        canonical_to_postings.setdefault(canonical, set()).add(str(row.posting_id))

    _LOOKUP_CACHE["mtimes"] = _current_mtimes()
    _LOOKUP_CACHE["posting_meta_by_id"] = posting_meta_by_id
    _LOOKUP_CACHE["raw_to_canonical"] = canonical_map
    _LOOKUP_CACHE["canonical_to_postings"] = {
        k: sorted(v) for k, v in canonical_to_postings.items()
    }


def lookup_postings_for_skill(
    canonical_name: str, limit: int = 50
) -> dict[str, Any]:
    """Return ``{"name": canonical_name, "total": N, "postings": [...]}`` for
    postings mentioning any raw name mapping to ``canonical_name``.

    Cached in-process; reloads on artefact mtime change so re-runs of
    skill_map propagate without restarting the server.
    """
    mtimes = _current_mtimes()
    if mtimes is None:
        return {"name": canonical_name, "total": 0, "postings": []}

    with _LOOKUP_LOCK:
        if _LOOKUP_CACHE["mtimes"] != mtimes:
            _reload_lookup()

    posting_ids = _LOOKUP_CACHE["canonical_to_postings"].get(canonical_name, [])
    total = len(posting_ids)

    posting_meta = _LOOKUP_CACHE["posting_meta_by_id"]
    results = []
    for pid in posting_ids:
        meta = posting_meta.get(pid)
        if meta is None:
            continue
        results.append(
            {
                "id": pid,
                "title": meta.get("title"),
                "company": meta.get("company"),
                "location": meta.get("location"),
                "url": meta.get("url"),
                "source": meta.get("source"),
            }
        )

    results.sort(
        key=lambda r: (
            _SOURCE_PRIORITY.get((r.get("source") or "").lower(), 99),
            (r.get("title") or "").lower(),
        )
    )
    return {"name": canonical_name, "total": total, "postings": results[:limit]}
