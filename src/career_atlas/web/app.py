"""FastAPI app that drives the CAREER-ATLAS pipelines from a browser."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from career_atlas.web import runner

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CAREER-ATLAS", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Single in-flight build at a time.
_JOBS: dict[str, runner.BuildJob] = {}
_JOB_LOCK = asyncio.Lock()


class CVBody(BaseModel):
    cv_text: str = Field(..., min_length=1)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return runner.status_flags()


@app.post("/api/cv")
async def post_cv(body: CVBody) -> dict[str, Any]:
    runner.write_cv_text(body.cv_text)
    try:
        await asyncio.to_thread(runner.run_cv_extraction)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        logger.exception("cv_extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    paths = runner.paths()
    profile = json.loads(paths["cv_profile"].read_text(encoding="utf-8"))
    targeting = json.loads(paths["targeting"].read_text(encoding="utf-8"))
    return {"profile": profile, "targeting": targeting}


@app.post("/api/build")
async def post_build() -> dict[str, str]:
    async with _JOB_LOCK:
        for job in _JOBS.values():
            if job.status == "running":
                raise HTTPException(
                    status_code=409, detail=f"build {job.job_id} already running"
                )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        job = runner.BuildJob(queue=queue, loop=loop)
        _JOBS[job.job_id] = job
        job.start()
        return {"job_id": job.job_id}


@app.get("/api/build/events")
async def build_events(job_id: str) -> StreamingResponse:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")

    async def event_stream():
        while True:
            event = await job.queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/map")
async def get_map() -> JSONResponse:
    path = runner.paths()["map"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="skill_map.json not yet built")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/skill/{name}/postings")
async def skill_postings(name: str, limit: int = 50) -> dict[str, Any]:
    return runner.lookup_postings_for_skill(name, limit=limit)


def run() -> None:
    """Console-script entry: ``career-atlas-ui`` launches uvicorn."""
    import uvicorn

    uvicorn.run(
        "career_atlas.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
