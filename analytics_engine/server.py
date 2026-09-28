"""FastAPI app: REST + Server-Sent Events API and the static chat front-end."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import run_turn
from .config import settings
from .session import SUPPORTED_EXTENSIONS, Session, SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("analytics_engine")

STATIC_DIR = Path(__file__).parent / "static"
manager = SessionManager(settings.workspace)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.has_credentials:
        log.warning("No Claude credentials found. Set ANTHROPIC_API_KEY in .env (see .env.example).")
    log.info("Model: %s (effort=%s). Workspace: %s", settings.model, settings.effort, settings.workspace)
    yield
    await manager.shutdown()


app = FastAPI(title="AI Analytics Engine", lifespan=lifespan)


def _session(sid: str) -> Session:
    session = manager.get(sid)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


class ChatRequest(BaseModel):
    message: str


@app.get("/api/config")
async def get_config():
    return {"model": settings.model, "effort": settings.effort, "has_credentials": settings.has_credentials,
            "max_upload_mb": settings.max_upload_mb, "extensions": sorted(SUPPORTED_EXTENSIONS)}


@app.post("/api/sessions")
async def create_session(previous: str | None = None):
    # Starting a new analysis closes the previous session's kernel (its files stay on disk).
    if previous and (old := manager.get(previous)):
        manager.sessions.pop(previous, None)
        if old.busy:
            old.cancel.set()
            old.task.cancel()
        await old.kernel.shutdown()
    session = await manager.create()
    return session.state()


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    return _session(sid).state()


@app.get("/api/sessions/{sid}/stream")
async def stream_session(sid: str):
    session = _session(sid)

    async def events():
        queue, snapshot = session.subscribe()
        try:
            yield _sse({"type": "reset", **session.state(), "events": snapshot})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(event)
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@app.post("/api/sessions/{sid}/chat", status_code=202)
async def chat(sid: str, body: ChatRequest):
    session = _session(sid)
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Message is empty")
    if session.busy:
        raise HTTPException(409, "An analysis is already running in this session")
    session.task = asyncio.create_task(run_turn(session, message))
    return {"ok": True}


@app.post("/api/sessions/{sid}/stop")
async def stop(sid: str):
    session = _session(sid)
    if session.busy:
        session.cancel.set()
        await session.kernel.interrupt()
    return {"ok": True}


@app.post("/api/sessions/{sid}/upload")
async def upload(sid: str, files: list[UploadFile]):
    session = _session(sid)
    if session.busy:
        raise HTTPException(409, "Wait for the current analysis to finish before uploading")
    added, errors = [], []
    for f in files:
        name = Path(f.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(f"{name}: unsupported file type (use {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
            continue
        dest = _unique_path(session.workdir / "data", name)
        size, limit = 0, settings.max_upload_mb * 1024 * 1024
        with dest.open("wb") as out:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    break
                out.write(chunk)
        if size > limit:
            dest.unlink(missing_ok=True)
            errors.append(f"{name}: larger than {settings.max_upload_mb} MB")
            continue
        try:
            added.append(await session.add_dataset(name, dest))
        except Exception as e:  # noqa: BLE001 - parsing errors are reported to the user
            dest.unlink(missing_ok=True)
            errors.append(f"{name}: could not be parsed ({e})")
    for err in errors:
        session.emit({"type": "error", "message": f"Upload failed - {err}"})
    return {"added": added, "errors": errors}


def _unique_path(folder: Path, name: str) -> Path:
    name = re.sub(r"[^\w.\- ]+", "_", name).strip() or "upload"
    path, stem, suffix, n = folder / name, Path(name).stem, Path(name).suffix, 2
    while path.exists():
        path, n = folder / f"{stem}_{n}{suffix}", n + 1
    return path


@app.get("/api/sessions/{sid}/files/{rel_path:path}")
async def get_file(sid: str, rel_path: str):
    workdir = (settings.workspace / sid).resolve()
    target = (workdir / rel_path).resolve()
    if not target.is_relative_to(workdir) or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run() -> None:
    uvicorn.run("analytics_engine.server:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
