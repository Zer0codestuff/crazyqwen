"""Local web server: one worker thread owns the model, the browser polls job state."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import PngImagePlugin
from pydantic import BaseModel, Field

from crazyqwen import models
from crazyqwen.engine import PROJECT_ROOT, Cancelled, Engine, GenerationRequest, Progress

STATIC = Path(__file__).resolve().parent / "static"
OUTPUTS = Path(os.environ.get("CRAZYQWEN_OUTPUTS", PROJECT_ROOT / "outputs"))
IDLE_UNLOAD_SECONDS = int(os.environ.get("CRAZYQWEN_IDLE_UNLOAD", "600"))


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    mode: str = "fast"
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    steps: int | None = None
    guidance: float = 1.0
    negative_prompt: str = ""


class Worker:
    def __init__(self):
        self.engine = Engine()
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.unload_requested = False
        threading.Thread(target=self._run, daemon=True, name="crazyqwen-worker").start()

    def submit(self, body: GenerateBody) -> dict:
        request = GenerationRequest(**body.model_dump()).normalized()
        job = {
            "id": uuid.uuid4().hex[:12],
            "status": "queued",
            "request": asdict(request),
            "progress": {"phase": "queued", "step": 0, "total": request.steps},
            "created": time.time(),
            "started": None,
            "image": None,
            "error": None,
            "cancel": False,
        }
        with self.lock:
            self.jobs[job["id"]] = job
        self.queue.put(job["id"])
        return self.public(job)

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job["cancel"] = True
            if job["status"] == "queued":
                job["status"] = "cancelled"
            return self.public(job)

    @staticmethod
    def public(job: dict) -> dict:
        return {k: v for k, v in job.items() if k != "cancel"}

    def _run(self) -> None:
        while True:
            try:
                job_id = self.queue.get(timeout=15)
            except queue.Empty:
                idle = time.time() - self.engine.last_used
                if self.unload_requested or (self.engine.loaded and idle > IDLE_UNLOAD_SECONDS):
                    self.engine.unload()
                    self.unload_requested = False
                continue
            job = self.jobs.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            self._process(job)
            if self.unload_requested and self.queue.empty():
                self.engine.unload()
                self.unload_requested = False

    def _process(self, job: dict) -> None:
        job["status"] = "running"
        job["started"] = time.time()

        def on_progress(p: Progress) -> None:
            job["progress"] = {"phase": p.phase, "step": p.step, "total": p.total, **p.extra}

        try:
            image, meta = self.engine.generate(
                GenerationRequest(**job["request"]), on_progress, cancelled=lambda: job["cancel"]
            )
            job["image"] = save_image(image, meta)
            job["status"] = "done"
        except Cancelled:
            job["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            traceback.print_exc()
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"


def save_image(image, meta: dict) -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + f"-{meta['seed']}"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", json.dumps(meta))
    image.save(OUTPUTS / f"{stem}.png", pnginfo=info)
    meta = {**meta, "id": stem, "file": f"{stem}.png", "created": time.time()}
    (OUTPUTS / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta


def list_images() -> list[dict]:
    if not OUTPUTS.exists():
        return []
    items = []
    for path in sorted(OUTPUTS.glob("*.json"), reverse=True):
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (OUTPUTS / meta.get("file", "")).exists():
            items.append(meta)
    return items


def create_app() -> FastAPI:
    app = FastAPI(title="CrazyQwen", docs_url=None, redoc_url=None)
    worker = Worker()

    @app.middleware("http")
    async def no_stale_ui(request, call_next):
        # Always revalidate the UI files so an update shows up on a plain reload.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    @app.get("/api/status")
    def status():
        import mlx.core as mx

        with worker.lock:
            active = [worker.public(j) for j in worker.jobs.values() if j["status"] in ("queued", "running")]
        return {
            "missing": models.missing(),
            "loaded": worker.engine.loaded,
            "jobs": active,
            "memory_gb": round(mx.get_active_memory() / 1e9, 2),
        }

    @app.post("/api/generate")
    def generate(body: GenerateBody):
        if (reason := models.missing()) is not None:
            raise HTTPException(409, f"Models are not downloaded yet: {reason}. Run ./run.sh to download them.")
        try:
            return worker.submit(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        with worker.lock:
            found = worker.jobs.get(job_id)
            if found is None:
                raise HTTPException(404, "Job not found")
            return worker.public(found)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return worker.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.post("/api/unload")
    def unload():
        worker.unload_requested = True
        return {"ok": True}

    @app.get("/api/images")
    def images():
        return list_images()

    @app.delete("/api/images/{image_id}")
    def delete_image(image_id: str):
        if "/" in image_id or ".." in image_id:
            raise HTTPException(400, "Bad id")
        removed = False
        for suffix in (".png", ".json"):
            path = OUTPUTS / f"{image_id}{suffix}"
            if path.exists():
                path.unlink()
                removed = True
        if not removed:
            raise HTTPException(404, "Image not found")
        return {"ok": True}

    app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
