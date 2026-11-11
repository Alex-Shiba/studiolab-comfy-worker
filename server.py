"""Loopback clip API. All public errors and logs are fixed, payload-free strings."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import io
import os
from pathlib import Path
import shutil
import time
import uuid

import aiohttp
from aiohttp import web

from comfy_client import ComfyClient, LabUnavailable
from contract import FPS, ID_PATTERN, Invalid, MAX_BODY_BYTES, TTL_S, save_frame, strict_json, validate_id, validate_start
from graph import build_graph
from reference_models import ReferenceModels, LabModels, RegenModels
from regen import enabled as regen_enabled, finish_video
from lab import validate_lab
from self_destruct import IdleRetirement


def log_marker(marker):
    path = Path(os.environ.get("COMFY_LOG_FILE", "/var/log/portal/comfy-worker.log"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(marker + "\n")
    except OSError:
        # A full/broken log disk must not prevent fatal instance retirement.
        print(marker, flush=True)


@dataclass
class Job:
    id: str
    directory: Path
    state: str = "queued"
    progress: float = 0.0
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    exec_s: float | None = None
    sampler_s: float | None = None
    lab: dict | None = None
    regen: dict | None = None
    draft: Path | None = None
    duration: float | None = None
    error: str | None = None
    finished_at: float | None = None
    file: Path | None = None
    stage: str | None = None
    fetching: bool = False
    request: str | None = None

    def public(self):
        result = {"state": self.state, "progress": self.progress, "fps": FPS}
        for key in ("width", "height", "frames", "exec_s", "sampler_s", "lab", "regen", "error", "stage"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


class JobManager:
    def __init__(self, backend, root, retirement, *, clock=time.monotonic, benchmark_id=None, reference_models=None, lab_models=None, regen_models=None):
        self.backend, self.root, self.retirement, self.clock = backend, Path(root), retirement, clock
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.reference_models = reference_models
        self.lab_models = lab_models
        self.regen_models = regen_models
        self.preparing = set()
        self.jobs = {}
        self.requests = {}
        self.queue = asyncio.Queue()
        self.active = 0
        if benchmark_id:
            if not ID_PATTERN.fullmatch(benchmark_id):
                raise ValueError("invalid benchmark id")
            # A valid failed-job record tests state lookup without submitting work.
            self.jobs[benchmark_id] = Job(benchmark_id, self.root / benchmark_id, state="failed",
                                          error="readiness probe", finished_at=self.clock())

    def lookup(self, clip):
        self.expire()
        fingerprint = clip.fingerprint()
        if clip.request in self.requests:
            previous, job_id = self.requests[clip.request]
            if previous != fingerprint:
                raise web.HTTPConflict(text='{"error":"conflict"}', content_type="application/json")
            return self.jobs[job_id]
        return None

    def available(self):
        return self.active < 3 and not self.retirement.retiring and self.backend.healthy

    def submit(self, clip, frames):
        # No await between this second lookup, capacity check and registration.
        existing = self.lookup(clip)
        if existing is not None:
            return existing
        if not self.available():
            return None
        job_id = uuid.uuid4().hex
        directory = self.root / job_id
        directory.mkdir(mode=0o700)
        width, height = clip.dimensions
        job = Job(job_id, directory, width=width, height=height, frames=clip.frames, request=clip.request,
                  lab=None if clip.mode == "regen" else (clip.lab or validate_lab({}, clip.mode)).public(),
                  regen=clip.regen.public() if clip.mode == "regen" else None)
        try:
            names = {}
            if clip.mode == "regen":
                name = uuid.uuid4().hex + ".mp4"
                job.draft = directory / name
                job.draft.write_bytes(clip.video.data)
                job.duration = clip.video.duration
                names["video"] = job_id + "/" + name
            for key, data in frames.items():
                name = uuid.uuid4().hex + ".png"
                (directory / name).write_bytes(data)
                names[key] = job_id + "/" + name
            refs = []
            for image in clip.refs:
                name = uuid.uuid4().hex + image.extension
                (directory / name).write_bytes(image.data)
                refs.append(job_id + "/" + name)
            graph = build_graph(clip, job_id, **names, refs=refs)
        except BaseException:
            shutil.rmtree(directory)
            raise
        self.jobs[job_id] = job
        self.requests[clip.request] = (clip.fingerprint(), job_id)
        self.active += 1
        self.retirement.job_started()
        if clip.mode == "regen" or clip.mode == "refs" or job.lab["lora"] in ("turbo8_768", "turbo8_544", "turbo4_768"):
            task = asyncio.create_task(self.prepare_regen(job, graph) if clip.mode == "regen"
                                       else self.prepare_references(job, graph, clip.mode == "refs"))
            self.preparing.add(task)
            task.add_done_callback(self.preparing.discard)
        else:
            self.queue.put_nowait((job, graph))
        return job

    async def prepare_regen(self, job, graph):
        job.stage = "downloading_regen_models"
        try:
            if self.regen_models is None or self.reference_models is None:
                raise RuntimeError("regen models unavailable")
            await self.regen_models.wait(lambda value: setattr(job, "progress", value * .6))
            # Both recipes need Ref2VA: IC's base and refine's Sharp curve table.
            await self.reference_models.wait(lambda value: setattr(job, "progress", .6 + value * .4))
            job.stage, job.progress = None, 0.0
            self.queue.put_nowait((job, graph))
        except asyncio.CancelledError:
            raise
        except Exception:
            job.error = "regen model download failed; retry the job"
            job.state, job.stage = "failed", None
            job.finished_at = self.clock()
            self.active -= 1
            self.retirement.job_finished()
            for path in job.directory.iterdir():
                path.unlink()
            graph.clear()

    async def prepare_references(self, job, graph, references=True):
        job.stage = "downloading_reference_model" if references else "downloading_lab_lora"
        try:
            if references:
                if self.reference_models is None:
                    raise RuntimeError("reference models unavailable")
                await self.reference_models.wait(lambda value: setattr(job, "progress", value))
            if job.lab["lora"] in ("ref_v01", "turbo8_768", "turbo8_544", "turbo4_768"):
                job.stage, job.progress = "downloading_lab_lora", 0.0
                if self.lab_models is None:
                    raise RuntimeError("lab models unavailable")
                await self.lab_models.wait(job.lab["lora"], lambda value: setattr(job, "progress", value))
            job.stage, job.progress = None, 0.0
            self.queue.put_nowait((job, graph))
        except asyncio.CancelledError:
            raise
        except Exception:
            job.error = ("lab LoRA download failed; retry the job" if job.stage == "downloading_lab_lora"
                         else "reference model download failed; retry the references job")
            job.state, job.stage = "failed", None
            job.finished_at = self.clock()
            self.active -= 1
            self.retirement.job_finished()
            for path in job.directory.iterdir():
                path.unlink()
            graph.clear()

    def remove(self, job):
        # Direct children generated by us only; never accept a client path.
        if job.directory.parent != self.root or not ID_PATTERN.fullmatch(job.directory.name):
            raise RuntimeError("invalid job directory")
        if job.directory.exists():
            shutil.rmtree(job.directory)
        self.jobs.pop(job.id, None)
        self.requests.pop(job.request, None)

    def expire(self):
        for job in list(self.jobs.values()):
            if not job.fetching and job.finished_at is not None and self.clock() - job.finished_at >= TTL_S:
                self.remove(job)

    async def run(self):
        while True:
            job, graph = await self.queue.get()
            started = self.clock()
            job.state, job.progress = "running", .05

            def progress(value):
                job.progress = max(job.progress, min(.99, value))

            try:
                job.file = await self.backend.render(graph, job.directory, progress)
                if job.regen:
                    job.file = await asyncio.to_thread(finish_video, job.file, job.draft, job.duration)
                job.state, job.progress = "done", 1.0
            except asyncio.CancelledError:
                raise
            except LabUnavailable as error:
                job.state, job.error = "failed", str(error)
            except Exception:
                job.state, job.error = "failed", "generation failed"
                job.file = None
            finally:
                job.sampler_s = getattr(self.backend, "sampler_s", None)
                # ComfyUI no longer needs keyframes or partial/failed outputs.
                for path in job.directory.iterdir():
                    if path != job.file and path.is_file():
                        path.unlink()
                job.exec_s = max(0.0, self.clock() - started)
                job.finished_at = self.clock()
                self.active -= 1
                self.retirement.job_finished()
                self.queue.task_done()
                graph.clear()

    async def sweep(self):
        while True:
            self.expire()
            await asyncio.sleep(15)


def make_app(manager):
    retirement = manager.retirement

    @web.middleware
    async def activity(request, handler):
        # SDK health probes cannot keep a GPU alive indefinitely.
        tracked = request.path != "/health"
        if tracked:
            retirement.request_started()
        try:
            return await handler(request)
        except Invalid as error:
            return web.json_response({"error": "invalid", "detail": str(error)}, status=400)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "invalid", "detail": "request exceeds size limit"}, status=400)
        except web.HTTPException:
            raise
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception:
            return web.json_response({"error": "invalid", "detail": "request failed"}, status=400)
        finally:
            if tracked:
                retirement.request_finished()

    async def lifecycle(app):
        if manager.reference_models:
            manager.reference_models.start()
        if manager.regen_models:
            manager.regen_models.start()
        tasks = [asyncio.create_task(manager.run()), asyncio.create_task(manager.sweep()),
                 asyncio.create_task(retirement.run())]
        try:
            yield
        finally:
            tasks.extend(manager.preparing)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            if manager.reference_models:
                await manager.reference_models.close()
            if manager.lab_models:
                await manager.lab_models.close()
            if manager.regen_models:
                await manager.regen_models.close()
            for job in list(manager.jobs.values()):
                manager.remove(job)

    async def payload(request):
        try:
            return strict_json(await request.text())
        except UnicodeError:
            raise Invalid("invalid JSON") from None

    # Bound CPU image decoding independently of the generation queue.
    decode_slots = asyncio.Semaphore(2)

    def crop_frames(clip):
        frames = {}
        for key in ("first", "last"):
            image = getattr(clip, key)
            if image is not None:
                output = io.BytesIO()
                save_frame(image, output, clip.dimensions)
                frames[key] = output.getvalue()
        return frames

    async def start(request):
        async with decode_slots:
            clip = await asyncio.to_thread(validate_start, await payload(request))
            job = manager.lookup(clip)
            if job is None and manager.available():
                frames = await asyncio.to_thread(crop_frames, clip)
                job = manager.submit(clip, frames)
        if job is None:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"id": job.id, "state": job.state})

    async def find(request):
        job_id = validate_id(await payload(request))
        manager.expire()
        return manager.jobs.get(job_id)

    async def state(request):
        job = await find(request)
        return web.json_response(job.public() if job else {"error": "unknown"}, status=200 if job else 404)

    async def file(request):
        job = await find(request)
        if job is None or job.fetching:
            return web.json_response({"error": "unknown"}, status=404)
        if job.state != "done":
            return web.json_response({"error": "not_ready"}, status=409)
        if job.file is None or not job.file.is_file():
            manager.remove(job)
            return web.json_response({"error": "unknown"}, status=404)
        job.fetching = True
        try:
            response = web.StreamResponse(headers={"Content-Type": "video/mp4",
                                                   "Content-Length": str(job.file.stat().st_size)})
            await response.prepare(request)
            with job.file.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            job.fetching = False

    async def health(request):
        healthy = manager.backend.healthy and not retirement.retiring
        return web.json_response({"status": "ready" if healthy else "unavailable"}, status=200 if healthy else 503)

    app = web.Application(client_max_size=MAX_BODY_BYTES, middlewares=[activity])
    app.cleanup_ctx.append(lifecycle)
    app.router.add_post("/clip/start", start)
    app.router.add_post("/clip/state", state)
    app.router.add_post("/clip/file", file)
    app.router.add_get("/health", health)
    return app


async def serve():
    runner = None
    try:
        async def request_retirement():
            # The parent alone holds the instance deletion credential.
            Path(os.environ["COMFY_RETIRE_FILE"]).touch(mode=0o600)
            await asyncio.Event().wait()

        retirement = IdleRetirement(log_marker, request_retirement=request_retirement)
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            backend = ComfyClient(session)
            manager = JobManager(backend, os.environ["COMFY_JOB_ROOT"], retirement,
                                 benchmark_id=os.environ.get("COMFY_BENCHMARK_ID"),
                                 reference_models=ReferenceModels(os.environ["COMFY_MODEL_ROOT"],
                                                                  os.environ["COMFY_READY_FILE"]),
                                 lab_models=LabModels(os.environ["COMFY_MODEL_ROOT"], os.environ["COMFY_READY_FILE"]),
                                 regen_models=RegenModels(os.environ["COMFY_MODEL_ROOT"], os.environ["COMFY_READY_FILE"])
                                 if regen_enabled() else None)
            runner = web.AppRunner(make_app(manager), access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 18001).start()
            try:
                await asyncio.Event().wait()
            finally:
                await runner.cleanup()
                runner = None
    except Exception as error:
        log_marker(f"COMFY_FATAL server {type(error).__name__}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    asyncio.run(serve())
