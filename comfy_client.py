"""Small ComfyUI HTTP/WS adapter; never log graphs, filenames or exceptions."""

import asyncio
from contextlib import suppress
import json
from pathlib import Path
import time
import uuid

import aiohttp


class GenerationFailed(RuntimeError):
    pass


class LabUnavailable(GenerationFailed):
    """Only fixed capability messages may be exposed to the caller."""


def combo_options(spec):
    """Options of a combo input in /object_info: V1 nodes send [[...options]], V3 nodes (ComfyUI 0.44
    io.Combo, e.g. ModelAttentionBackend) send ["COMBO", {"options": [...]}]."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return []
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options", [])
        return list(options) if isinstance(options, (list, tuple)) else []
    return list(spec[0]) if isinstance(spec[0], (list, tuple)) else []


def output_path(history, job_dir):
    # Only SaveVideo's output, inside this job, is eligible for transfer.
    outputs = history.get("outputs", {}).get("20", {})
    for key in ("images", "videos", "gifs"):
        for entry in outputs.get(key, []):
            if entry.get("type") != "output":
                continue
            name, folder = entry.get("filename", ""), entry.get("subfolder", "")
            if (not isinstance(name, str) or not isinstance(folder, str)
                    or Path(name).name != name or "/" in name or "\\" in name
                    or folder != job_dir.name or not name.endswith(".mp4")):
                raise GenerationFailed("invalid output")
            candidate = job_dir / name
            if candidate.is_symlink() or candidate.resolve().parent != job_dir.resolve():
                raise GenerationFailed("invalid output")
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    raise GenerationFailed("missing output")


class ComfyClient:
    def __init__(self, session, url="http://127.0.0.1:8188", *, poll_s=1, timeout_s=2700, clock=time.monotonic):
        self.session, self.url = session, url.rstrip("/")
        self.poll_s, self.timeout_s = poll_s, timeout_s
        self.clock = clock
        self.healthy = True
        self.sampler_s = None

    async def json(self, method, path, **kwargs):
        async with self.session.request(method, self.url + path, **kwargs) as response:
            if response.status != 200:
                raise GenerationFailed("backend request failed")
            return await response.json()

    async def status_only(self, method, path, **kwargs):
        # ComfyUI returns an empty 200 without a JSON Content-Type here.
        async with self.session.request(method, self.url + path, **kwargs) as response:
            if response.status != 200:
                raise GenerationFailed("backend request failed")

    async def _progress(self, ws, prompt_id, update):
        sampler_started = None
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue
                if event.get("type") == "executing":
                    if data.get("node") == "16":
                        sampler_started = self.clock()
                    elif sampler_started is not None:
                        self.sampler_s = max(0.0, self.clock() - sampler_started)
                        sampler_started = None
                if event.get("type") == "progress":
                    value, maximum = data.get("value"), data.get("max")
                    if isinstance(value, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
                        update(min(.94, .1 + .8 * max(0, value) / maximum))
                elif event.get("type") == "executing" and data.get("node") in ("17", "18", "19", "20"):
                    update(.95)
            except (ValueError, TypeError, AttributeError):
                continue

    async def _interrupt(self):
        try:
            await self.status_only("POST", "/interrupt", json={})
            for _ in range(30):
                queue = await self.json("GET", "/queue")
                if not queue.get("queue_running") and not queue.get("queue_pending"):
                    return
                await asyncio.sleep(1)
        except Exception:
            pass
        self.healthy = False

    async def check_lab(self, graph):
        kinds = {node["class_type"] for node in graph.values()}
        optional = kinds & {"ModelAttentionBackend", "BlockSparseAttention", "MiniMaxH3SigmaShift", "ManualSigmas"}
        if any(kind.startswith("OpenH3") for kind in kinds):
            optional = kinds | {"MinimaxH3LatentUpscaler3D"} if "OpenH3RegenUpscale" in kinds else kinds
        if "ReCaVSRUpscale" in kinds:
            optional = kinds
        if not optional:
            return
        info = await self.json("GET", "/object_info")
        for kind in sorted(optional):
            if kind not in info:
                raise LabUnavailable("lab requires ComfyUI node " + kind)
        if "ModelAttentionBackend" in optional:
            inputs = info["ModelAttentionBackend"].get("input", {})
            spec = {**inputs.get("optional", {}), **inputs.get("required", {})}.get("attention", [])
            if "comfy kitchen attention" not in combo_options(spec):
                raise LabUnavailable("lab kitchen requires the comfy kitchen INT8 attention backend")

    async def render(self, graph, job_dir, update):
        self.sampler_s = None
        if not self.healthy:
            raise GenerationFailed("backend unavailable")
        await self.check_lab(graph)
        prompt_id, ws, progress_task = None, None, None
        settled = False
        submitted = False
        try:
            client_id = uuid.uuid4().hex
            try:
                ws = await self.session.ws_connect(self.url + "/ws", params={"clientId": client_id},
                                                   heartbeat=30)
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                pass  # History still supplies completion and coarse progress.
            # A lost POST response can still leave work running: interrupt on failure.
            submitted = True
            result = await self.json("POST", "/prompt", json={"prompt": graph, "client_id": client_id})
            prompt_id = result.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id or result.get("node_errors"):
                raise GenerationFailed("graph rejected")
            if ws is not None:
                progress_task = asyncio.create_task(self._progress(ws, prompt_id, update))
            deadline = self.clock() + self.timeout_s
            update(.1)
            while self.clock() < deadline:
                history = (await self.json("GET", "/history/" + prompt_id)).get(prompt_id)
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        settled = True
                        raise GenerationFailed("generation failed")
                    if status.get("completed"):
                        settled = True
                        return output_path(history, Path(job_dir))
                await asyncio.sleep(self.poll_s)
            raise GenerationFailed("generation timed out")
        except asyncio.CancelledError:
            raise
        except Exception:
            raise GenerationFailed("generation failed") from None
        finally:
            if progress_task:
                progress_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await progress_task
            if ws is not None:
                await ws.close()
            if submitted and not settled:
                await self._interrupt()
            if prompt_id:
                with suppress(Exception):
                    await self.status_only("POST", "/history", json={"delete": [prompt_id]})
