import asyncio
import json
from unittest.mock import AsyncMock

from aiohttp import web, ClientSession
from aiohttp.test_utils import TestClient, TestServer
import pytest

from comfy_client import ComfyClient, GenerationFailed, output_path
from contract import TTL_S
from self_destruct import IdleRetirement
from server import JobManager, make_app
from conftest import connector, encoded_image, payload


class FakeBackend:
    healthy = True

    def __init__(self):
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.fail = False
        self.running = 0
        self.maximum = 0
        self.order = []

    async def render(self, graph, directory, progress):
        self.order.append(directory.name)
        self.running += 1
        self.maximum = max(self.maximum, self.running)
        self.entered.set()
        progress(.5)
        try:
            await self.release.wait()
            if self.fail:
                self.fail = False
                raise RuntimeError("private prompt must never escape")
            file = directory / (directory.name + "_00001_.mp4")
            file.write_bytes(b"test-mp4-body")
            return file
        finally:
            self.running -= 1


@pytest.fixture
async def app_client(tmp_path):
    now = [0.0]
    retirement = IdleRetirement(lambda _: None, env={}, clock=lambda: now[0])
    backend = FakeBackend()
    manager = JobManager(backend, tmp_path / "jobs", retirement, clock=lambda: now[0])
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        yield http, manager, backend, now


async def start(http, **kwargs):
    response = await http.post("/clip/start", json=payload(**kwargs))
    assert response.status == 200
    data = await response.json()
    assert data["state"] == "queued"
    return data["id"]


async def wait_state(http, job_id, expected):
    for _ in range(100):
        response = await http.post("/clip/state", json={"id": job_id})
        data = await response.json()
        if data.get("state") == expected:
            return data
        await asyncio.sleep(.01)
    pytest.fail("job did not reach expected state")


async def test_queue_capacity_fifo_states_and_single_render(app_client):
    http, manager, backend, _ = app_client
    first = await start(http)
    await backend.entered.wait()
    second, third = await start(http), await start(http)
    response = await http.post("/clip/start", json=payload())
    assert response.status == 429 and await response.json() == {"error": "busy"}
    assert manager.retirement.jobs == 3
    assert (await wait_state(http, first, "running"))["progress"] == .5
    assert (await wait_state(http, second, "queued"))["progress"] == 0
    response = await http.post("/clip/file", json={"id": second})
    assert response.status == 409 and await response.json() == {"error": "not_ready"}
    backend.release.set()
    done = await wait_state(http, third, "done")
    assert done == {"state": "done", "progress": 1.0, "fps": 24, "width": 864,
                    "height": 480, "frames": 124, "exec_s": 0.0,
                    "lab": {"attention": "kitchen", "sparse": "off", "lora": "pika6",
                            "steps": 6, "shift": {"video": 12.0, "audio": 3.0}}}
    assert backend.order == [first, second, third] and backend.maximum == 1
    assert manager.retirement.jobs == 0


async def test_failure_is_sanitized_and_next_job_recovers(app_client, caplog):
    http, manager, backend, _ = app_client
    backend.fail = True
    first = await start(http, mode="frames", first=encoded_image())
    second = await start(http)
    backend.release.set()
    failed = await wait_state(http, first, "failed")
    assert failed["error"] == "generation failed"
    assert list(manager.jobs[first].directory.iterdir()) == []
    assert (await http.post("/clip/file", json={"id": first})).status == 409
    await wait_state(http, second, "done")
    assert "private prompt" not in caplog.text


async def test_file_whole_body_length_repeat_fetch_and_retention(app_client):
    http, manager, backend, _ = app_client
    backend.release.set()
    job_id = await start(http, mode="frames", first=encoded_image(), last=encoded_image("JPEG"))
    await wait_state(http, job_id, "done")
    directory = manager.jobs[job_id].directory
    assert len(list(directory.iterdir())) == 1
    response = await http.post("/clip/file", json={"id": job_id}, headers={"Range": "bytes=0-2"})
    assert response.status == 200 and response.headers["Content-Type"] == "video/mp4"
    assert int(response.headers["Content-Length"]) == len(b"test-mp4-body")
    assert await response.read() == b"test-mp4-body"
    assert directory.exists()
    for route in ("state", "file"):
        response = await http.post("/clip/" + route, json={"id": job_id})
        assert response.status == 200


async def test_expiry_exact_boundary_and_failed_records(app_client):
    http, manager, backend, now = app_client
    backend.release.set()
    job_id = await start(http)
    await wait_state(http, job_id, "done")
    directory = manager.jobs[job_id].directory
    now[0] = TTL_S - .001
    assert (await http.post("/clip/state", json={"id": job_id})).status == 200
    now[0] = TTL_S
    assert (await http.post("/clip/file", json={"id": job_id})).status == 404
    assert not directory.exists()
    backend.fail = True
    job_id = await start(http)
    await wait_state(http, job_id, "failed")
    now[0] += TTL_S
    manager.expire()
    assert job_id not in manager.jobs


async def test_running_and_queued_do_not_expire(app_client):
    http, manager, _, now = app_client
    first, second = await start(http), await start(http)
    now[0] = 10000
    manager.expire()
    assert first in manager.jobs and second in manager.jobs


async def test_health_does_not_extend_idle_and_retirement_refuses_start(app_client):
    http, manager, _, now = app_client
    now[0] = 100
    assert (await http.get("/health")).status == 200
    assert manager.retirement.last_activity == 0
    await http.post("/clip/state", json={"id": "a" * 32})
    assert manager.retirement.last_activity == 100 and manager.retirement.in_flight == 0
    manager.retirement.retiring = True
    assert (await http.get("/health")).status == 503
    assert (await http.post("/clip/start", json=payload())).status == 429


@pytest.mark.parametrize("route", ["start", "state", "file"])
async def test_bad_json_unknown_keys_and_ids(app_client, route):
    http, _, _, _ = app_client
    for data in (b"{", b"[]", b'{"id":"aaa","id":"bbb"}', b'{"id":null}', b'{"unknown":1}'):
        response = await http.post("/clip/" + route, data=data)
        assert response.status == 400
        assert (await response.json())["error"] == "invalid"


async def test_body_limit(app_client):
    http, _, _, _ = app_client
    # Lower the application's limit to avoid allocating 32 MiB in this check.
    http.server.app._client_max_size = 128
    response = await http.post("/clip/start", data=b"x" * 129)
    assert response.status == 400 and (await response.json())["error"] == "invalid"


class FakeComfy:
    def __init__(self, root, *, websocket=True, fail=False, hang=False):
        self.root, self.websocket, self.fail, self.hang = root, websocket, fail, hang
        self.graph = None
        self.ws = None
        self.polls = 0
        self.deleted = []
        self.interrupted = False
        self.prompt_id = "fake-prompt"

    def app(self):
        app = web.Application()

        async def ws(request):
            if not self.websocket:
                raise web.HTTPNotFound()
            self.ws = web.WebSocketResponse()
            await self.ws.prepare(request)
            async for _ in self.ws:
                pass
            return self.ws

        async def prompt(request):
            data = await request.json()
            self.graph = data["prompt"]
            assert data["client_id"]
            for node in self.graph.values():
                if node["class_type"] == "LoadImage":
                    assert (self.root / node["inputs"]["image"]).is_file()
            if self.ws is not None:
                await self.ws.send_json({"type": "progress", "data": {"prompt_id": self.prompt_id, "value": 4, "max": 8}})
            return web.json_response({"prompt_id": self.prompt_id, "node_errors": {}})

        async def history(request):
            self.polls += 1
            if self.hang or self.polls < 2:
                return web.json_response({})
            prefix = self.graph["20"]["inputs"]["filename_prefix"]
            path = self.root / (prefix + "_00001_.mp4")
            path.write_bytes(b"fake-comfy-video")
            return web.json_response({self.prompt_id: {"status": {"completed": True,
                "status_str": "error" if self.fail else "success"}, "outputs": {"20": {"images": [
                    {"filename": path.name, "subfolder": path.parent.name, "type": "output"}]}}}})

        async def delete(request):
            self.deleted.extend((await request.json())["delete"])
            return web.Response(status=200)

        async def interrupt(request):
            self.interrupted = True
            return web.Response(status=200)

        async def queue(request):
            return web.json_response({"queue_running": [], "queue_pending": []})

        async def object_info(request):
            # The default text/frames graph (pika6_kitchen since 2026-11-10) is checked for its nodes before /prompt.
            return web.json_response({"MiniMaxH3SigmaShift": {}, "BlockSparseAttention": {}, "ManualSigmas": {},
                "ModelAttentionBackend": {"input": {"required": {"attention": ["COMBO", {"options": ["comfy kitchen attention"]}]}}}})

        app.router.add_get("/ws", ws)
        app.router.add_get("/object_info", object_info)
        app.router.add_post("/prompt", prompt)
        app.router.add_get("/history/{id}", history)
        app.router.add_post("/history", delete)
        app.router.add_post("/interrupt", interrupt)
        app.router.add_get("/queue", queue)
        return app


@pytest.mark.parametrize("websocket,fail,hang", [(True, False, False), (False, False, False),
                                               (True, True, False), (False, False, True)])
async def test_whole_job_with_fake_comfy_http(tmp_path, websocket, fail, hang):
    fake = FakeComfy(tmp_path, websocket=websocket, fail=fail, hang=hang)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.01, timeout_s=.2,
                              clock=lambda: fake.polls * .05)
        retirement = IdleRetirement(lambda _: None, env={})
        manager = JobManager(backend, tmp_path, retirement)
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            job_id = await start(http, mode="frames", first=encoded_image(), last=encoded_image(), audio=False)
            await wait_state(http, job_id, "failed" if fail or hang else "done")
            assert fake.graph["6"]["inputs"]["last_frame"] == ["7", 0]
            assert "audio" not in fake.graph["19"]["inputs"]
            assert fake.deleted == [fake.prompt_id]
            assert fake.interrupted is hang
            assert backend.healthy
            if not fail and not hang:
                response = await http.post("/clip/file", json={"id": job_id})
                assert response.status == 200 and await response.read() == b"fake-comfy-video"
                assert (tmp_path / job_id).exists()


@pytest.mark.parametrize("folder,name,kind", [("../other", "a.mp4", "output"),
                                            ("a" * 32, "../a.mp4", "output"),
                                            ("a" * 32, "a.mp4", "input")])
def test_output_path_refuses_cross_job_or_traversal(tmp_path, folder, name, kind):
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    history = {"outputs": {"20": {"images": [{"filename": name, "subfolder": folder, "type": kind}]}}}
    with pytest.raises(GenerationFailed):
        output_path(history, directory)


async def test_interrupt_failure_fails_closed():
    client = ComfyClient(None)
    client.status_only = AsyncMock(side_effect=OSError("private"))
    await client._interrupt()
    assert not client.healthy


async def test_websocket_progress_filters_prompt_and_is_bounded():
    from aiohttp import WSMsgType
    from types import SimpleNamespace

    async def messages():
        for prompt_id, value in (("other", 4), ("ours", 4), ("ours", 100)):
            yield SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps({"type": "progress",
                "data": {"prompt_id": prompt_id, "value": value, "max": 8}}))

    values = []
    await ComfyClient(None)._progress(messages(), "ours", values.append)
    assert values == [.5, .94]


async def test_sequential_clips_delete_history_without_freeing_models(tmp_path):
    from boot import comfy_command
    from pathlib import Path

    args = comfy_command(tmp_path, Path("python"), tmp_path)
    assert "--cache-none" not in args and "--disable-metadata" in args
    fake = FakeComfy(tmp_path, websocket=False)
    app = fake.app()
    freed = []
    async def free(request):
        freed.append(await request.json())
        return web.json_response({})
    app.router.add_post("/free", free)
    async with TestServer(app) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.001)
        manager = JobManager(backend, tmp_path, IdleRetirement(lambda _: None, env={}))
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            for prompt in ("First clip", "Second clip"):
                job_id = await start(http, prompt=prompt)
                await wait_state(http, job_id, "done")
            assert fake.deleted == [fake.prompt_id, fake.prompt_id]
            assert freed == []
