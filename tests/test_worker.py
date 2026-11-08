import io
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
import pytest
from vastai.serverless.server.worker import EndpointHandlerFactory

import boot
from graph import REQUIRED_NODES
from self_destruct import IdleRetirement
from server import JobManager, make_app
import worker
from conftest import connector


def test_sdk_routes_costs_and_benchmark_configuration():
    config = worker.build_config("a" * 32)
    assert config.model_server_url == "http://127.0.0.1" and config.model_server_port == 18001
    assert config.model_healthcheck_url == "/health"
    assert config.log_action_config.on_load == ["COMFY_READY"]
    assert config.log_action_config.on_error == ["COMFY_FATAL"]
    assert [h.route for h in config.handlers] == ["/clip/start", "/clip/state", "/clip/file"]
    assert [h.workload_calculator({}) for h in config.handlers] == [100, 1, 1]
    assert all(h.allow_parallel_requests for h in config.handlers)
    assert sum(h.benchmark_config is not None for h in config.handlers) == 1
    benchmark = EndpointHandlerFactory(config).get_benchmark_handler()
    assert benchmark.endpoint == "/clip/state"
    assert benchmark.make_benchmark_payload().generate_payload_json() == {"id": "a" * 32}
    assert benchmark.benchmark_runs == 1 and benchmark.concurrency == 1
    assert not benchmark.do_warmup_benchmark


async def test_real_sdk_benchmark_payload_never_submits_render(tmp_path):
    class Backend:
        healthy = True

        async def render(self, *args):
            pytest.fail("benchmark rendered")

    retirement = IdleRetirement(lambda _: None, env={})
    manager = JobManager(Backend(), tmp_path, retirement, benchmark_id="a" * 32)
    handler = EndpointHandlerFactory(worker.build_config("a" * 32)).get_benchmark_handler()
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        response = await http.post(handler.endpoint, json=handler.make_benchmark_payload().generate_payload_json())
        assert response.status == 200 and (await response.json())["state"] == "failed"
        assert manager.active == 0 and manager.queue.empty() and retirement.jobs == 0


async def test_real_sdk_mp4_passthrough_preserves_length_and_status():
    body = b"fake-mp4" * 100

    async def file(request):
        return web.Response(body=body, content_type="video/mp4")

    app = web.Application()
    app.router.add_post("/clip/file", file)
    handler = EndpointHandlerFactory(worker.build_config("a" * 32)).get_handler("/clip/file")
    async with TestServer(app) as backend, ClientSession(connector=connector()) as session:
        async def proxy(request):
            async with session.post(backend.make_url("/clip/file"), json={}) as response:
                return await handler.generate_client_response(request, response)
        proxy_app = web.Application()
        proxy_app.router.add_post("/clip/file", worker.bounded_route(proxy))
        async with TestClient(TestServer(proxy_app), connector=connector()) as http:
            response = await http.post("/clip/file", json={})
            assert response.status == 200 and response.headers["Content-Type"] == "video/mp4"
            assert int(response.headers["Content-Length"]) == len(body)
            assert await response.read() == body


async def test_public_sdk_envelope_has_a_body_limit(monkeypatch):
    monkeypatch.setattr(worker, "MAX_BODY_BYTES", 1)

    async def consume(request):
        await request.read()
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=0)
    app.router.add_post("/clip/start", worker.bounded_route(consume))
    async with TestClient(TestServer(app), connector=connector()) as http:
        response = await http.post("/clip/start", data=b"x" * 65538)
        assert response.status == 400 and (await response.json())["error"] == "invalid"


def test_debug_disabled_after_sdk_construction(monkeypatch):
    class Proxy:
        def __init__(self, config):
            logging.getLogger().setLevel(logging.DEBUG)
            self.routes = []
            self.backend = SimpleNamespace()

        def run(self):
            assert not logging.getLogger("payload").isEnabledFor(logging.DEBUG)

    old_level, old_disable = logging.getLogger().level, logging.root.manager.disable
    monkeypatch.setattr(worker, "Worker", Proxy)
    monkeypatch.setenv("COMFY_BENCHMARK_ID", "a" * 32)
    try:
        worker.run_proxy()
    finally:
        logging.disable(old_disable)
        logging.getLogger().setLevel(old_level)


def test_readiness_requires_all_native_nodes_and_process_alive():
    now = [0]
    calls = []

    class Response(io.BytesIO):
        status = 200

    def opener(url, timeout):
        calls.append(url)
        return Response(json.dumps({n: {} for n in (REQUIRED_NODES if len(calls) > 1 else [])}).encode())

    boot.wait_ready("unused", [SimpleNamespace(poll=lambda: None)], nodes=True, timeout=10,
                    clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n), opener=opener)
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="exited"):
        boot.wait_ready("unused", [SimpleNamespace(poll=lambda: 1)])


def test_readiness_timeout_with_missing_nodes():
    now = [0]

    class Response(io.BytesIO):
        status = 200

    with pytest.raises(RuntimeError, match="timeout"):
        boot.wait_ready("unused", [], nodes=True, timeout=2, clock=lambda: now[0],
                        sleep=lambda n: now.__setitem__(0, now[0] + n), opener=lambda *a, **k: Response(b"{}"))


def test_comfy_command_all_state_private_and_no_public_bind(tmp_path):
    args = boot.comfy_command(tmp_path / "ComfyUI", Path("python"), tmp_path / "private")
    for flag, suffix in (("--input-directory", "jobs"), ("--output-directory", "jobs"),
                         ("--temp-directory", "temp"), ("--user-directory", "user")):
        assert args[args.index(flag) + 1] == str(tmp_path / "private" / suffix)
    assert args[args.index("--listen") + 1] == "127.0.0.1"
    assert "--cache-none" not in args
    assert "--disable-metadata" in args
    assert "--enable-triton-backend" not in args
