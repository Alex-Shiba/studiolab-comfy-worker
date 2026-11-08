import asyncio
from contextlib import suppress
import gzip
from unittest.mock import AsyncMock

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
from vastai.serverless.server.lib.backend import Backend
from vastai.serverless.server.lib import server as sdk_server
from vastai.serverless.server.lib.data_types import LogAction

from conftest import connector, payload
from worker import close_sessions, unknown_session


async def test_real_sdk_reads_fatal_before_ready(tmp_path, monkeypatch):
    for name, value in {"CONTAINER_ID": "123", "REPORT_ADDR": "http://127.0.0.1",
                        "WORKER_PORT": "3000", "VAST_TCP_PORT_3000": "3000",
                        "PUBLIC_IPADDR": "127.0.0.1", "MASTER_TOKEN": "fake-master"}.items():
        monkeypatch.setenv(name, value)
    log = tmp_path / "worker.log"
    log.write_text("COMFY_STAGE download\nCOMFY_FATAL download OSError\n")
    backend = Backend("http://127.0.0.1", str(log), None,
                      [(LogAction.ModelLoaded, "COMFY_READY"), (LogAction.ModelError, "COMFY_FATAL")])
    reported = asyncio.Event()
    messages = []
    def errored(message):
        messages.append(message)
        reported.set()
    backend.metrics._model_errored = errored
    task = asyncio.create_task(backend._Backend__read_logs())
    try:
        await asyncio.wait_for(reported.wait(), timeout=3)
        assert messages == ["COMFY_FATAL"]
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_session_handler_never_reads_body():
    class Unreadable:
        def __getattr__(self, name):
            raise AssertionError("session handler touched request")
    response = await unknown_session(Unreadable())
    assert response.status == 404 and response.text == '{"error": "unknown"}'


async def test_real_sdk_apps_close_sessions_without_backend_traffic(monkeypatch):
    for name, value in {"CONTAINER_ID": "123", "REPORT_ADDR": "http://127.0.0.1",
                        "WORKER_PORT": "3000", "VAST_TCP_PORT_3000": "3000",
                        "PUBLIC_IPADDR": "127.0.0.1", "MASTER_TOKEN": "fake-master",
                        "UNSECURED": "false", "USE_SSL": "false"}.items():
        monkeypatch.setenv(name, value)
    received = []

    async def record(request):
        received.append(request.path)
        return web.json_response({})

    def fake_app():
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", record)
        return app

    async with TestServer(fake_app()) as comfy, TestServer(fake_app()) as model:
        backend = Backend(str(model.make_url("/")), "unused", None, [])
        close_sessions(backend)
        assert all(getattr(backend, f"session_{name}_handler") is unknown_session
                   for name in ("create", "get", "end", "health"))
        backend._start_tracking = AsyncMock()
        runners = []

        class Site:
            def __init__(self, runner, **kwargs):
                runners.append(runner)

            async def start(self):
                pass

        # Exercise the real SDK factory for both apps, with no public/TLS listener.
        with monkeypatch.context() as patch:
            patch.setenv("WORKER_PORT", "3000")
            patch.setenv("USE_SSL", "false")
            patch.setattr(sdk_server.web, "TCPSite", Site)
            await sdk_server.start_server_async(backend, [])
        assert len(runners) == 2
        async with ClientSession(connector=connector()) as session:
            backend.session = session
            try:
                for runner in runners:
                    app = runner.app
                    routes = {route.resource.canonical for route in app.router.routes()}
                    async with TestClient(TestServer(app), connector=connector()) as http:
                        for route in routes & {"/session/create", "/session/get", "/session/end", "/session/health"}:
                            for target in ("/clip/start", str(model.make_url("/clip/start")), str(comfy.make_url("/prompt"))):
                                body = {"auth_data": {"cost": 0, "reqnum": 0, "request_idx": 0},
                                        "payload": {"lifetime": 0, "on_close_route": target,
                                                    "on_close_payload": payload()}}
                                response = await http.post(route, json=body)
                                assert response.status == 404 and await response.json() == {"error": "unknown"}
                            response = await http.post(route, data=gzip.compress(b"x" * 100000),
                                                       headers={"Content-Encoding": "gzip"})
                            assert response.status == 404 and await response.json() == {"error": "unknown"}
                            response = await http.post(route, data=b"not json")
                            assert response.status == 404
                assert backend.sessions == {} and received == []
            finally:
                for runner in runners:
                    await runner.cleanup()
