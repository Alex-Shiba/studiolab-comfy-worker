import asyncio
import base64
import hashlib
import io
import threading
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

from aiohttp import ClientSession
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
import pytest

from comfy_client import ComfyClient, GenerationFailed
from conftest import connector, encoded_image, payload
from contract import decode_image, save_frame
from diagnostics import Diagnostics, Redactor
from download import download_file
from self_destruct import IdleRetirement
import server
from server import JobManager, make_app
from test_download import Response
from test_server import FakeComfy, app_client, start, wait_state


async def test_empty_interrupt_and_history_allow_next_start(tmp_path):
    fake = FakeComfy(tmp_path, websocket=False, hang=True)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.001,
                              timeout_s=.2, clock=lambda: fake.polls * .05)
        manager = JobManager(backend, tmp_path, IdleRetirement(lambda _: None, env={}))
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            first = await start(http)
            await wait_state(http, first, "failed")
            assert fake.interrupted and backend.healthy
            assert fake.deleted == [fake.prompt_id]
            assert (await http.get("/health")).status == 200
            fake.hang = False
            second = await start(http)
            await wait_state(http, second, "done")
            assert second != first and fake.deleted == [fake.prompt_id, fake.prompt_id]


@pytest.mark.parametrize("status", [200, 201, 429, 500])
async def test_status_only_never_parses_body(status):
    class Empty:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            pytest.fail("empty ComfyUI response was parsed")
    response = Empty()
    response.status = status
    session = SimpleNamespace(request=lambda *a, **k: response)
    client = ComfyClient(session)
    if status == 200:
        await client.status_only("POST", "/history", json={"delete": ["test"]})
    else:
        with pytest.raises(GenerationFailed):
            await client.status_only("POST", "/history", json={"delete": ["test"]})


@pytest.mark.parametrize("failure", ["oserror", "urlerror", 429, 500, 503, "range", "oversized"])
def test_download_retries_past_five_failures(tmp_path, failure):
    data = b"abcdef"
    path = tmp_path / "model"
    segment = tmp_path / "model.range-0-5"
    calls, delays = [], []
    def opener(request, timeout):
        calls.append(request.headers["Range"])
        if len(calls) <= 8:
            if failure == "oserror":
                raise OSError("temporary")
            if failure == "urlerror":
                raise URLError("temporary")
            if isinstance(failure, int):
                raise HTTPError(request.full_url, failure, "temporary", {}, None)
            if failure == "range":
                # Each invalid answer discards the old prefix before retrying.
                if len(calls) == 1:
                    segment.write_bytes(b"ab")
                else:
                    assert not segment.exists()
                return Response(b"wrong!", 1, 5, 6)
            return Response(data + b"extra", 0, 5, 6)
        assert not segment.exists()
        return Response(data, 0, 5, 6)
    download_file("https://example.invalid/model", path, len(data), hashlib.sha256(data).hexdigest(),
                  connections=1, opener=opener, sleep=delays.append)
    assert len(calls) == 9 and path.read_bytes() == data
    assert len(delays) == 8 and max(delays) <= 30
    assert delays == [1, 2, 4, 8, 16, 30, 30, 30]
    assert list(tmp_path.iterdir()) == [path]


def test_download_progress_resets_backoff(tmp_path):
    data = b"abcdef"
    starts, delays = [], []
    def opener(request, timeout):
        start = int(request.headers["Range"].split("=")[1].split("-")[0])
        starts.append(start)
        if len(starts) in (1, 2, 4):
            raise URLError("temporary")
        return Response(data[start:2] if len(starts) == 3 else data[start:], start, 5, 6)
    path = tmp_path / "model"
    download_file("https://example.invalid/model", path, 6, hashlib.sha256(data).hexdigest(),
                  connections=1, opener=opener, sleep=delays.append)
    assert starts == [0, 0, 0, 2, 2] and delays == [1, 2, 1, 2]
    assert path.read_bytes() == data


def test_short_secret_flags_preserve_diagnostics_and_long_keys_do_not(tmp_path):
    key = "fake-key-" * 5
    env = {"OPEN_BUTTON_TOKEN": "1", "ZERO_TOKEN": "0", "BOOL_TOKEN": "true",
           "FALSE_TOKEN": "false", "SHORT_TOKEN": "1234567", "MASTER_TOKEN": key}
    stdout = io.StringIO()
    path = tmp_path / "log"
    diagnostics = Diagnostics(path, env=env, stdout=stdout)
    text = "torch 2.14.1 frames=124 duration=1.01 true false 1234567 "
    # The key crosses the 8192-character stream boundary in Diagnostics.pump.
    prefix = text + "." * (8192 - len(text) - 17)
    diagnostics.pump(io.StringIO(prefix + key + "\n"))
    diagnostics.close()
    for output in (stdout.getvalue(), path.read_text()):
        assert output == prefix + "[REDACTED]\n"
        assert key not in output and text in output
    redactor = Redactor(["1234567", "12345678", "TRUE", "FALSE"])
    assert redactor.feed("1234567 12345678 TRUE FALSE", final=True) == "1234567 [REDACTED] TRUE FALSE"


async def test_slow_crop_keeps_state_and_health_responsive(app_client, monkeypatch):
    http, manager, backend, _ = app_client
    existing = await start(http)
    await backend.entered.wait()
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    original = server.save_frame
    thread_ids = []
    def slow(*args):
        thread_ids.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        original(*args)
    monkeypatch.setattr(server, "save_frame", slow)
    task = asyncio.create_task(http.post("/clip/start", json=payload(
        mode="frames", first=encoded_image(), last=encoded_image())))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        state = await asyncio.wait_for(http.post("/clip/state", json={"id": existing}), .5)
        assert state.status == 200 and (await state.json())["state"] == "running"
        assert (await asyncio.wait_for(http.get("/health"), .5)).status == 200
        assert not task.done()
    finally:
        release.set()
        response = await task
    assert response.status == 200
    assert len(thread_ids) == 2 and threading.get_ident() not in thread_ids
    assert manager.active == 2


@pytest.mark.parametrize("race", ["same", "conflict", "capacity", "unhealthy", "retiring"])
async def test_crop_rechecks_request_and_capacity_atomically(app_client, monkeypatch, race):
    http, manager, _, _ = app_client
    if race == "capacity":
        await start(http)
        await start(http)
    original = server.save_frame
    loop = asyncio.get_running_loop()
    both_entered, release = asyncio.Event(), threading.Event()
    lock = threading.Lock()
    entered = 0
    def blocked(*args):
        nonlocal entered
        with lock:
            entered += 1
            if entered == 2:
                loop.call_soon_threadsafe(both_entered.set)
        assert release.wait(5)
        original(*args)
    monkeypatch.setattr(server, "save_frame", blocked)
    first = payload(mode="frames", first=encoded_image())
    second = dict(first)
    if race == "conflict":
        second["seed"] += 1
    elif race == "capacity":
        second["request"] = "b" * 32
    tasks = [asyncio.create_task(http.post("/clip/start", json=data)) for data in (first, second)]
    try:
        await asyncio.wait_for(both_entered.wait(), 2)
        if race == "unhealthy":
            manager.backend.healthy = False
        if race == "retiring":
            manager.retirement.retiring = True
    finally:
        release.set()
        responses = await asyncio.gather(*tasks)
    statuses = sorted(response.status for response in responses)
    if race == "same":
        assert statuses == [200, 200]
        assert (await responses[0].json())["id"] == (await responses[1].json())["id"]
        assert manager.active == len(manager.jobs) == len(manager.requests) == 1
    elif race == "conflict":
        assert statuses == [200, 409]
        assert manager.active == len(manager.jobs) == len(manager.requests) == 1
    elif race == "capacity":
        assert statuses == [200, 429]
        assert manager.active == len(manager.jobs) == len(manager.requests) == 3
    else:
        assert statuses == [429, 429]
        assert not manager.jobs and not manager.requests and not list(manager.root.iterdir())
    assert len(list(manager.root.iterdir())) == len(manager.jobs)


async def test_decode_semaphore_bounds_crop_threads(app_client, monkeypatch):
    http, _, _, _ = app_client
    original = server.save_frame
    loop = asyncio.get_running_loop()
    both_entered, release = asyncio.Event(), threading.Event()
    lock = threading.Lock()
    entered = 0
    def blocked(*args):
        nonlocal entered
        with lock:
            entered += 1
            if entered == 2:
                loop.call_soon_threadsafe(both_entered.set)
        assert release.wait(5)
        original(*args)
    monkeypatch.setattr(server, "save_frame", blocked)
    tasks = [asyncio.create_task(http.post("/clip/start", json=payload(
        mode="frames", first=encoded_image()))) for _ in range(4)]
    try:
        await asyncio.wait_for(both_entered.wait(), 2)
        assert (await asyncio.wait_for(http.get("/health"), .5)).status == 200
        assert entered == 2 and not any(task.done() for task in tasks)
    finally:
        release.set()
        responses = await asyncio.gather(*tasks)
    assert sorted(response.status for response in responses) == [200, 200, 200, 429]


def test_16bit_gradient_preserves_midtones_after_crop(tmp_path):
    source = Image.frombytes("I;16", (256, 32),
                             b"".join((x * 257).to_bytes(2, "little") for x in range(256)) * 32)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    decoded = decode_image(base64.b64encode(buffer.getvalue()).decode())
    path = tmp_path / "frame.png"
    save_frame(decoded, path, (128, 16))
    with Image.open(path) as result:
        assert result.mode == "RGB" and result.size == (128, 16)
        row = [result.getpixel((x, 8))[0] for x in range(128)]
        assert row == sorted(row)
        assert row[0] < 5 and row[-1] > 250
        assert 125 <= row[64] <= 132
        assert len(set(row)) > 100
