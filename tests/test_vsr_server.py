import asyncio
from contextlib import suppress
from fractions import Fraction
import subprocess

from aiohttp import ClientSession
from aiohttp.test_utils import TestClient, TestServer
import pytest

from comfy_client import ComfyClient
from contract import validate_start
from graph import build_graph
from self_destruct import IdleRetirement
from server import JobManager, make_app
import vsr
from conftest import connector, payload
from test_regen_server import ReadyModels, audio_hash
from test_server import FakeComfy, wait_state
from test_vsr import enabled, make_clip, vsr_payload


@pytest.mark.parametrize("fps,audio", [("30", True), ("30000/1001", True), ("24", False)])
async def test_fake_comfy_mux_preserves_source_timing_and_audio(enabled, tmp_path, fps, audio):
    raw = make_clip(tmp_path / "draft.mp4", fps=fps, audio=audio)
    request = vsr_payload(raw)
    graph = build_graph(validate_start(request), "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    generated = make_clip(tmp_path / "generated.mp4", size="128x96", fps=fps, tone=880)
    fake = FakeComfy(tmp_path, video_bytes=generated, object_nodes={n["class_type"] for n in graph.values()}, node_events=True)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.01)
        manager = JobManager(backend, tmp_path, IdleRetirement(lambda _: None, env={}), vsr_models=ReadyModels())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=request)
            assert response.status == 200, await response.text()
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "done")
            assert state["vsr"] == dict(scale=2, fit=None, width=128, height=96)
            assert state["exec_s"] >= state["model_s"] >= 0
            assert state["fps"] == float(Fraction(fps)) and state["frame_rate"] == str(Fraction(fps))
            assert state["frames"] == 60 and "lab" not in state and "regen" not in state
            response = await http.post("/clip/file", json={"id": job_id})
            output = tmp_path / "output.mp4"
            output.write_bytes(await response.read())
            assert response.status == 200 and int(response.headers["Content-Length"]) == output.stat().st_size
            info = vsr.video_info(vsr.probe_video(output, timing=True))
            assert info.fps == Fraction(fps) and info.duration == 60 / Fraction(fps)
            assert info.audio == audio
            if audio:
                assert audio_hash(output) == audio_hash(tmp_path / "draft.mp4")
            assert len(list(manager.jobs[job_id].directory.iterdir())) == 1
            assert fake.deleted == [fake.prompt_id] and not fake.interrupted
            assert (await (await http.post("/clip/start", json=request)).json())["id"] == job_id
            request["vsr"]["scale"] = 4
            assert (await http.post("/clip/start", json=request)).status == 409


def test_fit_single_pass_and_audio_copy(enabled, tmp_path, monkeypatch):
    # Lower the fit target in this test to exercise the real pass cheaply.
    monkeypatch.setitem(vsr.FIT, "2k", 96)
    draft = tmp_path / "draft.mp4"
    request = validate_start(vsr_payload(make_clip(draft, fps="30000/1001"), fit="2k"))
    rendered = tmp_path / "rendered.mp4"
    make_clip(rendered, fps="30000/1001", size="128x96", tone=880)
    command, calls = vsr.media_command, []
    def capture(args, **kw):
        calls.append(args)
        return command(args, **kw)
    monkeypatch.setattr(vsr, "media_command", capture)
    result = vsr.finish_video(rendered, draft, request.video, request.vsr.public())
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("-vf") + 1] == "scale=96:72:flags=lanczos"
    assert args[args.index("-c:a") + 1] == "copy" and args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-fps_mode") + 1] == "passthrough" and "-r" not in args
    info = vsr.video_info(vsr.probe_video(result, timing=True))
    assert (info.width, info.height) == (96, 72) and info.fps == Fraction(30000, 1001)
    assert info.duration == request.video.duration and audio_hash(result) == audio_hash(draft)


def test_without_downscale_stream_copies_video(enabled, tmp_path, monkeypatch):
    draft, rendered = tmp_path / "draft.mp4", tmp_path / "rendered.mp4"
    request = validate_start(vsr_payload(make_clip(draft), fit="4k"))
    make_clip(rendered, size="128x96", tone=880)
    command, calls = vsr.media_command, []
    def capture(args, **kw):
        calls.append(args)
        return command(args, **kw)
    monkeypatch.setattr(vsr, "media_command", capture)
    vsr.finish_video(rendered, draft, request.video, request.vsr.public())
    assert len(calls) == 1 and "-vf" not in calls[0]
    assert calls[0][calls[0].index("-c:v") + 1] == "copy"


@pytest.mark.parametrize("fit", [None, "2k"])
def test_audio_tail_and_video_delay_survive_mux(enabled, tmp_path, monkeypatch, fit):
    monkeypatch.setitem(vsr.FIT, "2k", 96)
    base, draft, rendered = (tmp_path / name for name in ("base.mp4", "draft.mp4", "rendered.mp4"))
    make_clip(base, audio=False)
    # Video begins half a second after the audio and ends half a second before it.
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-itsoffset", "0.5", "-i", str(base),
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-t", "3", str(draft)], check=True, capture_output=True, timeout=30)
    request = validate_start(vsr_payload(draft.read_bytes(), fit=fit))
    assert request.video.start == Fraction(1, 2) and request.video.clip_duration == 3
    make_clip(rendered, size="128x96", tone=880)
    result = vsr.finish_video(rendered, draft, request.video, request.vsr.public())
    info = vsr.video_info(vsr.probe_video(result, timing=True))
    assert info.start == request.video.start and info.clip_duration == 3
    assert audio_hash(result) == audio_hash(draft)


def test_temp_cleanup_is_scoped(tmp_path):
    scratch = tmp_path / "recavsr"
    scratch.mkdir()
    owned = scratch / ("a" * 32 + ".mp4")
    other = scratch / "unrelated.mp4"
    outside = tmp_path / ("b" * 32 + ".mp4")
    for path in (owned, other, outside):
        path.write_bytes(b"file")
    vsr.clear_temporary_videos(scratch)
    assert not owned.exists() and other.exists() and outside.exists()


@pytest.mark.parametrize("changes", [dict(frames=61), dict(fps="24"), dict(size="64x48")])
def test_mismatched_render_rejected(enabled, tmp_path, changes):
    draft, rendered = tmp_path / "draft.mp4", tmp_path / "rendered.mp4"
    request = validate_start(vsr_payload(make_clip(draft)))
    make_clip(rendered, **(dict(size="128x96") | changes))
    with pytest.raises(RuntimeError, match="timing or dimensions"):
        vsr.finish_video(rendered, draft, request.video, request.vsr.public())


async def test_download_running_done_timing_and_failure_recovery(enabled, tmp_path):
    release_download, release_render, entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    now = [0.]
    raw = make_clip(tmp_path / "draft.mp4")
    generated = make_clip(tmp_path / "generated.mp4", size="128x96", tone=880)
    scratch = tmp_path / "temp" / "recavsr"
    scratch.mkdir(parents=True)
    class Models(ReadyModels):
        async def wait(self, progress):
            progress(.4)
            await release_download.wait()
    class Backend:
        healthy = True
        sampler_s = .3
        fail = True
        async def render(self, graph, directory, progress):
            (scratch / ("c" * 32 + ".mp4")).write_bytes(b"intermediate")
            entered.set()
            await release_render.wait()
            now[0] += 1
            if self.fail:
                self.fail = False
                (directory / "partial.mp4").write_bytes(b"partial")
                raise RuntimeError("CUDA out of memory: private path")
            output = directory / "rendered.mp4"
            output.write_bytes(generated)
            return output
    manager = JobManager(Backend(), tmp_path / "jobs", IdleRetirement(lambda _: None, env={}),
                         clock=lambda: now[0], vsr_models=Models(), vsr_temp=scratch)
    run = asyncio.create_task(manager.run())
    try:
        first = manager.submit(validate_start(vsr_payload(raw)), {})
        await asyncio.sleep(.02)
        assert first.state == "queued" and first.stage == "downloading_vsr_models" and first.progress == .4
        assert first.exec_s is None and not entered.is_set()
        now[0] = 100
        release_download.set()
        await asyncio.wait_for(entered.wait(), 1)
        assert first.state == "running" and first.stage == "running"
        release_render.set()
        await asyncio.wait_for(manager.queue.join(), 3)
        assert first.state == "failed" and first.error == "generation failed"
        assert first.exec_s == 1 and first.stage is None and not list(first.directory.iterdir())
        assert not list(scratch.iterdir())
        second = manager.submit(validate_start({**vsr_payload(raw), "request": "b" * 32}), {})
        await asyncio.sleep(.02)
        await asyncio.wait_for(manager.queue.join(), 3)
        assert second.state == "done" and second.exec_s == 1 and second.model_s == .3
        assert manager.active == manager.retirement.jobs == 0 and manager.backend.healthy
        assert not list(scratch.iterdir())
    finally:
        run.cancel()
        with suppress(asyncio.CancelledError):
            await run


async def test_download_failure_does_not_block_shooting(enabled, tmp_path):
    release = asyncio.Event()
    class Waiting(ReadyModels):
        async def wait(self, progress):
            progress(.5)
            await release.wait()
            raise OSError("private download failure")
    fake = FakeComfy(tmp_path)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        manager = JobManager(ComfyClient(session, str(comfy.make_url("/")), poll_s=.01), tmp_path,
                             IdleRetirement(lambda _: None, env={}), vsr_models=Waiting())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            raw = make_clip(tmp_path / "draft.mp4")
            response = await http.post("/clip/start", json=vsr_payload(raw))
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "queued")
            assert state["stage"] == "downloading_vsr_models"
            response = await http.post("/clip/start", json=payload())
            await wait_state(http, (await response.json())["id"], "done")
            release.set()
            state = await wait_state(http, job_id, "failed")
            assert state["error"] == "vsr model download failed; retry the job"
            assert not list(manager.jobs[job_id].directory.iterdir())


async def test_application_does_not_prefetch_vsr(enabled, tmp_path):
    class LazyModels(ReadyModels):
        closed = False
        def start(self):
            pytest.fail("VSR transfer must start only on first use")
        async def close(self):
            self.closed = True
    class Backend:
        healthy = True
    models = LazyModels()
    manager = JobManager(Backend(), tmp_path, IdleRetirement(lambda _: None, env={}), vsr_models=models)
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        assert (await http.get("/health")).status == 200
    assert models.closed


@pytest.mark.parametrize("missing", ["ReCaVSRLoader", "ReCaVSRUpscale", "LoadVideo", "SaveVideo"])
async def test_missing_nodes_fail_cleanly(enabled, tmp_path, missing):
    raw = make_clip(tmp_path / "draft.mp4")
    fake = FakeComfy(tmp_path, object_nodes={"ReCaVSRLoader", "ReCaVSRUpscale", "LoadVideo", "SaveVideo"} - {missing})
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.01)
        manager = JobManager(backend, tmp_path, IdleRetirement(lambda _: None, env={}), vsr_models=ReadyModels())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=vsr_payload(raw))
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "failed")
            assert state["error"] == "lab requires ComfyUI node " + missing
            assert backend.healthy and fake.graph is None and not list(manager.jobs[job_id].directory.iterdir())
