import asyncio
import json
import subprocess

from aiohttp import ClientSession
from aiohttp.test_utils import TestClient, TestServer
import pytest

from comfy_client import ComfyClient
from contract import validate_start
from graph import build_graph
from self_destruct import IdleRetirement
from server import JobManager, make_app
from conftest import connector, payload
from test_regen import draft_bytes, enabled, make_video, regen_payload
from test_server import FakeComfy, wait_state


class ReadyModels:
    def start(self):
        pass

    async def wait(self, progress):
        progress(1.)

    async def close(self):
        pass


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)]))


def audio_hash(path):
    return subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
        "-f", "hash", "-hash", "sha256", "-"])


@pytest.mark.parametrize("method", ["refine"])
@pytest.mark.parametrize("resolution", ["1080p", "2k"])
async def test_regen_end_to_end_fake_comfy_keeps_audio(enabled, draft_bytes, tmp_path, method, resolution):
    request = regen_payload(draft_bytes, method, resolution)
    expected = build_graph(validate_start(request), "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    node_kinds = {n["class_type"] for n in expected.values()} | {"MinimaxH3LatentUpscaler3D"}
    # The published graphs pad to 124 internally; a real MP4 stands in for GPU
    # output. Deliberately give it a different soundtrack, which must be replaced.
    generated = make_video(tmp_path / "generated.mp4", frames=124, tone=880)
    fake = FakeComfy(tmp_path, video_bytes=generated, object_nodes=node_kinds, node_events=True)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        backend = ComfyClient(session, str(comfy.make_url("/")), poll_s=.01)
        manager = JobManager(backend, tmp_path, IdleRetirement(lambda _: None, env={}),
                             reference_models=ReadyModels(), regen_models=ReadyModels())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=request)
            assert response.status == 200, await response.text()
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "done")
            assert state["regen"] == ({"method": method, "resolution": resolution} |
                                      ({"start_sigma": .8} if method == "refine" else {}))
            assert state["exec_s"] >= 0 and state["sampler_s"] >= 0
            assert state["frames"] == 120 and state["fps"] == 24
            assert (state["width"], state["height"]) == ((1920, 1080) if resolution == "1080p" else (2560, 1440))
            response = await http.post("/clip/file", json={"id": job_id})
            assert response.status == 200 and response.headers["Content-Type"] == "video/mp4"
            output = tmp_path / "download.mp4"
            output.write_bytes(await response.read())
            assert int(response.headers["Content-Length"]) == output.stat().st_size
            info = probe(output)
            assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
            assert int(next(s for s in info["streams"] if s["codec_type"] == "video")["nb_read_frames"]) == 120
            assert abs(float(info["format"]["duration"]) - 5.) <= 1 / 24
            assert audio_hash(output) == audio_hash(tmp_path / "draft.mp4")
            assert len(list(manager.jobs[job_id].directory.iterdir())) == 1
            assert fake.deleted == [fake.prompt_id] and not fake.interrupted
            assert (await (await http.post("/clip/start", json=request)).json())["id"] == job_id
            request["regen"]["resolution"] = "2k" if resolution == "1080p" else "1080p"
            assert (await http.post("/clip/start", json=request)).status == 409


async def test_silent_draft_does_not_gain_audio(enabled, tmp_path, monkeypatch):
    import regen
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    raw = make_video(tmp_path / "silent.mp4", audio=False)
    request = regen_payload(raw)
    graph = build_graph(validate_start(request), "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    fake = FakeComfy(tmp_path, video_bytes=make_video(tmp_path / "generated.mp4", frames=124),
                     object_nodes={n["class_type"] for n in graph.values()} | {"MinimaxH3LatentUpscaler3D"})
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        manager = JobManager(ComfyClient(session, str(comfy.make_url("/")), poll_s=.01), tmp_path,
                             IdleRetirement(lambda _: None, env={}), reference_models=ReadyModels(), regen_models=ReadyModels())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=request)
            job_id = (await response.json())["id"]
            await wait_state(http, job_id, "done")
            assert {s["codec_type"] for s in probe(manager.jobs[job_id].file)["streams"]} == {"video"}


@pytest.mark.parametrize("missing", ["OpenH3RegenFastVAEDecode", "OpenH3RegenPrepareDraft", "OpenH3RegenLora", "OpenH3RegenUpscale", "OpenH3RegenRefineLatent", "OpenH3RegenSigmas", "OpenH3RegenFinish", "LTXVSeparateAVLatent", "MinimaxH3LatentUpscaler3D"])
async def test_missing_custom_node_fails_before_prompt(enabled, draft_bytes, tmp_path, missing):
    method = "refine"
    request = regen_payload(draft_bytes, method)
    graph = build_graph(validate_start(request), "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    kinds = {n["class_type"] for n in graph.values()} | {"MinimaxH3LatentUpscaler3D"}
    fake = FakeComfy(tmp_path, object_nodes=kinds - {missing})
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        manager = JobManager(ComfyClient(session, str(comfy.make_url("/")), poll_s=.01), tmp_path,
                             IdleRetirement(lambda _: None, env={}), reference_models=ReadyModels(), regen_models=ReadyModels())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=request)
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "failed")
            assert state["error"] == "lab requires ComfyUI node " + missing
            assert fake.graph is None and not fake.interrupted
            assert not list(manager.jobs[job_id].directory.iterdir())


async def test_regen_download_stage_does_not_block_normal_jobs(enabled, draft_bytes, tmp_path):
    release = asyncio.Event()
    class Waiting(ReadyModels):
        async def wait(self, progress):
            progress(.5)
            await release.wait()
            raise RuntimeError("private download failure")
    fake = FakeComfy(tmp_path)
    async with TestServer(fake.app()) as comfy, ClientSession(connector=connector()) as session:
        manager = JobManager(ComfyClient(session, str(comfy.make_url("/")), poll_s=.01), tmp_path,
                             IdleRetirement(lambda _: None, env={}), reference_models=ReadyModels(), regen_models=Waiting())
        async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
            response = await http.post("/clip/start", json=regen_payload(draft_bytes))
            job_id = (await response.json())["id"]
            state = await wait_state(http, job_id, "queued")
            assert state["stage"] == "downloading_regen_models"
            response = await http.post("/clip/start", json=payload())
            ordinary = (await response.json())["id"]
            await wait_state(http, ordinary, "done")
            release.set()
            state = await wait_state(http, job_id, "failed")
            assert state["error"] == "regen model download failed; retry the job"
            assert not list(manager.jobs[job_id].directory.iterdir())
            assert manager.active == manager.retirement.jobs == 0
