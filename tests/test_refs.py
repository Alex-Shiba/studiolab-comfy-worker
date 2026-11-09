import asyncio
import base64
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer

import download
from contract import Invalid, validate_start
from graph import build_graph, REF_DIFFUSION, REF_LORA
from reference_models import ReferenceModels
from self_destruct import IdleRetirement
from server import JobManager, make_app
from conftest import connector, encoded_image, payload
from test_server import FakeBackend, start, wait_state


@pytest.mark.parametrize("count", [1, 4])
def test_refs_validation_and_order_fingerprint(count):
    pictures = [encoded_image(size=(32 + i, 24)) for i in range(count)]
    raw = payload(mode="refs", refs=pictures, prompt=f"<Picture {count}>")
    clip = validate_start(raw)
    assert [image.data for image in clip.refs] == [base64.b64decode(p) for p in pictures]
    assert clip.fingerprint() == validate_start(raw).fingerprint()
    changed = {**raw, "refs": [encoded_image(size=(80, 30)), *pictures[1:]]}
    assert clip.fingerprint() != validate_start(changed).fingerprint()
    if count > 1:
        assert clip.fingerprint() != validate_start({**raw, "refs": pictures[::-1]}).fingerprint()


@pytest.mark.parametrize("change", [
    {"refs": []}, {"refs": None}, {"refs": "bad"}, {"refs": ["bad"]},
    {"refs": [encoded_image()] * 5}, {"first": encoded_image()}, {"last": encoded_image()},
    {"mode": "text"}, {"mode": "frames", "first": encoded_image()},
    {"prompt": "<Picture 0>"}, {"prompt": "<Picture 2>"},
    {"refs": [encoded_image(fmt="GIF")]}, {"refs": [encoded_image(size=(8193, 1))]},
])
def test_invalid_refs(change):
    with pytest.raises(Invalid):
        validate_start(payload(mode="refs", refs=[encoded_image()]) | change)


def test_refs_missing_and_same_byte_limits(monkeypatch):
    with pytest.raises(Invalid):
        validate_start(payload(mode="refs"))
    import contract
    monkeypatch.setattr(contract, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(Invalid, match="12 MiB"):
        validate_start(payload(mode="refs", refs=[encoded_image()]))


@pytest.mark.parametrize("audio", [False, True])
def test_reference_graph(audio):
    clip = validate_start(payload(mode="refs", refs=[encoded_image()] * 4, audio=audio))
    job = "a" * 32
    names = [job + "/" + str(i) * 32 + ".png" for i in range(4)]
    graph = build_graph(clip, job, refs=names)
    assert graph["1"]["inputs"]["unet_name"] == REF_DIFFUSION
    assert graph["10"]["inputs"]["lora_name"] == REF_LORA
    assert graph["15"]["inputs"]["steps"] == 8
    assert graph["21"]["inputs"] == {"model": ["10", 0], "shift_video": 6, "shift_audio": 3}
    assert graph["14"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    inputs = graph["6"]["inputs"]
    assert inputs["vae"] == ["3", 0] and inputs["audio_vae"] == ["4", 0]
    assert inputs["ref_image_size"] == "match"
    for i, name in enumerate(names):
        assert inputs[f"ref_images.ref_image_{i}"] == [str(30 + i), 0]
        assert graph[str(30 + i)]["inputs"]["image"] == name
    assert ("audio" in graph["19"]["inputs"]) is audio
    assert graph["16"]["inputs"]["latent_image"] == ["6", 1]
    with pytest.raises(ValueError):
        build_graph(clip, job, refs=["../bad.png"] * 4)
    with pytest.raises(ValueError):
        build_graph(clip, job, refs=names[:1])


def test_pins_and_boot_download_excludes_refs(tmp_path, monkeypatch):
    data = download.manifest()
    assert {item["path"].split("/")[-1] for item in data["reference_files"]} == {REF_DIFFUSION, REF_LORA}
    assert [item["size"] for item in data["reference_files"]] == [20970379616, 1956193000]
    calls = []
    def fetch(url, target, size, digest, **kwargs):
        calls.append((url, size, digest, kwargs))
        if "progress" in kwargs:
            kwargs["progress"](1)
    monkeypatch.setattr(download, "download_file", fetch)
    download.download_models(tmp_path)
    assert len(calls) == 5 and not any("ref2" in call[0] for call in calls)
    calls.clear()
    progress = []
    download.download_reference_models(tmp_path, progress.append)
    assert len(calls) == 2 and progress[-1] == 1
    for call, item in zip(calls, data["reference_files"]):
        assert item.get("revision", data["revision"]) in call[0]
        assert call[1:3] == (item["size"], item["sha256"])
        assert call[3]["max_attempts"] == 5


async def test_background_after_ready_shared_wait_failure_retry_and_health(tmp_path):
    ready = tmp_path / "ready"
    entered, release = threading.Event(), threading.Event()
    attempts = []
    def fetch(root, progress, cancelled):
        attempts.append(root)
        entered.set()
        progress(.4)
        assert release.wait(5)
        if len(attempts) == 1:
            raise OSError("private URL must not escape")
    models = ReferenceModels(tmp_path, ready, fetch)
    backend = FakeBackend()
    backend.release.set()
    manager = JobManager(backend, tmp_path / "jobs", IdleRetirement(lambda _: None, env={}), reference_models=models)
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        assert (await http.get("/health")).status == 200
        await asyncio.sleep(.15)
        assert not entered.is_set()
        ready.touch()
        assert await asyncio.to_thread(entered.wait, 2)
        raw = payload(mode="refs", refs=[encoded_image(size=(80, 30))])
        response = await http.post("/clip/start", json=raw)
        job_id = (await response.json())["id"]
        await asyncio.sleep(.15)
        state = await (await http.post("/clip/state", json={"id": job_id})).json()
        assert state["stage"] == "downloading_reference_model" and state["progress"] == .4
        assert not backend.order and (await http.get("/health")).status == 200
        assert (await (await http.post("/clip/start", json=raw)).json())["id"] == job_id
        assert (await http.post("/clip/start", json={**raw, "refs": [encoded_image()]})).status == 409
        second_ref = await start(http, mode="refs", refs=[encoded_image()])
        # A text job passes a waiting refs job; the optional download never occupies the renderer.
        text_id = await start(http)
        await wait_state(http, text_id, "done")
        assert backend.order == [text_id]
        # The original reference is saved without a keyframe's cover crop.
        from PIL import Image
        picture = next(manager.jobs[job_id].directory.glob("*.png"))
        with Image.open(picture) as image:
            assert image.size == (80, 30)
        release.set()
        failed = await wait_state(http, job_id, "failed")
        assert failed["error"] == "reference model download failed; retry the references job"
        await wait_state(http, second_ref, "failed")
        assert len(attempts) == 1 and not list(manager.jobs[job_id].directory.iterdir())
        retried = await start(http, mode="refs", refs=[encoded_image()])
        await wait_state(http, retried, "done")
        assert len(attempts) == 2 and manager.active == 0
        assert (await http.get("/health")).status == 200


def test_background_retry_bound(tmp_path):
    def fail(*args, **kwargs):
        raise OSError("offline")
    sleeps = []
    with pytest.raises(RuntimeError, match="retries exhausted"):
        download.download_file("https://example.invalid/model", tmp_path / "file", 1, "0" * 64,
                               opener=fail, sleep=sleeps.append, max_attempts=3)
    assert sleeps == [1, 2]


def test_failed_transfer_does_not_cancel_next_transfer(tmp_path):
    import hashlib
    from test_download import Response
    cancelled = threading.Event()
    target = tmp_path / "reference"
    data = b"verified reference"
    digest = hashlib.sha256(data).hexdigest()
    def fail(*args, **kwargs):
        raise OSError("offline")
    with pytest.raises(RuntimeError):
        download.download_file("https://example.invalid/model", target, len(data), digest, connections=1,
                               opener=fail, max_attempts=1, cancelled=cancelled)
    assert not cancelled.is_set()
    progress = []
    download.download_file("https://example.invalid/model", target, len(data), digest, connections=1,
                           opener=lambda *a, **k: Response(data, 0, len(data) - 1, len(data)),
                           max_attempts=1, cancelled=cancelled, progress=progress.append)
    assert target.read_bytes() == data and progress[-1] == 1


async def test_shutdown_before_ready_never_downloads(tmp_path):
    def fetch(*args):
        pytest.fail("download before READY")
    models = ReferenceModels(tmp_path, tmp_path / "not-ready", fetch)
    models.start()
    await asyncio.sleep(0)
    await models.close()
    assert models.stopped.is_set() and models.task.cancelled()
