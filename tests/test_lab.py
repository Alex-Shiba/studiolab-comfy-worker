import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
import pytest

import download
from comfy_client import ComfyClient, LabUnavailable
from conftest import connector, encoded_image, payload
from contract import Invalid, validate_start
from graph import build_graph
from lab import PROFILES
from reference_models import LabModels, ReferenceModels
from self_destruct import IdleRetirement
from server import JobManager, make_app
from test_server import FakeBackend, start, wait_state


def request_for(profile, **lab):
    mode = {"mode": "refs", "refs": [encoded_image()]} if profile.startswith("ref") else {}
    return payload(**mode, lab={"lora": profile, **lab})


def graph_for(raw):
    job = "a" * 32
    return build_graph(validate_start(raw), job,
                       first=job + "/" + "b" * 32 + ".png" if raw["mode"] == "frames" else None,
                       refs=[job + "/" + "b" * 32 + ".png"] if raw["mode"] == "refs" else [])


@pytest.mark.parametrize("mode", ["text", "frames", "refs"])
def test_empty_lab_matches_absent_graph_and_fingerprint(mode):
    images = {"first": encoded_image()} if mode == "frames" else {"refs": [encoded_image()]} if mode == "refs" else {}
    raw = payload(mode=mode, **images)
    explicit = {"lora": "ref8_768"} if mode == "refs" else {
        "lora": "turbo4_768", "attention": "kitchen", "sparse": "sol"}
    assert graph_for(raw) == graph_for({**raw, "lab": {}}) == graph_for({**raw, "lab": explicit})
    assert validate_start(raw).fingerprint() == validate_start({**raw, "lab": {}}).fingerprint()
    assert validate_start(raw).fingerprint() == validate_start({**raw, "lab": explicit}).fingerprint()
    g = graph_for(raw)
    assert g["10"]["inputs"]["lora_name"] == PROFILES[explicit["lora"]].filename
    assert g["21"]["inputs"] == {"model": ["10", 0], "shift_video": 6, "shift_audio": 3}
    assert g["15"]["inputs"]["steps"] == (8 if mode == "refs" else 4)
    if mode == "refs":
        assert "22" not in g and "23" not in g
        assert g["12"]["inputs"]["model"] == g["15"]["inputs"]["model"] == ["21", 0]
    else:
        assert g["22"] == {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["21", 0], "attention": "comfy kitchen attention"}}
        assert g["23"] == {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["22", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": .2,
            "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        assert sum(n["class_type"] == "BlockSparseAttention" for n in g.values()) == 1
        assert g["12"]["inputs"]["model"] == g["15"]["inputs"]["model"] == ["23", 0]


@pytest.mark.parametrize("mode", ["text", "frames"])
@pytest.mark.parametrize("lab", [{"lora": "turbo8_544"}, {"steps": 8}, {"attention": "default"}, {"sparse": "off"}])
def test_explicit_legacy_profile_and_per_key_defaults(mode, lab):
    raw = payload(mode=mode, lab=lab, **({"first": encoded_image()} if mode == "frames" else {}))
    g = graph_for(raw)
    assert g["10"]["inputs"]["lora_name"] == PROFILES["turbo8_544"].filename
    assert g["21"]["inputs"] == {"model": ["10", 0], "shift_video": 12, "shift_audio": 3}
    assert g["15"]["inputs"] == {"model": ["21", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}
    assert g["12"]["inputs"]["model"] == ["21", 0]
    assert "22" not in g and "23" not in g


@pytest.mark.parametrize("mode", ["text", "frames"])
@pytest.mark.parametrize("missing", ["ModelAttentionBackend", "BlockSparseAttention", "backend"])
async def test_default_capabilities_fail_before_prompt(mode, missing, tmp_path):
    raw = payload(mode=mode, **({"first": encoded_image()} if mode == "frames" else {}))
    graph = graph_for(raw)
    info = {node["class_type"]: {} for node in graph.values()}
    info["ModelAttentionBackend"] = {"input": {"required": {"attention": [
        "COMBO", {"options": ["pytorch attention"] if missing == "backend" else ["comfy kitchen attention"]}]}}}
    if missing != "backend":
        del info[missing]
    client = ComfyClient(None)
    client.json = AsyncMock(return_value=info)
    message = "INT8 attention backend" if missing == "backend" else "lab requires ComfyUI node " + missing
    with pytest.raises(LabUnavailable, match=message):
        await client.render(graph, tmp_path, lambda _: None)
    client.json.assert_awaited_once_with("GET", "/object_info")


@pytest.mark.parametrize("profile,steps,video", [
    ("turbo8_544", 8, 12), ("turbo8_768", 8, 6), ("turbo4_768", 4, 6),
    ("pika6", 6, 12), ("ref_v01", 4, 12), ("ref8_768", 8, 6),
])
@pytest.mark.parametrize("attention", ["default", "kitchen"])
@pytest.mark.parametrize("sparse", ["off", "sol"])
def test_profiles_and_accelerators(profile, steps, video, attention, sparse):
    g = graph_for(request_for(profile, attention=attention, sparse=sparse))
    assert g["10"]["inputs"] == {"model": ["1", 0], "lora_name": PROFILES[profile].filename, "strength_model": 1.0}
    assert sum(n["class_type"] == "LoraLoaderModelOnly" for n in g.values()) == 1
    assert g["21"]["inputs"] == {"model": ["10", 0], "shift_video": video, "shift_audio": 3}
    model = ["21", 0]
    assert ("22" in g) == (attention == "kitchen")
    if attention == "kitchen":
        assert g["22"] == {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": "comfy kitchen attention"}}
        model = ["22", 0]
    assert ("23" in g) == (sparse == "sol")
    if sparse == "sol":
        assert g["23"] == {"class_type": "BlockSparseAttention", "inputs": {
            "model": model, "selection": "sol-attn", "selection.tau": 1.3, "start_percent": .2,
            "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["23", 0]
    assert g["12"] == {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}}
    if profile == "pika6":
        assert g["14"]["inputs"]["sampler_name"] == "lcm"
        assert g["15"]["class_type"] == "ManualSigmas"
        assert [float(n) for n in g["15"]["inputs"]["sigmas"].split(",")] == [1, .972973, .923077, .8, .571429, .333333, 0]
    else:
        assert g["15"]["inputs"] == {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}
        assert g["14"]["inputs"]["sampler_name"] == "res_multistep"


@pytest.mark.parametrize("lab", [
    None, [], True, "default", {"other": 1}, {"attention": []}, {"attention": "sage"},
    {"sparse": ["sol", "sla"]}, {"sparse": "sla"}, {"sparse": "vsa"}, {"sparse": None},
    {"lora": []}, {"lora": "ref_v01"}, {"lora": "ref8_768"}, {"lora": "unknown"},
    {"steps": True}, {"steps": 8.0}, {"steps": 0}, {"steps": 9},
    {"lora": "turbo4_768", "steps": 8}, {"lora": "turbo8_768", "steps": 4},
    {"lora": "pika6", "steps": 4}, {"lora": "pika6", "shift": {"video": 12, "audio": 3}},
    {"shift": None}, {"shift": [6, 3]}, {"shift": {"video": 6}},
    {"shift": {"video": 6, "audio": 3, "extra": 1}}, {"shift": {"video": True, "audio": 3}},
    {"shift": {"video": 0, "audio": 3}}, {"shift": {"video": 6, "audio": 101}},
    {"shift": {"video": float("nan"), "audio": 3}}, {"shift": {"video": 6, "audio": float("inf")}},
    {"shift": {"video": 10 ** 400, "audio": 3}},
])
def test_strict_lab_validation(lab):
    with pytest.raises(Invalid):
        validate_start(payload(lab=lab))


@pytest.mark.parametrize("profile", ["turbo8_544", "turbo8_768", "turbo4_768", "pika6"])
def test_refs_reject_fl2va_profiles(profile):
    with pytest.raises(Invalid, match="mode"):
        validate_start(payload(mode="refs", refs=[encoded_image()], lab={"lora": profile}))


@pytest.mark.parametrize("profile", [p for p in PROFILES if p != "pika6"])
def test_turbo_shift_overrides(profile):
    g = graph_for(request_for(profile, shift={"video": 7.5, "audio": 2}))
    assert g["21"]["inputs"]["shift_video"] == 7.5 and g["21"]["inputs"]["shift_audio"] == 2


def test_steps_override_and_normalized_fingerprint():
    raw = payload()
    base = validate_start(raw).fingerprint()
    assert base == validate_start({**raw, "lab": {}}).fingerprint()
    assert base != validate_start({**raw, "lab": {"lora": "turbo8_544", "steps": 8}}).fingerprint()
    for lab in [{"attention": "kitchen"}, {"sparse": "sol"}, {"lora": "turbo4_768"},
                {"steps": 4}, {"shift": {"video": 7, "audio": 3}}]:
        assert base != validate_start({**raw, "lab": lab}).fingerprint()
    assert graph_for(payload(lab={"steps": 4}))["15"]["inputs"]["steps"] == 4


@pytest.mark.parametrize("profile", ["ref_v01", "turbo8_768", "turbo8_544", "pika6"])
def test_lazy_pins_and_download_destination(profile, tmp_path, monkeypatch):
    data = download.manifest()
    item = data["lab_files"][profile]
    assert item["path"] == "loras/" + PROFILES[profile].filename
    assert len(item["sha256"]) == 64 and len(item.get("revision", data["revision"])) == 40
    calls = []
    def fetch(url, path, size, digest, **kwargs):
        calls.append((url, path, size, digest))
        if "progress" in kwargs:
            kwargs["progress"](1)
    monkeypatch.setattr(download, "download_file", fetch)
    download.download_models(tmp_path)
    download.download_reference_models(tmp_path)
    assert not any(c[1].name == PROFILES[profile].filename for c in calls)
    calls.clear()
    progress = []
    download.download_lab_model(profile, tmp_path, progress.append)
    assert calls == [(f"https://huggingface.co/{item.get('repository', data['repository'])}/resolve/"
                      f"{item.get('revision', data['revision'])}/{item.get('source_path', item['path'])}",
                      tmp_path / item["path"], item["size"], item["sha256"])]
    assert progress[-1] == 1


@pytest.mark.parametrize("profile", ["ref_v01", "turbo8_768", "turbo8_544", "pika6"])
async def test_lazy_background_job_stage_idempotency_state_and_retry(profile, tmp_path):
    entered, release = threading.Event(), threading.Event()
    attempts = []
    def fetch(selected, root, progress, cancelled):
        attempts.append(selected)
        entered.set()
        progress(.4)
        assert release.wait(5)
        if len(attempts) == 1:
            raise OSError("private URL")
    ready = tmp_path / "ready"
    models = LabModels(tmp_path, ready, fetch)
    refs = ReferenceModels(tmp_path, ready, lambda *args: None)
    backend = FakeBackend()
    backend.sampler_s = 1.25
    backend.release.set()
    manager = JobManager(backend, tmp_path / "jobs", IdleRetirement(lambda _: None, env={}),
                         reference_models=refs, lab_models=models)
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        assert not models.models and not attempts
        raw = request_for(profile)
        job = (await (await http.post("/clip/start", json=raw)).json())["id"]
        await asyncio.sleep(.12)
        assert not entered.is_set()
        ready.touch()
        assert await asyncio.to_thread(entered.wait, 2)
        for _ in range(100):
            state = await (await http.post("/clip/state", json={"id": job})).json()
            if state.get("progress") == .4:
                break
            await asyncio.sleep(.01)
        assert state["stage"] == "downloading_lab_lora" and state["progress"] == .4
        assert state["lab"]["lora"] == profile
        assert (await http.get("/health")).status == 200
        assert (await (await http.post("/clip/start", json=raw)).json())["id"] == job
        assert (await http.post("/clip/start", json={**raw, "lab": {**raw["lab"], "attention": "kitchen"}})).status == 409
        second = await start(http, **{k: v for k, v in raw.items() if k != "request"})
        text = await start(http)
        await wait_state(http, text, "done")
        assert backend.order == [text]
        release.set()
        for job_id in (job, second):
            state = await wait_state(http, job_id, "failed")
            assert state["error"] == "lab LoRA download failed; retry the job"
        assert attempts == [profile]
        retried = await start(http, **{k: v for k, v in raw.items() if k != "request"})
        done = await wait_state(http, retried, "done")
        assert done["exec_s"] >= 0 and done["sampler_s"] == 1.25 and done["lab"]["lora"] == profile
        assert attempts == [profile, profile]
        await models.wait(profile, lambda _: None)
        assert len(attempts) == 2 and manager.active == 0


@pytest.mark.parametrize("kind,detail", [
    ("ModelAttentionBackend", "lab requires ComfyUI node ModelAttentionBackend"),
    ("BlockSparseAttention", "lab requires ComfyUI node BlockSparseAttention"),
    ("ManualSigmas", "lab requires ComfyUI node ManualSigmas"),
    ("MiniMaxH3SigmaShift", "lab requires ComfyUI node MiniMaxH3SigmaShift"),
])
async def test_missing_optional_nodes_fail_before_submission(kind, detail, tmp_path):
    client = ComfyClient(None)
    client.json = AsyncMock(return_value={})
    with pytest.raises(LabUnavailable, match=detail):
        await client.render({"x": {"class_type": kind, "inputs": {}}}, tmp_path, lambda _: None)
    client.json.assert_awaited_once_with("GET", "/object_info")


@pytest.mark.parametrize("shape", ["v3", "v1"])
@pytest.mark.parametrize("backends,available", [([], False), (["pytorch attention"], False), (["pytorch attention", "comfy kitchen attention"], True)])
async def test_kitchen_must_not_silently_fall_back(backends, available, shape):
    # ComfyUI 0.44 serves the V3 io.Combo of ModelAttentionBackend as ["COMBO", {"options": [...]}]; the V1 list shape stays supported.
    spec = ["COMBO", {"options": backends}] if shape == "v3" else [backends]
    client = ComfyClient(None)
    client.json = AsyncMock(return_value={"ModelAttentionBackend": {"input": {"required": {"attention": spec}}}})
    graph = {"22": {"class_type": "ModelAttentionBackend"}}
    if available:
        await client.check_lab(graph)
    else:
        with pytest.raises(LabUnavailable, match="INT8 attention backend"):
            await client.check_lab(graph)


async def test_sampler_seconds_only_from_matching_execution_events():
    client = ComfyClient(None, clock=iter([10, 13.5]).__next__)
    async def events():
        for prompt, node in [("other", "16"), ("ours", "16"), ("ours", "17"), ("ours", "18")]:
            yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps({"type": "executing", "data": {"prompt_id": prompt, "node": node}}))
    await client._progress(events(), "ours", lambda _: None)
    assert client.sampler_s == 3.5


async def test_capability_error_reaches_state_and_next_default_job_recovers(tmp_path):
    client = ComfyClient(None)
    client.json = AsyncMock(return_value={"MiniMaxH3SigmaShift": {}, "BlockSparseAttention": {}})
    manager = JobManager(client, tmp_path, IdleRetirement(lambda _: None, env={}))
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        job = await start(http)
        failed = await wait_state(http, job, "failed")
        assert failed["error"] == "lab requires ComfyUI node ModelAttentionBackend"
        assert failed["lab"]["attention"] == "kitchen" and failed["exec_s"] >= 0
        assert "sampler_s" not in failed
        assert (await http.get("/health")).status == 200 and manager.active == 0
        backend = FakeBackend()
        backend.release.set()
        manager.backend = backend
        done = await wait_state(http, await start(http), "done")
        assert done["lab"]["attention"] == "kitchen" and "sampler_s" not in done


def test_lab_does_not_leak_into_next_default_graph():
    baseline = graph_for(payload())
    graph_for(payload(lab={"lora": "pika6", "attention": "kitchen", "sparse": "sol"}))
    assert graph_for(payload()) == baseline
