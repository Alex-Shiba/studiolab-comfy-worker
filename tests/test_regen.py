import base64
import subprocess

import pytest

from contract import Invalid, strict_json, validate_start
from graph import build_graph, HYPERFLOW, SHARP, IC_LORA, FAST_IC, FAST_REFINE, REF_DIFFUSION
import regen


def make_video(path, frames=120, fps=24, audio=True, tone=440):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=64x48:r={fps}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=32000"]
    args += ["-t", str(frames / fps), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run(args + [str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
    return path.read_bytes()


@pytest.fixture
def draft_bytes(tmp_path, monkeypatch):
    # Keep validation's short-lived probe files inside the test workspace.
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    return make_video(tmp_path / "draft.mp4")


def regen_payload(raw=b"video", method="ic", resolution="2k", **changes):
    return dict(request="a" * 32, mode="regen", prompt="A blue landscape", seed=42,
                video=base64.b64encode(raw).decode(), regen=dict(method=method, resolution=resolution), **changes)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("COMFY_REGEN", "1")


@pytest.fixture
def no_probe(enabled, monkeypatch):
    monkeypatch.setattr(regen, "inspect_draft", lambda raw, method: regen.Draft(raw, 120, 5., True))


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", "2"])
def test_disabled_rejects_regen(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("COMFY_REGEN", raising=False)
    else:
        monkeypatch.setenv("COMFY_REGEN", value)
    with pytest.raises(Invalid):
        validate_start(regen_payload())


@pytest.mark.parametrize("method", ["ic", "refine"])
@pytest.mark.parametrize("resolution", ["1080p", "2k"])
def test_regen_valid_and_fingerprint(no_probe, method, resolution):
    payload = regen_payload(method=method, resolution=resolution)
    clip = validate_start(payload)
    assert clip.regen.method == method and clip.regen.resolution == resolution
    assert clip.dimensions == regen.RESOLUTIONS[resolution]
    assert clip.regen.start_sigma == (.8 if method == "refine" else None)
    assert validate_start(payload).fingerprint() == clip.fingerprint()
    for key, value in (("prompt", "Another scene"), ("seed", 7), ("video", base64.b64encode(b"another").decode())):
        assert validate_start({**payload, key: value}).fingerprint() != clip.fingerprint()
    if method == "refine":
        explicit = {**payload, "regen": {**payload["regen"], "start_sigma": .8}}
        assert validate_start(explicit).fingerprint() == clip.fingerprint()
        explicit["regen"]["start_sigma"] = .7
        assert validate_start(explicit).fingerprint() != clip.fingerprint()


@pytest.mark.parametrize("field,value", [
    ("method", "upscale"), ("method", []), ("method", None), ("resolution", "4k"),
    ("resolution", []), ("resolution", 1080), ("extra", True), ("start_sigma", .8),
])
def test_ic_option_validation(no_probe, field, value):
    p = regen_payload()
    p["regen"][field] = value
    with pytest.raises(Invalid):
        validate_start(p)


@pytest.mark.parametrize("sigma", [.499, .951, True, None, "0.8", [], float("nan"), float("inf")])
def test_refine_invalid_sigma(no_probe, sigma):
    p = regen_payload(method="refine")
    p["regen"]["start_sigma"] = sigma
    with pytest.raises(Invalid, match="start_sigma"):
        validate_start(p)


@pytest.mark.parametrize("sigma", [.5, .8, .95])
def test_refine_sigma_endpoints(no_probe, sigma):
    p = regen_payload(method="refine")
    p["regen"]["start_sigma"] = sigma
    assert validate_start(p).regen.start_sigma == sigma


@pytest.mark.parametrize("key,value", [
    ("prompt", ""), ("prompt", "a" * 7001), ("prompt", None), ("seed", True),
    ("seed", -1), ("seed", 2147483648), ("request", "A" * 32), ("request", 1),
    ("seconds", 5), ("lab", {}), ("audio", True), ("first", "x"), ("size", "768p"),
    ("regen", None), ("regen", []), ("regen", {}), ("video", []), ("video", "!!!"),
    ("video", ""), ("video", "data:video/mp4;base64,YQ=="),
])
def test_regen_wire_validation(no_probe, key, value):
    p = regen_payload()
    p[key] = value
    with pytest.raises(Invalid):
        validate_start(p)


def test_prompt_and_video_limits(no_probe, monkeypatch):
    assert len(validate_start({**regen_payload(), "prompt": "a" * 7000}).prompt) == 7000
    # Same base64 limit as the unchanged four-image body allowance, 48 MiB decoded.
    assert regen.MAX_VIDEO_BYTES == 48 * 1024 * 1024
    monkeypatch.setattr(regen, "MAX_VIDEO_BYTES", 4)
    monkeypatch.setattr(regen, "MAX_VIDEO_BASE64", 8)
    assert validate_start(regen_payload(b"1234")).video.data == b"1234"
    with pytest.raises(Invalid, match="48 MiB"):
        validate_start(regen_payload(b"12345"))
    with pytest.raises(Invalid, match="48 MiB"):
        validate_start(regen_payload(b"1234567"))
    with pytest.raises(Invalid):
        strict_json('{"regen":{"method":"ic","method":"refine"}}')


@pytest.mark.parametrize("method,frames,valid", [
    ("ic", 112, False), ("ic", 113, True), ("ic", 124, True), ("ic", 125, False),
    ("refine", 119, False), ("refine", 120, True), ("refine", 360, True), ("refine", 361, False),
])
def test_real_mp4_frame_bounds(enabled, tmp_path, monkeypatch, method, frames, valid):
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    raw = make_video(tmp_path / "input.mp4", frames=frames)
    if valid:
        clip = validate_start(regen_payload(raw, method))
        assert clip.frames == frames and clip.video.audio
    else:
        with pytest.raises(Invalid, match="113 to 124" if method == "ic" else "5 to 15"):
            validate_start(regen_payload(raw, method))
    assert not list(tmp_path.glob("regen-probe-*"))


@pytest.mark.parametrize("method", ["ic", "refine"])
def test_30fps_read_at_24(enabled, tmp_path, monkeypatch, method):
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    clip = validate_start(regen_payload(make_video(tmp_path / "input.mp4", frames=150, fps=30, audio=False), method))
    assert clip.frames == 120 and not clip.video.audio


def test_invalid_mp4_and_probe_timeout(enabled, tmp_path, monkeypatch):
    monkeypatch.setattr(regen.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(Invalid, match="invalid MP4"):
        validate_start(regen_payload(b"not an mp4"))
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 60)
    monkeypatch.setattr(regen, "media_command", timeout)
    with pytest.raises(Invalid, match="invalid MP4"):
        validate_start(regen_payload(b"timeout"))
    assert not list(tmp_path.glob("regen-probe-*"))


@pytest.mark.parametrize("method", ["ic", "refine"])
@pytest.mark.parametrize("resolution", ["1080p", "2k"])
def test_graph_recipe(no_probe, method, resolution):
    clip = validate_start(regen_payload(method=method, resolution=resolution))
    g = build_graph(clip, "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    assert g["16"]["class_type"] == "SamplerCustomAdvanced"
    assert g["20"]["class_type"] == "SaveVideo"
    assert g["14"]["inputs"]["sampler_name"] == "euler"
    assert g["12"]["class_type"] == "BasicGuider"
    assert g["18"]["inputs"]["frame_count"] == clip.frames
    assert g["18"]["inputs"]["crop"] == "auto (1088 -> 1080, 2176 -> 2160)"
    assert g["17"]["inputs"]["keep_loaded"] is False
    assert g["17"]["inputs"]["samples"] == ["29", 0]
    assert g["29"] == dict(class_type="LTXVSeparateAVLatent", inputs=dict(av_latent=["16", 0]))
    assert g["17"]["inputs"]["vae"] == (["5", 0] if method == "ic" else ["3", 0])
    assert "audio" not in g["19"]["inputs"]  # packet-copy after render
    if method == "ic":
        assert g["1"]["inputs"]["unet_name"] == REF_DIFFUSION
        assert [(g[n]["inputs"]["lora_name"], g[n]["inputs"]["model"]) for n in ("10", "11", "24")] == [
            (HYPERFLOW, ["1", 0]), (SHARP, ["10", 0]), (IC_LORA, ["11", 0])]
        assert all(g[n]["inputs"]["strength_model"] == 1 for n in ("10", "11", "24"))
        assert g["21"]["inputs"] == dict(model=["24", 0], shift_video=12., shift_audio=3.)
        assert g["15"] == dict(class_type="OpenH3ICSigmas", inputs=dict(sampler="up3", custom_sigmas=""))
        assert g["8"]["inputs"]["ref_videos.ref_video_0"] == ["6", 0]
        assert g["8"]["inputs"]["prompt"] == ["7", 0]
        assert g["28"]["inputs"] == dict(video_latent=["27", 0], reference_latent=["8", 1], pin_audio=True)
        for n in ("8", "27"):
            assert (g[n]["inputs"]["width"], g[n]["inputs"]["height"]) == ((1920, 1088) if resolution == "1080p" else (2560, 1440))
        assert g["6"]["inputs"]["frame_count"] == 124
        assert g["6"]["inputs"]["target_width"] == 1344 and g["6"]["inputs"]["target_height"] == 768
        assert g["5"]["inputs"]["tile_size"] == 256 and g["5"]["inputs"]["tile_overlap"] == 64
        assert g["5"]["inputs"]["compile_decoder"] is False
        assert g["17"]["inputs"]["vae_name"] == FAST_IC
    else:
        assert g["10"]["class_type"] == "OpenH3RegenLora"
        assert g["10"]["inputs"]["curve_table"] == REF_DIFFUSION
        assert g["10"]["inputs"]["lora_dtype"] == "bf16"
        assert g["27"]["inputs"]["temporal_chunking"] is True
        assert g["27"]["inputs"]["resolution"].startswith(resolution)
        assert g["15"]["inputs"] == dict(steps=3, start_sigma=.8)
        assert g["21"]["inputs"]["selection.tau"] == 1.2
        assert g["21"]["inputs"]["dense_blocks"] == "0,1,49"
        assert g["28"]["inputs"]["audio_mode"] == "silence (reference)"
        assert g["17"]["inputs"]["vae_name"] == FAST_REFINE
    # Every edge is internal and every node is reachable from SaveVideo.
    visited = set()
    def visit(number):
        visited.add(number)
        for value in g[number]["inputs"].values():
            if isinstance(value, list):
                assert value[0] in g and isinstance(value[1], int)
                if value[0] not in visited:
                    visit(value[0])
    visit("20")
    assert visited == g.keys()


@pytest.mark.parametrize("name", [None, "../../draft.mp4", "b" * 32 + "/draft.mp4", "c" * 32 + "/" + "d" * 32 + ".mp4"])
def test_graph_refuses_external_video_paths(no_probe, name):
    with pytest.raises(ValueError, match="video name"):
        build_graph(validate_start(regen_payload()), "b" * 32, video=name)


def test_refine_portrait_reports_the_upstream_orientation(no_probe, monkeypatch):
    monkeypatch.setattr(regen, "inspect_draft", lambda raw, method: regen.Draft(raw, 120, 5., True, 768, 1344))
    clip = validate_start(regen_payload(method="refine", resolution="1080p"))
    assert clip.dimensions == (1080, 1920)
    graph = build_graph(clip, "b" * 32, video="b" * 32 + "/" + "c" * 32 + ".mp4")
    assert graph["27"]["inputs"]["resolution"] == "1080p (1920x1080)"
    # IC's published cover crop is deliberately always landscape.
    assert validate_start(regen_payload(method="ic", resolution="1080p")).dimensions == (1920, 1080)
