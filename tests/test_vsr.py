import base64
from dataclasses import replace
from fractions import Fraction
import json
import subprocess

import pytest

from contract import Invalid, validate_start
from graph import build_graph
import vsr


def vsr_payload(raw=b"video", scale=2, fit=None, **changes):
    options = {"scale": scale}
    if fit is not None:
        options["fit"] = fit
    return dict(request="a" * 32, mode="vsr", video=base64.b64encode(raw).decode(), vsr=options, **changes)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("COMFY_VSR", "1")


@pytest.fixture
def no_probe(enabled, monkeypatch):
    monkeypatch.setattr(vsr, "inspect_video", lambda raw: vsr.Video(
        60, Fraction(2), Fraction(30), 640, 360, True, Fraction(1, 15360), raw))


def make_clip(path, *, frames=60, fps="30", size="64x48", audio=True, tone=440):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s={size}:r={fps}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000"]
    args += ["-t", format(float(frames / Fraction(fps)), ".12f"), "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run(args + [str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
    return path.read_bytes()


@pytest.mark.parametrize("switch", [None, "", "0", "true", "yes", "2"])
def test_switch_off_is_unknown(monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv("COMFY_VSR", raising=False)
    else:
        monkeypatch.setenv("COMFY_VSR", switch)
    assert not vsr.enabled()
    for mode in ("vsr", "unknown"):
        with pytest.raises(Invalid, match="missing or unknown keys"):
            validate_start({**vsr_payload(), "mode": mode})


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("fit", [None, "2k", "4k"])
def test_contract_and_fingerprint(no_probe, scale, fit):
    request = vsr_payload(scale=scale, fit=fit)
    clip = validate_start(request)
    assert clip.mode == "vsr" and clip.frames == 60
    assert clip.vsr.public() == dict(scale=scale, fit=fit, width=640 * scale, height=360 * scale)
    assert clip.fingerprint() == validate_start(request).fingerprint()
    assert clip.fingerprint() != validate_start(vsr_payload(b"other", scale, fit)).fingerprint()
    assert clip.fingerprint() != validate_start(vsr_payload(scale=6 - scale, fit=fit)).fingerprint()
    assert clip.fingerprint() != validate_start(vsr_payload(scale=scale, fit="2k" if fit != "2k" else "4k")).fingerprint()
    assert not hasattr(clip, "seed") and not hasattr(clip, "prompt")


@pytest.mark.parametrize("key,value", [
    ("prompt", "x"), ("seed", 42), ("seconds", 5), ("lab", {}), ("audio", True), ("first", "x"),
    ("request", "A" * 32), ("request", 1), ("vsr", None), ("vsr", []), ("vsr", {}),
    ("vsr", {"scale": 2, "extra": True}), ("video", []), ("video", "!!!"), ("video", ""),
    ("video", "data:video/mp4;base64,YQ=="),
])
def test_invalid_wire_keys(no_probe, key, value):
    with pytest.raises(Invalid):
        validate_start({**vsr_payload(), key: value})


@pytest.mark.parametrize("scale", [True, False, 2.0, 0, 1, 3, 8, "2", None, []])
def test_invalid_scale(no_probe, scale):
    with pytest.raises(Invalid, match="scale"):
        validate_start(vsr_payload(scale=scale))


@pytest.mark.parametrize("fit", [None, True, 2, "1080p", "", "4K", [], {}])
def test_invalid_fit(no_probe, fit):
    with pytest.raises(Invalid, match="fit"):
        validate_start({**vsr_payload(), "vsr": {"scale": 2, "fit": fit}})


@pytest.mark.parametrize("key", ["request", "mode", "video", "vsr"])
def test_missing_keys(no_probe, key):
    p = vsr_payload()
    del p[key]
    with pytest.raises(Invalid):
        validate_start(p)


def test_size_limit_before_and_after_base64(no_probe, monkeypatch):
    assert vsr.MAX_VIDEO_BYTES == 48 * 1024 * 1024
    monkeypatch.setattr(vsr, "MAX_VIDEO_BYTES", 4)
    monkeypatch.setattr(vsr, "MAX_VIDEO_BASE64", 8)
    assert validate_start(vsr_payload(b"1234")).video.data == b"1234"
    for raw in (b"12345", b"1234567"):
        with pytest.raises(Invalid, match="48 MiB"):
            validate_start(vsr_payload(raw))


@pytest.mark.parametrize("size,scale,fit,expected", [
    ((1344, 768), 4, None, (5376, 3072)),
    ((1344, 768), 4, "2k", (2560, 1462)),
    ((768, 1344), 4, "4k", (2194, 3840)),
    ((1920, 1080), 2, "2k", (2560, 1440)),
    ((641, 359), 2, "4k", (1282, 718)),
])
def test_output_bounds_and_even_downscale(no_probe, monkeypatch, size, scale, fit, expected):
    source = validate_start(vsr_payload()).video
    monkeypatch.setattr(vsr, "inspect_video", lambda raw: replace(source, width=size[0], height=size[1]))
    assert validate_start(vsr_payload(scale=scale, fit=fit)).dimensions == expected


def test_model_area_checked_before_fit(no_probe, monkeypatch):
    source = validate_start(vsr_payload()).video
    monkeypatch.setattr(vsr, "inspect_video", lambda raw: replace(source, width=1920, height=1080))
    with pytest.raises(Invalid, match="area"):
        validate_start(vsr_payload(scale=4, fit="2k"))


def probe_info(frames=60, fps=30, width=640, height=360):
    return {"streams": [dict(codec_type="video", width=width, height=height, nb_read_frames=str(frames),
                             nb_frames=str(frames),
                             avg_frame_rate=f"{fps}/1", time_base="1/30000", duration_ts=frames * 30000 // fps)],
            "format": {"duration": str(frames / fps)},
            "frames": [dict(media_type="video", best_effort_timestamp=i * 30000 // fps) for i in range(frames)]}


@pytest.mark.parametrize("change,message", [
    (lambda x: x["streams"].append(dict(x["streams"][0])), "one video"),
    (lambda x: x["frames"][20].update(best_effort_timestamp=20500), "constant frame"),
    (lambda x: x["frames"].pop(), "constant frame"),
    (lambda x: x["streams"][0].update(duration_ts=90000), "constant frame"),
    (lambda x: x["streams"][0].update(avg_frame_rate="0/0"), "invalid MP4"),
    (lambda x: x["streams"][0].update(width=1921), "long side"),
])
def test_inspection_bounds_and_hidden_vfr(enabled, monkeypatch, change, message):
    info = probe_info()
    change(info)
    monkeypatch.setattr(vsr, "probe_video", lambda *a, **kw: info)
    with pytest.raises(Invalid, match=message):
        validate_start(vsr_payload())


@pytest.mark.parametrize("frames,valid", [(59, False), (60, True), (450, True), (451, False)])
def test_duration_bounds(enabled, monkeypatch, frames, valid):
    monkeypatch.setattr(vsr, "probe_video", lambda *a, **kw: probe_info(frames))
    if valid:
        assert validate_start(vsr_payload()).frames == frames
    else:
        with pytest.raises(Invalid, match="2 to 15"):
            validate_start(vsr_payload())


@pytest.mark.parametrize("fps,frames", [("24", 48), ("30", 60), ("30000/1001", 60), ("60", 120)])
def test_real_mp4_timing(enabled, tmp_path, monkeypatch, fps, frames):
    monkeypatch.setattr(vsr.tempfile, "tempdir", str(tmp_path))
    clip = validate_start(vsr_payload(make_clip(tmp_path / "draft.mp4", fps=fps, frames=frames)))
    assert clip.video.fps == Fraction(fps) and clip.frames == frames and clip.video.audio
    assert clip.video.duration == frames / Fraction(fps)
    assert not list(tmp_path.glob("vsr-probe-*"))


def test_invalid_mp4_and_local_protocols(enabled, tmp_path, monkeypatch):
    import regen
    monkeypatch.setattr(vsr.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(Invalid, match="invalid MP4"):
        validate_start(vsr_payload(b"not MP4"))
    calls = []
    def probe(args, **kwargs):
        calls.append(args)
        return json.dumps(probe_info()).encode()
    monkeypatch.setattr(regen, "media_command", probe)
    validate_start(vsr_payload())
    assert calls[0][calls[0].index("-protocol_whitelist") + 1] == "file,pipe"
    assert calls[0][calls[0].index("-f") + 1] == "mov"
    assert "-show_frames" not in calls[0] and "-count_frames" not in calls[0]
    assert "codec_name" in calls[0][calls[0].index("-show_entries") + 1]
    assert "-show_frames" in calls[1]
    assert calls[1][calls[1].index("-read_intervals") + 1] == "%+#901"
    assert calls[1][calls[1].index("-select_streams") + 1] == "v:0"
    assert not list(tmp_path.glob("vsr-probe-*"))


def test_graph_uses_verified_template(no_probe):
    job = "b" * 32
    video = job + "/" + "c" * 32 + ".mp4"
    graph = build_graph(validate_start(vsr_payload()), job, video=video)
    assert graph == {
        "1": dict(class_type="LoadVideo", inputs=dict(file=video)),
        "3": dict(class_type="ReCaVSRLoader", inputs=dict(transformer="transformer_int8_convrot.safetensors")),
        "16": dict(class_type="ReCaVSRUpscale", inputs=dict(model=["3", 0], video=["1", 0], scale=2.,
            color_fix="none", seed=42, decoder="flashdecoder", kernels="fast", kv_offload=False, encoder="auto")),
        "20": dict(class_type="SaveVideo", inputs={"video": ["16", 0], "filename_prefix": job + "/" + job,
                                                   "format": "mp4", "format.codec": "auto"}),
    }


@pytest.mark.parametrize("name", [None, "../../draft.mp4", "b" * 32 + "/draft.mp4", "c" * 32 + "/" + "d" * 32 + ".mp4"])
def test_graph_file_bindings(no_probe, name):
    with pytest.raises(ValueError, match="video name"):
        build_graph(validate_start(vsr_payload()), "b" * 32, video=name)


def test_graph_gpu_kernel_selection(no_probe, monkeypatch):
    monkeypatch.setenv("COMFY_VSR_KERNELS", "exact")
    assert build_graph(validate_start(vsr_payload()), "b" * 32,
                       video="b" * 32 + "/" + "c" * 32 + ".mp4")["16"]["inputs"]["kernels"] == "exact"
