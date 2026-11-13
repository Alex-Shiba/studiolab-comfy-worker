"""Round-two regressions: dependency pins and bounded pre-GPU media work."""

from dataclasses import replace
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import sys

import pytest

import boot
from contract import Invalid, validate_start
import vsr
from test_vsr import enabled, make_clip, no_probe, probe_info, vsr_payload
from test_vsr_boot import vsr_provision
from test_regen_boot import provision


@pytest.mark.parametrize("package", ["triton", "opencv-python-headless"])
@pytest.mark.parametrize("installed", ["wrong", None])
def test_exact_installed_version_check(monkeypatch, capsys, package, installed):
    constraints = Path(boot.__file__).with_name("vsr-constraints.txt")
    pins = {line.split("==")[0]: line.split("==")[1] for line in constraints.read_text().splitlines()
            if line and not line.startswith("#")}
    assert pins == {"triton": "3.8.0", "opencv-python-headless": "5.0.0.93"}
    evidence = json.loads(constraints.with_name("vsr-evidence.json").read_text())
    assert {name: value["version"] for name, value in evidence["dependency_pins"].items()} == pins

    def version(name):
        if name == package:
            if installed is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed
        return pins[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(sys, "argv", ["check", str(constraints)])
    exec(boot.VSR_VERSIONS_CHECK, {})
    assert capsys.readouterr().out.strip() == "mismatch"
    monkeypatch.setattr(importlib.metadata, "version", pins.__getitem__)
    exec(boot.VSR_VERSIONS_CHECK, {})
    assert capsys.readouterr().out.strip() == "satisfied"


@pytest.mark.parametrize("post_install", ["satisfied", "mismatch"])
def test_wrong_installed_pin_repaired_or_fails_install(vsr_provision, monkeypatch, post_install):
    monkeypatch.setenv("COMFY_REGEN", "1")
    original, checks = boot.command, []

    def command(args):
        result = original(args)
        if boot.VSR_VERSIONS_CHECK in args:
            checks.append(args)
            return "mismatch" if len(checks) == 1 else post_install
        return result

    monkeypatch.setattr(boot, "command", command)
    if post_install == "mismatch":
        with pytest.raises(RuntimeError, match="vsr installed dependency versions mismatch"):
            boot.prepare(stage=vsr_provision[1].append)
        assert vsr_provision[1] == ["install"]
    else:
        boot.prepare()
    assert len(checks) == 2
    calls = vsr_provision[0]
    installs = [c for c in calls if "pip" in c and any("vsr-sources" in x for x in c)]
    assert len(installs) == 1
    assert installs[0][-4:] == ["-c", str(Path(boot.__file__).with_name("runtime-constraints.txt")),
                              "-c", str(Path(boot.__file__).with_name("vsr-constraints.txt"))]
    assert not any("vsr-constraints.txt" in str(c) for c in calls if any("regen-sources" in x for x in c))


@pytest.mark.parametrize("changes,container,message", [
    ({"nb_frames": "1000000"}, {}, "900 frames"),
    ({"duration_ts": 480000}, {"duration": "16"}, "2 to 15"),
    ({"nb_frames": "121"}, {}, "density"),
    ({"avg_frame_rate": "61/1"}, {}, "60 fps"),
    ({"nb_frames": "N/A"}, {}, "invalid MP4"),
])
def test_preflight_refuses_before_timing_probe(enabled, monkeypatch, changes, container, message):
    info = probe_info()
    info["streams"][0].update(changes)
    info["format"].update(container)
    calls = []

    def probe(path, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"metadata_only": True}
        return info

    monkeypatch.setattr(vsr, "probe_video", probe)
    with pytest.raises(Invalid, match=message):
        validate_start(vsr_payload())
    assert len(calls) == 1


def test_actual_timing_probe_stops_at_packet_budget(tmp_path):
    path = tmp_path / "dense.mp4"
    make_clip(path, frames=1000, fps="60", size="16x16", audio=False)
    info = vsr.probe_video(path, timing=True, max_frames=901)
    assert int(info["streams"][0]["nb_frames"]) == 1000
    assert int(info["streams"][0]["nb_read_frames"]) == len(info["frames"]) == 901
    with pytest.raises(Invalid, match="900 frames"):
        vsr.video_info(info)


def test_audio_packets_do_not_exhaust_video_budget(enabled, tmp_path):
    raw = make_clip(tmp_path / "limit.mp4", frames=900, fps="60", size="16x16")
    clip = validate_start(vsr_payload(raw))
    assert clip.frames == 900 and clip.video.audio_codecs == ("aac",)


def test_false_container_count_cannot_hide_excess_frames(enabled, monkeypatch):
    metadata, decoded = probe_info(), probe_info(frames=901, fps=60)
    calls = []

    def probe(path, **kwargs):
        calls.append(kwargs)
        return decoded if kwargs.get("timing") else metadata

    monkeypatch.setattr(vsr, "probe_video", probe)
    with pytest.raises(Invalid, match="900 frames"):
        validate_start(vsr_payload())
    assert calls[1]["max_frames"] == 901


@pytest.mark.parametrize("frames,fps,accepted", [(900, 60, True), (901, 60, False), (900, 61, False)])
def test_validate_vsr_work_limit(no_probe, monkeypatch, frames, fps, accepted):
    video = validate_start(vsr_payload()).video
    monkeypatch.setattr(vsr, "inspect_video", lambda raw: replace(video, frames=frames, fps=Fraction(fps)))
    if accepted:
        assert validate_start(vsr_payload()).frames == frames
    else:
        with pytest.raises(Invalid, match="60 fps|900 frames"):
            validate_start(vsr_payload())


@pytest.mark.parametrize("fit", [None, "2k"])
def test_finishing_timeouts_scale_with_frames_and_pixels(tmp_path, monkeypatch, fit):
    video = vsr.Video(900, Fraction(15), Fraction(60), 1344, 768, False, Fraction(1, 15360))
    model_size = (5376, 3072)
    size = vsr.output_dimensions(video.width, video.height, 4, fit)
    options = vsr.Vsr(4, fit, *size).public()
    rendered, draft = tmp_path / "rendered.mp4", tmp_path / "draft.mp4"
    probes, muxes = [], []

    def probe(path, **kwargs):
        probes.append(kwargs)
        dimensions = size if path.name == "result.mp4" else model_size
        return replace(video, width=dimensions[0], height=dimensions[1])

    def mux(args, **kwargs):
        muxes.append(kwargs)
        Path(args[-1]).write_bytes(b"output")

    monkeypatch.setattr(vsr, "probe_video", probe)
    monkeypatch.setattr(vsr, "video_info", lambda info: info)
    monkeypatch.setattr(vsr, "media_command", mux)
    vsr.finish_video(rendered, draft, video, options)
    assert [p["timeout"] for p in probes] == [vsr.finishing_timeout(900, model_size), vsr.finishing_timeout(900, size)]
    assert all(p["max_frames"] == 901 and p["timeout"] > 60 for p in probes)
    assert muxes[0]["timeout"] == vsr.finishing_timeout(900, model_size, encode=fit is not None)
    assert muxes[0]["timeout"] > 600
    assert vsr.finishing_timeout(450, model_size) < vsr.finishing_timeout(900, model_size)
    assert vsr.finishing_timeout(900, (2560, 1440)) < vsr.finishing_timeout(900, model_size)


@pytest.mark.parametrize("codec", ["aac", "mp3", "opus", "alac", "ac3", "eac3", "pcm_s16le", "vorbis", None])
def test_each_audio_codec_checked_before_decode(enabled, monkeypatch, codec):
    info = probe_info()
    info["streams"] += [{"codec_type": "audio", "codec_name": "aac"},
                        {"codec_type": "audio", "codec_name": codec}]
    calls = []

    def probe(path, **kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr(vsr, "probe_video", probe)
    if codec in vsr.MP4_AUDIO_CODECS:
        assert validate_start(vsr_payload()).video.audio_codecs == ("aac", codec)
        assert len(calls) == 2
    else:
        with pytest.raises(Invalid, match="audio codec"):
            validate_start(vsr_payload())
        assert calls == [{"metadata_only": True}]
