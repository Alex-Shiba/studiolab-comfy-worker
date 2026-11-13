"""Lab-only regeneration contract and bounded local MP4 processing (no model imports)."""

import base64
import binascii
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from diagnostics import model_environment

MAX_VIDEO_BYTES = 48 * 1024 * 1024
MAX_VIDEO_BASE64 = 4 * ((MAX_VIDEO_BYTES + 2) // 3)
RESOLUTIONS = {"1080p": (1920, 1080), "2k": (2560, 1440)}


def enabled():
    return os.environ.get("COMFY_REGEN") == "1"


@dataclass(frozen=True)
class Regen:
    method: str
    resolution: str
    start_sigma: float | None = None

    def public(self):
        result = {"method": self.method, "resolution": self.resolution}
        if self.start_sigma is not None:
            result["start_sigma"] = self.start_sigma
        return result


@dataclass(frozen=True)
class Draft:
    data: bytes
    frames: int
    duration: float
    audio: bool
    width: int = 1344
    height: int = 768


@dataclass(frozen=True)
class RegenRequest:
    request: str
    prompt: str
    seed: int
    video: Draft
    regen: Regen
    mode: str = "regen"
    first: None = None
    last: None = None
    refs: tuple = ()

    @property
    def dimensions(self):
        dimensions = RESOLUTIONS[self.regen.resolution]
        if self.video.height > self.video.width:
            return dimensions[::-1]
        return dimensions

    @property
    def frames(self):
        return self.video.frames

    def fingerprint(self):
        fields = dict(mode=self.mode, prompt=self.prompt, seed=self.seed, regen=self.regen.public(),
                      video=hashlib.sha256(self.video.data).hexdigest())
        return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def media_command(args, *, timeout=60):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          check=True, timeout=timeout, env=model_environment(os.environ)).stdout


def probe_video(path, *, timing=False, metadata_only=False, max_frames=901, timeout=60, metadata=None):
    """Inspect a local MP4 only; optionally decode its video timestamps for CFR checks."""
    entries = "stream=codec_type,width,height,avg_frame_rate,nb_read_frames,duration:format=duration"
    if metadata_only:
        entries += ":stream=codec_name,nb_frames,time_base,duration_ts,start_pts:format=start_time"
    if timing:
        # Select video so audio packets cannot exhaust the read budget. One extra
        # packet detects excess frames; never trust the container count alone.
        if not isinstance(max_frames, int) or not 1 <= max_frames <= 901:
            raise ValueError("invalid timing probe frame limit")
        if metadata is None:
            metadata = probe_video(path, metadata_only=True, timeout=timeout)
        entries += ":stream=time_base,duration_ts,start_pts:format=start_time:frame=media_type,best_effort_timestamp,pkt_duration"
        entries += ":frame_side_data="
    result = json.loads(media_command([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-f", "mov",
        *([] if metadata_only else ["-count_frames"]),
        *(["-select_streams", "v:0", "-read_intervals", f"%+#{max_frames}", "-show_frames"] if timing else []),
        "-show_entries", entries, "-of", "json", str(path)], timeout=timeout))
    if timing:
        videos = iter(result["streams"])
        result["streams"] = [{**s, **next(videos, {})} if s["codec_type"] == "video" else s
                             for s in metadata["streams"]]
        result["format"] = metadata.get("format", {})
    return result


def inspect_draft(raw, method):
    # Decode/count real frames, not the untrusted container's nb_frames field. Limit
    # protocols to local files so an uploaded container cannot cause network reads.
    from contract import Invalid
    try:
        with tempfile.TemporaryDirectory(prefix="regen-probe-") as directory:
            path = Path(directory) / "draft.mp4"
            path.write_bytes(raw)
            info = probe_video(path)
        videos = [s for s in info["streams"] if s["codec_type"] == "video"]
        if len(videos) != 1:
            raise Invalid("video must contain one video stream")
        stream = videos[0]
        w, h = int(stream["width"]), int(stream["height"])
        if w < 1 or h < 1 or max(w, h) > 8192 or w * h > 16_777_216:
            raise Invalid("video dimensions exceed limit")
        count, fps = int(stream["nb_read_frames"]), Fraction(stream["avg_frame_rate"])
        if not 0 < fps <= 240 or count < 1:
            raise Invalid("invalid video frame rate")
        duration = float(stream.get("duration") or info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise Invalid("invalid video duration")
        # Match OpenH3RegenPrepareDraft's CFR sampling rules.
        frames = round(count * 24 / fps)
        if not 5 <= duration <= 15 or not 120 <= frames <= 360:
            raise Invalid("refine requires a draft from 5 to 15 seconds")
        # VFR timestamps that disagree with the recipe's count/fps math cannot
        # preserve timing through the draft node's CFR resampling.
        if abs(duration - float(count / fps)) > 1 / 24:
            raise Invalid("video must have consistent frame timing")
        return Draft(raw, int(frames), duration, any(s["codec_type"] == "audio" for s in info["streams"]), w, h)
    except Invalid:
        raise
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError):
        raise Invalid("invalid MP4 video") from None


def validate_regen(payload):
    from contract import Invalid, ID_PATTERN
    required = {"request", "mode", "prompt", "seed", "video", "regen"}
    if set(payload) != required:
        raise Invalid("missing or unknown keys")
    if type(payload["request"]) is not str or not ID_PATTERN.fullmatch(payload["request"]):
        raise Invalid("request must be 32 lowercase hex characters")
    if type(payload["prompt"]) is not str or not 1 <= len(payload["prompt"]) <= 7000:
        raise Invalid("prompt must contain 1 to 7000 characters")
    if type(payload["seed"]) is not int or not 0 <= payload["seed"] <= 2147483647:
        raise Invalid("invalid seed")
    options = payload["regen"]
    if (type(options) is not dict or not {"resolution"} <= options.keys()
            or options.keys() - {"method", "resolution", "start_sigma"}):
        raise Invalid("invalid regen keys")
    method, resolution = options.get("method", "refine"), options["resolution"]
    if type(method) is not str or method != "refine":
        raise Invalid("regen method must be refine")
    if type(resolution) is not str or resolution not in RESOLUTIONS:
        raise Invalid("regen resolution must be 1080p or 2k")
    sigma = options.get("start_sigma", .8)
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or not .5 <= sigma <= .95:
        raise Invalid("start_sigma must be a number from 0.5 to 0.95")
    sigma = float(sigma)
    encoded = payload["video"]
    if type(encoded) is not str or not 1 <= len(encoded) <= MAX_VIDEO_BASE64:
        raise Invalid("video must be base64 MP4 of at most 48 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise Invalid("invalid video base64") from None
    if not raw or len(raw) > MAX_VIDEO_BYTES:
        raise Invalid("video exceeds 48 MiB")
    draft = inspect_draft(raw, method)
    return RegenRequest(payload["request"], payload["prompt"], payload["seed"], draft,
                        Regen(method, resolution, sigma))


def finish_video(rendered, draft, duration):
    """Keep the original compressed audio packets and exact draft duration."""
    target = rendered.with_name("result.mp4")
    media_command([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
        "-i", str(rendered), "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(draft),
        "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-t", str(duration),
        "-map_metadata", "-1", "-movflags", "+faststart", str(target)])
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError("regen mux failed")
    return target
