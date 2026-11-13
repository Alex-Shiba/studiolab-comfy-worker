"""Lab-only super-resolution contract and local media finalization; no model imports."""

import base64
import binascii
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from regen import MAX_VIDEO_BYTES, MAX_VIDEO_BASE64, media_command, probe_video

MAX_OUTPUT_AREA = 5376 * 3072
FIT = {"2k": 2560, "4k": 3840}
MAX_FPS = 60
MAX_FRAMES = 15 * MAX_FPS
MP4_AUDIO_CODECS = frozenset({"aac", "mp3", "opus", "alac", "ac3", "eac3"})


def enabled():
    return os.environ.get("COMFY_VSR") == "1"


@dataclass(frozen=True)
class Video:
    frames: int
    duration: Fraction
    fps: Fraction
    width: int
    height: int
    audio: bool
    time_base: Fraction
    data: bytes = b""
    start: Fraction = Fraction(0)
    clip_duration: Fraction | None = None
    audio_codecs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Vsr:
    scale: int
    fit: str | None
    width: int
    height: int

    def public(self):
        return dict(scale=self.scale, fit=self.fit, width=self.width, height=self.height)


@dataclass(frozen=True)
class VsrRequest:
    request: str
    video: Video
    vsr: Vsr
    mode: str = "vsr"
    first: None = None
    last: None = None
    refs: tuple = ()

    @property
    def dimensions(self):
        return self.vsr.width, self.vsr.height

    @property
    def frames(self):
        return self.video.frames

    def fingerprint(self):
        fields = dict(mode=self.mode, vsr=self.vsr.public(), video=hashlib.sha256(self.video.data).hexdigest())
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def validate_work(frames, fps):
    from contract import Invalid
    if not 0 < fps <= MAX_FPS:
        raise Invalid("vsr frame rate must be at most 60 fps")
    if not 1 <= frames <= MAX_FRAMES:
        raise Invalid("vsr requires at most 900 frames")


def validate_audio(codecs):
    from contract import Invalid
    if any(codec not in MP4_AUDIO_CODECS for codec in codecs):
        raise Invalid("vsr audio codec cannot be copied to MP4")


def preflight(info):
    """Reject excessive or incompatible media using only container metadata."""
    from contract import Invalid
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise Invalid("video must contain one video stream")
    stream = videos[0]
    width, height = int(stream["width"]), int(stream["height"])
    if min(width, height) < 1 or max(width, height) > 1920:
        raise Invalid("vsr source long side must be at most 1920")
    frames, fps = int(stream["nb_frames"]), Fraction(stream["avg_frame_rate"])
    validate_work(frames, fps)
    duration = int(stream["duration_ts"]) * Fraction(stream["time_base"])
    container_duration = Fraction(info["format"]["duration"])
    if not 2 <= duration <= 15 or not 2 <= container_duration <= 15:
        raise Invalid("vsr requires a clip from 2 to 15 seconds")
    if frames > math.ceil(duration * MAX_FPS):
        raise Invalid("vsr frame density exceeds 60 fps")
    validate_audio(tuple(s.get("codec_name") for s in info["streams"] if s["codec_type"] == "audio"))
    return frames


def video_info(info, raw=b""):
    """Check every decoded frame timestamp, including VFR hidden by an average rate."""
    from contract import Invalid
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise Invalid("video must contain one video stream")
    stream = videos[0]
    width, height = int(stream["width"]), int(stream["height"])
    frames, fps = int(stream["nb_read_frames"]), Fraction(stream["avg_frame_rate"])
    tick = Fraction(stream["time_base"])
    validate_work(frames, fps)
    if width < 1 or height < 1 or not 0 < tick <= 1 / fps:
        raise Invalid("invalid video geometry or frame rate")
    duration = int(stream["duration_ts"]) * tick
    timestamps = [int(f["best_effort_timestamp"]) * tick for f in info["frames"] if f["media_type"] == "video"]
    if (len(timestamps) != frames or abs(duration - frames / fps) > tick
            or any(abs((stamp - timestamps[0]) - index / fps) > tick / 2 for index, stamp in enumerate(timestamps))
            or any(b <= a or abs((b - a) - 1 / fps) > tick for a, b in zip(timestamps, timestamps[1:]))):
        raise Invalid("video must have constant frame timing")
    container = info.get("format", {})
    start = timestamps[0] - Fraction(container.get("start_time", "0"))
    clip_duration = Fraction(container.get("duration", str(duration)))
    return Video(frames, duration, fps, width, height,
                 any(s["codec_type"] == "audio" for s in info["streams"]), tick, raw, start, clip_duration,
                 tuple(s.get("codec_name") for s in info["streams"] if s["codec_type"] == "audio"))


def inspect_video(raw):
    from contract import Invalid
    try:
        with tempfile.TemporaryDirectory(prefix="vsr-probe-") as directory:
            path = Path(directory) / "draft.mp4"
            path.write_bytes(raw)
            metadata = probe_video(path, metadata_only=True)
            frames = preflight(metadata)
            video = video_info(probe_video(path, timing=True, max_frames=MAX_FRAMES + 1, metadata=metadata), raw)
            if video.frames != frames:
                raise Invalid("video frame count does not match container")
        if max(video.width, video.height) > 1920:
            raise Invalid("vsr source long side must be at most 1920")
        if not 2 <= video.duration <= 15 or not 2 <= video.clip_duration <= 15:
            raise Invalid("vsr requires a clip from 2 to 15 seconds")
        return video
    except Invalid:
        raise
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, subprocess.SubprocessError):
        raise Invalid("invalid MP4 video") from None


def output_dimensions(width, height, scale, fit):
    width, height = width * scale, height * scale
    if fit and max(width, height) > FIT[fit]:
        ratio = Fraction(FIT[fit], max(width, height))
        width, height = max(2, int(width * ratio) // 2 * 2), max(2, int(height * ratio) // 2 * 2)
    return width, height


def validate_vsr(payload):
    from contract import Invalid, ID_PATTERN
    if set(payload) != {"request", "mode", "video", "vsr"}:
        raise Invalid("missing or unknown keys")
    if type(payload["request"]) is not str or not ID_PATTERN.fullmatch(payload["request"]):
        raise Invalid("request must be 32 lowercase hex characters")
    options = payload["vsr"]
    if type(options) is not dict or "scale" not in options or options.keys() - {"scale", "fit"}:
        raise Invalid("invalid vsr keys")
    scale, fit = options["scale"], options.get("fit")
    if type(scale) is not int or scale not in (2, 4):
        raise Invalid("vsr scale must be 2 or 4")
    if "fit" in options and (type(fit) is not str or fit not in FIT):
        raise Invalid("vsr fit must be 2k or 4k")
    encoded = payload["video"]
    if type(encoded) is not str or not 1 <= len(encoded) <= MAX_VIDEO_BASE64:
        raise Invalid("video must be base64 MP4 of at most 48 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise Invalid("invalid video base64") from None
    if not raw or len(raw) > MAX_VIDEO_BYTES:
        raise Invalid("video exceeds 48 MiB")
    video = inspect_video(raw)
    validate_work(video.frames, video.fps)
    validate_audio(video.audio_codecs)
    if video.width * video.height * scale * scale > MAX_OUTPUT_AREA:
        raise Invalid("vsr model output exceeds area limit")
    width, height = output_dimensions(video.width, video.height, scale, fit)
    return VsrRequest(payload["request"], video, Vsr(scale, fit, width, height))


def finishing_timeout(frames, size, *, encode=False):
    # Budget in decoded pixels: 8 Mpix/s for verification/mux, 1 Mpix/s for
    # Lanczos + slow x264. Floors cover startup and small clips; benchmark live.
    # A ceiling as well: a hung ffmpeg must not hold the worker's only queue (and a paid machine)
    # longer than ComfyUI itself may run (45 minutes) until live measurements say otherwise.
    pixels = frames * size[0] * size[1]
    return min(2700, max(600 if encode else 60, math.ceil(pixels / (1_000_000 if encode else 8_000_000))))


def finish_video(rendered, draft, video, options):
    """Replace the pack's AAC re-encode with source packets; fit in one video pass."""
    target = rendered.with_name("result.mp4")
    model_size = (video.width * options["scale"], video.height * options["scale"])

    def verify(path, size, *, final=False):
        result = video_info(probe_video(path, timing=True, max_frames=video.frames + 1,
                                       timeout=finishing_timeout(video.frames, size)))
        if ((result.width, result.height) != size or result.frames != video.frames or result.fps != video.fps
                or abs(result.duration - video.duration) > max(result.time_base, video.time_base)):
            raise RuntimeError("vsr output timing or dimensions mismatch")
        if final and (result.audio != video.audio or abs(result.start - video.start) > max(video.time_base, result.time_base)
                      or (video.clip_duration is not None and abs(result.clip_duration - video.clip_duration) > Fraction(1, 1000))):
            raise RuntimeError("vsr mux timing mismatch")

    verify(rendered, model_size)
    size = (options["width"], options["height"])
    encoding = ["-c:v", "copy"]
    if size != model_size:
        encoding = ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos", "-c:v", "libx264",
                    "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-video_track_timescale", str(video.time_base.denominator)]
    media_command([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe", "-f", "mov",
        "-itsoffset", format(float(video.start), ".12f"), "-i", str(rendered),
        "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(draft),
        "-map", "0:v:0", "-map", "1:a?", *encoding, "-c:a", "copy",
        # All source audio packets survive, including a tail beyond the last video
        # frame. The verified render already has the exact source video length.
        "-map_metadata", "-1", "-movflags", "+faststart", str(target)],
        timeout=finishing_timeout(video.frames, model_size, encode=size != model_size))
    if not target.is_file() or not target.stat().st_size:
        raise RuntimeError("vsr mux failed")
    verify(target, size, final=True)
    return target


def clear_temporary_videos(root):
    """Remove only the pack's UUID MP4 files from the worker's private temp folder."""
    if root is None:
        return
    root = Path(root)
    if root.is_symlink() or root.parent.is_symlink():
        raise RuntimeError("invalid vsr temp directory")
    for path in root.glob("*.mp4"):
        if re.fullmatch(r"[0-9a-f]{32}\.mp4", path.name) and not path.is_symlink() and path.is_file():
            path.unlink(missing_ok=True)
