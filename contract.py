"""Strict wire validation. This module never imports a model or the Vast SDK."""

import base64
import binascii
from dataclasses import dataclass
import io
import hashlib
import json
import re
import warnings

from PIL import Image, ImageOps

FPS = 24
FRAMES = {seconds: 24 * seconds + (5 - 24 * seconds) % 17 for seconds in range(5, 11)}
SIZES = {
    "480p": {"16:9": (864, 480), "9:16": (480, 864), "1:1": (640, 640)},
    # 720 is halfway between grid points; choose the lower one consistently.
    "720p": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960)},
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BASE64 = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_EDGE = 8192
MAX_BODY_BYTES = 2 * MAX_IMAGE_BASE64 + 65536
TTL_S = 1800
ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class Invalid(ValueError):
    """Only fixed, non-user-controlled messages may cross the HTTP boundary."""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Invalid("duplicate key")
            result[key] = value
        return result

    def constant(_):
        raise Invalid("invalid JSON")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, RecursionError):
        raise Invalid("invalid JSON") from None


@dataclass(frozen=True)
class InputImage:
    data: bytes
    extension: str


@dataclass(frozen=True)
class ClipRequest:
    request: str
    mode: str
    prompt: str
    seconds: int
    size: str
    aspect: str
    seed: int
    audio: bool
    first: InputImage | None = None
    last: InputImage | None = None

    @property
    def dimensions(self):
        return SIZES[self.size][self.aspect]

    @property
    def frames(self):
        return FRAMES[self.seconds]

    def fingerprint(self):
        fields = {key: getattr(self, key) for key in
                  ("mode", "prompt", "seconds", "size", "aspect", "seed", "audio")}
        for key in ("first", "last"):
            image = getattr(self, key)
            fields[key] = hashlib.sha256(image.data).hexdigest() if image else None
        return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def save_frame(image, path, dimensions):
    with Image.open(io.BytesIO(image.data)) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        canvas = ImageOps.fit(oriented, dimensions, method=Image.Resampling.LANCZOS,
                              centering=(0.5, 0.5))
        canvas.save(path, format="PNG")


def decode_image(encoded):
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= MAX_IMAGE_BASE64:
        raise Invalid("invalid image encoding or size")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise Invalid("image exceeds 12 MiB")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                fmt = image.format
                if fmt not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                    raise Invalid("image must be a single JPEG or PNG")
                w, h = image.size
                if max(w, h) > MAX_IMAGE_EDGE or w * h > MAX_IMAGE_PIXELS:
                    raise Invalid("image dimensions exceed limit")
                image.verify()
            # verify() checks structure; load() checks compressed pixel contents.
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
        return InputImage(raw, ".jpg" if fmt == "JPEG" else ".png")
    except Invalid:
        raise
    except (ValueError, binascii.Error, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise Invalid("invalid JPEG or PNG") from None


def validate_start(payload):
    required = {"request", "mode", "prompt", "seconds", "size", "aspect", "seed", "audio"}
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - required - {"first", "last"}:
        raise Invalid("missing or unknown keys")
    if not isinstance(payload["request"], str) or not ID_PATTERN.fullmatch(payload["request"]):
        raise Invalid("request must be 32 lowercase hex characters")
    for key, options in (("mode", ("text", "frames")), ("size", tuple(SIZES)),
                         ("aspect", ("16:9", "9:16", "1:1"))):
        if type(payload[key]) is not str or payload[key] not in options:
            raise Invalid("invalid " + key)
    if type(payload["prompt"]) is not str or not 1 <= len(payload["prompt"]) <= 4000:
        raise Invalid("prompt must contain 1 to 4000 characters")
    if type(payload["seconds"]) is not int or payload["seconds"] not in FRAMES:
        raise Invalid("seconds must be an integer from 5 to 10")
    if type(payload["seed"]) is not int or not 0 <= payload["seed"] <= 2147483647:
        raise Invalid("invalid seed")
    if type(payload["audio"]) is not bool:
        raise Invalid("audio must be boolean")
    if payload["mode"] == "text" and ("first" in payload or "last" in payload):
        raise Invalid("text mode does not accept frames")
    if payload["mode"] == "frames" and "first" not in payload:
        raise Invalid("frames mode requires first")
    images = {key: decode_image(payload[key]) for key in ("first", "last") if key in payload}
    return ClipRequest(**{key: payload[key] for key in required}, **images)


def validate_id(payload):
    if (not isinstance(payload, dict) or set(payload) != {"id"}
            or not isinstance(payload["id"], str) or not ID_PATTERN.fullmatch(payload["id"])):
        raise Invalid("id must be 32 lowercase hex characters")
    return payload["id"]
