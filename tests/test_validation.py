import base64
import io

from PIL import Image
import pytest

import contract
from contract import Invalid, decode_image, strict_json, validate_id, validate_start
from conftest import encoded_image, payload


@pytest.mark.parametrize("field,value", [
    ("mode", "image"), ("mode", []), ("prompt", ""), ("prompt", "x" * 7001), ("prompt", 12),
    ("seconds", True), ("seconds", 5.0), ("seconds", 4), ("seconds", 16), ("size", "1080p"), ("aspect", "4:3"),
    ("seed", True), ("seed", -1), ("seed", 2147483648), ("seed", 1.0), ("audio", 1), ("audio", "false"),
    ("graph", {}), ("first", None), ("last", encoded_image()),
])
def test_invalid_fields(field, value):
    with pytest.raises(Invalid):
        validate_start(payload(**{field: value}))


@pytest.mark.parametrize("field", list(payload()))
def test_every_required_key(field):
    data = payload()
    del data[field]
    with pytest.raises(Invalid):
        validate_start(data)


@pytest.mark.parametrize("data", [[], None, "hello", 1, {}])
def test_non_object(data):
    with pytest.raises(Invalid):
        validate_start(data)


def test_frames_and_text_rules():
    with pytest.raises(Invalid):
        validate_start(payload(mode="frames"))
    with pytest.raises(Invalid):
        validate_start(payload(mode="frames", last=encoded_image()))
    request = validate_start(payload(mode="frames", first=encoded_image("JPEG"), last=encoded_image()))
    assert request.first.extension == ".jpg" and request.last.extension == ".png"
    assert validate_start(payload(seed=0, prompt="x")).seed == 0
    assert validate_start(payload(seed=2147483647, prompt="я" * 7000)).seed == 2147483647


@pytest.mark.parametrize("value", ["!", "", "data:image/png;base64,abcd", None, 4,
                                    base64.b64encode(b"not a PNG").decode(), encoded_image("GIF"),
                                    encoded_image("WEBP")])
def test_image_decoding(value):
    with pytest.raises(Invalid):
        decode_image(value)


def test_real_decoding_truncated_pixels_and_animation():
    jpeg = base64.b64decode(encoded_image("JPEG"))
    with pytest.raises(Invalid):
        decode_image(base64.b64encode(jpeg[:-30]).decode())
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG", save_all=True,
                                       append_images=[Image.new("RGB", (8, 8), "blue")], duration=100)
    with pytest.raises(Invalid):
        decode_image(base64.b64encode(stream.getvalue()).decode())


def test_encoded_limit_before_decode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversize input reached decoder")
    monkeypatch.setattr(contract.base64, "b64decode", forbidden)
    with pytest.raises(Invalid):
        decode_image("A" * (contract.MAX_IMAGE_BASE64 + 1))


def test_decoded_and_pixel_limits(monkeypatch):
    monkeypatch.setattr(contract, "MAX_IMAGE_BYTES", 30)
    with pytest.raises(Invalid, match="12 MiB"):
        decode_image(encoded_image())
    monkeypatch.setattr(contract, "MAX_IMAGE_BYTES", 12 * 1024 * 1024)
    monkeypatch.setattr(contract, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(Invalid, match="dimensions"):
        decode_image(encoded_image())


@pytest.mark.parametrize("data", [{"id": "A" * 32}, {"id": "a" * 31}, {"id": "../x"},
                                   {"id": "a" * 32, "other": 0}, {"id": 5}, [], {}])
def test_strict_id(data):
    with pytest.raises(Invalid):
        validate_id(data)
    assert validate_id({"id": "a" * 32}) == "a" * 32


@pytest.mark.parametrize("text", ['{"id":"a","id":"b"}', '{"seed":NaN}', '{', '[Infinity]'])
def test_strict_json(text):
    with pytest.raises(Invalid):
        strict_json(text)
