import asyncio
import base64
import io
import uuid

from PIL import Image
import pytest

from contract import TTL_S, Invalid, save_frame, validate_start
from conftest import encoded_image, payload
from test_server import app_client, wait_state


@pytest.mark.parametrize("seconds,frames", [(5, 124), (6, 158), (7, 175), (8, 192), (9, 226), (10, 243)])
async def test_each_duration_reaches_graph_and_state(app_client, seconds, frames):
    http, manager, backend, _ = app_client
    data = payload(seconds=seconds)
    response = await http.post("/clip/start", json=data)
    job_id = (await response.json())["id"]
    assert response.status == 200
    await backend.entered.wait()
    assert manager.jobs[job_id].frames == frames
    assert validate_start(data).frames == frames
    assert 24 * seconds <= frames < 24 * seconds + 17 and frames % 17 == 5
    assert (await wait_state(http, job_id, "running"))["frames"] == frames


@pytest.mark.parametrize("value", [None, 123, "A" * 32, "a" * 31, "g" * 32, "a" * 32 + "\n"])
def test_request_must_be_lowercase_hex(value):
    with pytest.raises(Invalid):
        validate_start(payload(request=value))


async def test_idempotency_conflict_and_busy_keep_request(app_client):
    http, manager, backend, _ = app_client
    first = payload()
    results = await asyncio.gather(*(http.post("/clip/start", json=first) for _ in range(5)))
    bodies = [await r.json() for r in results]
    job_id = bodies[0]["id"]
    assert all(r.status == 200 for r in results)
    assert {b["id"] for b in bodies} == {job_id}
    await backend.entered.wait()
    queued = payload()
    queued_id = (await (await http.post("/clip/start", json=queued)).json())["id"]
    await http.post("/clip/start", json=payload())
    refused = payload()
    assert (await http.post("/clip/start", json=refused)).status == 429
    assert refused["request"] not in manager.requests
    assert manager.active == 3 and len(manager.requests) == 3
    for data, expected_id, state in ((first, job_id, "running"), (queued, queued_id, "queued")):
        response = await http.post("/clip/start", json=data)
        assert response.status == 200 and await response.json() == {"id": expected_id, "state": state}
        response = await http.post("/clip/start", json={**data, "seed": data["seed"] + 1})
        assert response.status == 409 and await response.json() == {"error": "conflict"}
    backend.release.set()
    await wait_state(http, job_id, "done")
    for _ in range(2):
        assert (await http.post("/clip/file", json={"id": job_id})).status == 200
        response = await http.post("/clip/start", json=first)
        assert await response.json() == {"id": job_id, "state": "done"}
    assert backend.order.count(job_id) == 1


@pytest.mark.parametrize("field,value", [
    ("mode", "text"), ("prompt", "changed"), ("seconds", 6), ("size", "720p"),
    ("aspect", "1:1"), ("seed", 7), ("audio", False),
    ("first", encoded_image(size=(40, 30))), ("last", encoded_image(size=(40, 30)))])
async def test_conflict_covers_every_field(app_client, field, value):
    http, _, _, _ = app_client
    original = payload(mode="frames", first=encoded_image(), last=encoded_image())
    assert (await http.post("/clip/start", json=original)).status == 200
    changed = {**original, field: value}
    if field == "mode":
        changed.pop("first")
        changed.pop("last")
    response = await http.post("/clip/start", json=changed)
    assert response.status == 409 and await response.json() == {"error": "conflict"}


@pytest.mark.parametrize("failed", [False, True])
async def test_request_lifetime_tracks_job_and_file(app_client, failed):
    http, manager, backend, now = app_client
    data = payload()
    job_id = (await (await http.post("/clip/start", json=data)).json())["id"]
    await backend.entered.wait()
    now[0] = 10000
    manager.expire()
    assert data["request"] in manager.requests
    backend.fail = failed
    backend.release.set()
    state = "failed" if failed else "done"
    await wait_state(http, job_id, state)
    now[0] += TTL_S - .01
    response = await http.post("/clip/start", json=data)
    assert await response.json() == {"id": job_id, "state": state}
    now[0] += .01
    manager.expire()
    assert data["request"] not in manager.requests
    assert job_id not in manager.jobs and not (manager.root / job_id).exists()
    response = await http.post("/clip/start", json=data)
    assert response.status == 200 and (await response.json())["id"] != job_id


async def test_first_and_last_cover_crop_to_square_png(app_client):
    http, manager, backend, _ = app_client
    # Red/blue outside the central square must be removed, without stretching.
    source = Image.new("RGB", (160, 90), "green")
    source.paste("red", (0, 0, 35, 90))
    source.paste("blue", (125, 0, 160, 90))
    stream = io.BytesIO()
    source.save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    response = await http.post("/clip/start", json=payload(mode="frames", aspect="1:1", first=encoded, last=encoded))
    job_id = (await response.json())["id"]
    await backend.entered.wait()
    files = list(manager.jobs[job_id].directory.iterdir())
    assert len(files) == 2
    for file in files:
        with Image.open(file) as image:
            assert image.format == "PNG" and image.size == (640, 640)
            assert image.getpixel((320, 320)) == (0, 128, 0)
            for x in (32, 608):
                red, green, blue = image.getpixel((x, 320))
                assert red < 5 and blue < 5 and abs(green - 128) < 5


def test_exif_transpose_precedes_cover_crop(tmp_path):
    image = Image.new("RGB", (80, 40), "red")
    image.paste("blue", (40, 0, 80, 40))
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    clip = validate_start(payload(mode="frames", first=base64.b64encode(stream.getvalue()).decode()))
    path = tmp_path / "frame.png"
    save_frame(clip.first, path, (40, 80))
    with Image.open(path) as result:
        assert result.size == (40, 80) and result.format == "PNG"
        assert result.getpixel((20, 10))[0] > 240
        assert result.getpixel((20, 70))[2] > 240
        assert result.getexif().get(274, 1) == 1
