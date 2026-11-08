import hashlib
import io
import re
import threading

import pytest

from download import download_file, manifest, valid_file
from graph import AUDIO_VAE, DIFFUSION, LORA, TEXT_ENCODER, VIDEO_VAE


class Response(io.BytesIO):
    def __init__(self, content, start, end, size, status=206):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{size}"}


def test_manifest_complete_and_only_turbo8():
    data = manifest()
    assert data["repository"] == "Comfy-Org/MiniMax-H3"
    assert re.fullmatch("[0-9a-f]{40}", data["revision"])
    assert data["comfyui"]["tag"] == "v0.44.0"
    assert re.fullmatch("[0-9a-f]{40}", data["comfyui"]["commit"])
    assert len(data["files"]) == 5
    assert {f["path"].split("/")[-1] for f in data["files"]} == {DIFFUSION, VIDEO_VAE, AUDIO_VAE, TEXT_ENCODER, LORA}
    assert sum(f["size"] for f in data["files"]) == 42030035159
    for item in data["files"]:
        assert type(item["size"]) is int and item["size"] > 0
        assert re.fullmatch("[0-9a-f]{64}", item["sha256"])
        assert ".." not in item["path"] and not item["path"].startswith("/")


def test_parallel_resume_integrity_and_second_boot(tmp_path):
    data = b"abcdefghijklmnopqrstuvwxyz012345"
    path = tmp_path / "model"
    path.with_name("model.range-0-15").write_bytes(data[:5])
    calls, threads = [], set()
    barrier = threading.Barrier(2)

    def opener(request, timeout):
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        calls.append((start, end))
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return Response(data[start:end + 1], start, end, len(data))

    digest = hashlib.sha256(data).hexdigest()
    download_file("https://example.invalid/pinned/model", path, len(data), digest, connections=2, opener=opener)
    assert path.read_bytes() == data and set(calls) == {(5, 15), (16, 31)} and len(threads) == 2
    assert list(tmp_path.iterdir()) == [path]
    download_file("unused", path, len(data), digest, opener=lambda *a, **k: pytest.fail("cached model fetched"))
    assert valid_file(path, len(data), digest)


@pytest.mark.parametrize("bad", ["range", "checksum", "oversized"])
def test_bad_download_is_never_installed(tmp_path, bad):
    data = b"good"

    def opener(request, timeout):
        body = b"evil" if bad == "checksum" else data + b"x" if bad == "oversized" else data
        return Response(body, 0, 3, 4, status=200 if bad == "range" else 206)

    path = tmp_path / "model"
    with pytest.raises(RuntimeError):
        download_file("https://example.invalid/model", path, 4, hashlib.sha256(data).hexdigest(), connections=1, opener=opener)
    assert not path.exists() and not list(tmp_path.glob("*.range-*"))


def test_partial_network_failure_resumes(tmp_path):
    data = b"abcdef"
    calls = []

    def opener(request, timeout):
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        calls.append(start)
        return Response(data[:2] if len(calls) == 1 else data[start:], start, end, len(data))

    path = tmp_path / "model"
    download_file("https://example.invalid/model", path, 6, hashlib.sha256(data).hexdigest(), connections=1,
                  opener=opener, sleep=lambda _: None)
    assert calls == [0, 2] and path.read_bytes() == data


def test_same_size_corrupt_cache_is_rechecked(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"evil")
    download_file("https://example.invalid/model", path, 4, hashlib.sha256(b"good").hexdigest(), connections=1,
                  opener=lambda *a, **k: Response(b"good", 0, 3, 4))
    assert path.read_bytes() == b"good"
