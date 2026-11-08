"""Pinned downloads with resumable, validated parallel byte ranges."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request, urlopen

MANIFEST_PATH = Path(__file__).with_name("model-manifest.json")


def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def valid_file(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest() == digest


def download_file(url, target, size, digest, *, connections=6, opener=urlopen, sleep=time.sleep):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, size, digest):
        return target
    target.unlink(missing_ok=True)
    # Ranges depend only on the manifest size, never the available prefix.
    # Their names remain stable across interruptions and reboots.
    count = min(max(1, connections), size)
    step = (size + count - 1) // count
    segments = [(start, min(size - 1, start + step - 1),
                 target.with_name(f"{target.name}.range-{start}-{min(size - 1, start + step - 1)}"))
                for start in range(0, size, step)]
    stopped = threading.Event()

    def segment(spec):
        start, end, path = spec
        length = end - start + 1
        attempt = 0
        while not stopped.is_set():
            previous_attempt = attempt
            try:
                have = path.stat().st_size if path.exists() else 0
                if have == length:
                    return
                if have > length:
                    path.unlink()
                    have = 0
                request = Request(url, headers={"Range": f"bytes={start + have}-{end}",
                                                "Accept-Encoding": "identity"})
                with opener(request, timeout=120) as response:
                    expected = f"bytes {start + have}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("invalid byte range response")
                    remaining = length - have
                    with path.open("ab") as output:
                        while remaining:
                            if stopped.is_set():
                                return
                            data = response.read(min(8 * 1024 * 1024, remaining))
                            if not data:
                                raise OSError("short range")
                            output.write(data)
                            remaining -= len(data)
                            attempt = 0
                        if response.read(1):
                            raise ValueError("oversized range")
                return
            except ValueError:
                # A bad response must never become a trusted resume prefix.
                path.unlink(missing_ok=True)
                attempt = previous_attempt
            except OSError:
                # Includes URLError, HTTPError (429/5xx), timeouts and short reads.
                pass
            # Only the parent's boot watchdog bounds transient retries. Progress
            # resets the delay; persistent failures never grow beyond 30 seconds.
            delay = min(30, 2 ** min(attempt, 5))
            attempt = min(attempt + 1, 5)
            if sleep is time.sleep:
                stopped.wait(delay)
            else:
                sleep(delay)

    pool = ThreadPoolExecutor(max_workers=count)
    try:
        list(pool.map(segment, segments))
    except BaseException:
        # In particular, propagate the parent's boot watchdog immediately.
        # Waiting for every HTTP transfer here could postpone fatal retirement.
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    assembled = target.with_name(target.name + ".assembling")
    try:
        with assembled.open("wb") as output:
            for _, _, path in segments:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        if not valid_file(assembled, size, digest):
            for _, _, path in segments:
                path.unlink(missing_ok=True)
            raise RuntimeError("download checksum mismatch")
        assembled.replace(target)
        for _, _, path in segments:
            path.unlink(missing_ok=True)
    finally:
        assembled.unlink(missing_ok=True)
    return target


def download_models(model_root):
    data = manifest()
    for item in data["files"]:
        url = f"https://huggingface.co/{data['repository']}/resolve/{data['revision']}/{item['path']}"
        download_file(url, Path(model_root) / item["path"], item["size"], item["sha256"])
