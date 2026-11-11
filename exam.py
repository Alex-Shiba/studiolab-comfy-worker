"""A look at the rented machine in its first minute, before anything heavy is installed.

Owner, 2026-11-12: «да если это быстро - до разворачивания всего сервака и скачивания моделей». A machine whose
boot could not finish anyway (too little memory or disk, a line on which the models would not arrive before the
boot deadline) is given back at once instead of after twenty-five minutes. Everything else is only measured: the
numbers travel with every state answer, and the studio keeps its own list of machines from them.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request, urlopen

GIB = 1024 ** 3
MIB = 1024 ** 2
# What a boot needs at the least; a worker group asks for more in its search, this only catches a wrong answer.
NEED = {"ram_gb": 48.0, "disk_free_gb": 110.0, "boot_download_s": 1200.0}
UPLOAD_SINK = "https://speed.cloudflare.com/__up"


def number(env, name, default):
    try:
        value = float(env.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def memory_gb(read=lambda path: Path(path).read_text(encoding="ascii")):
    """The memory this container may really use: the host's, or its own limit when that is smaller."""
    total = None
    try:
        for line in read("/proc/meminfo").splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = int(read(path).strip())
        except (OSError, ValueError):
            continue
        # An unlimited group reads as "max" or as a number far above any machine.
        if 0 < limit < 1 << 50:
            total = limit if total is None else min(total, limit)
    return None if total is None else round(total / GIB, 1)


def disk_write_mb_s(folder, megabytes=512, clock=time.monotonic):
    """Megabytes a second of a plain write that has reached the disk."""
    path = Path(folder) / ".exam-write"
    block = os.urandom(4 * MIB)
    try:
        started = clock()
        with path.open("wb") as stream:
            for _ in range(megabytes // 4):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        seconds = clock() - started
        return round(megabytes / seconds, 1) if seconds > 0 else None
    except OSError:
        return None
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def download_mb_s(url, size, *, connections=6, seconds=8.0, window=512 * MIB, opener=urlopen, clock=time.monotonic):
    """Megabytes a second of the model's own source, read the way the boot reads it: byte ranges in parallel.

    Nothing is kept: the measure costs a second of the line, and the boot's own resumable download starts clean.
    """
    count = max(1, min(connections, size // MIB or 1))
    step = size // count
    received = [0] * count
    stop = threading.Event()
    started = clock()

    def read(index):
        start = index * step
        end = min(size - 1, start + min(window, step) - 1)
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
            with opener(request, timeout=20) as response:
                if response.status != 206:
                    return
                while not stop.is_set():
                    data = response.read(MIB)
                    if not data:
                        return
                    received[index] += len(data)
        except (OSError, ValueError):
            return

    pool = ThreadPoolExecutor(max_workers=count)
    try:
        futures = [pool.submit(read, index) for index in range(count)]
        while clock() - started < seconds and not all(future.done() for future in futures):
            time.sleep(.05)
    finally:
        stop.set()
        elapsed = clock() - started
        pool.shutdown(wait=False, cancel_futures=True)
    total = sum(received)
    return round(total / MIB / elapsed, 1) if total and elapsed > 0 else None


def upload_mb_s(megabytes=16, *, url=UPLOAD_SINK, opener=urlopen, clock=time.monotonic):
    """Megabytes a second of a plain upload to a public speed sink; None when it cannot be measured (never a reason to fail)."""
    body = os.urandom(megabytes * MIB)
    try:
        request = Request(url, data=body, method="POST", headers={"Content-Type": "application/octet-stream"})
        started = clock()
        with opener(request, timeout=45) as response:
            response.read()
            if not 200 <= response.status < 300:
                return None
        seconds = clock() - started
        return round(megabytes / seconds, 1) if seconds > 0 else None
    except (OSError, ValueError):
        return None


def machine_identity(env, opener=urlopen):
    """Vast's own names for this machine, asked with the instance's key; empty when Vast does not say."""
    instance, key = env.get("CONTAINER_ID"), env.get("CONTAINER_API_KEY")
    found = {}
    if instance and key and str(instance).isdigit():
        try:
            request = Request(f"https://console.vast.ai/api/v0/instances/{instance}/",
                              headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
            with opener(request, timeout=15) as response:
                body = json.load(response)
            record = body.get("instances", body) if isinstance(body, dict) else {}
            if isinstance(record, list):
                record = record[0] if record else {}
            for name in ("machine_id", "host_id"):
                if isinstance(record.get(name), int):
                    found[name] = record[name]
            if isinstance(record.get("geolocation"), str):
                found["place"] = record["geolocation"][:60]
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    if instance and str(instance).isdigit():
        found["instance"] = int(instance)
    return found


def boot_files(data, model_root):
    """The files the boot will fetch and their bytes still missing (a file of the right size is taken as present)."""
    missing = []
    for item in data["files"]:
        path = Path(model_root) / item["path"]
        try:
            present = path.is_file() and path.stat().st_size == item["size"]
        except OSError:
            present = False
        if not present:
            missing.append(item)
    return missing


def source_url(data, item):
    repo, revision = item.get("repository", data["repository"]), item.get("revision", data["revision"])
    return f"https://huggingface.co/{repo}/resolve/{revision}/{item.get('source_path', item['path'])}"


def measure(workspace, data, model_root, *, env=None, probes=None):
    """Every number of the exam; a number that could not be taken is None."""
    env = os.environ if env is None else env
    probes = probes or {}
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    facts = dict((probes.get("identity") or machine_identity)(env))
    facts["ram_gb"] = (probes.get("memory") or memory_gb)()
    try:
        facts["disk_free_gb"] = round(shutil.disk_usage(workspace).free / GIB, 1)
    except OSError:
        facts["disk_free_gb"] = None
    facts["disk_mb_s"] = (probes.get("disk") or disk_write_mb_s)(workspace)
    missing = boot_files(data, model_root)
    facts["boot_gb"] = round(sum(item["size"] for item in missing) / GIB, 1)
    if missing:
        largest = max(missing, key=lambda item: item["size"])
        facts["down_mb_s"] = (probes.get("download") or download_mb_s)(source_url(data, largest), largest["size"])
    else:
        facts["down_mb_s"] = None
    facts["up_mb_s"] = (probes.get("upload") or upload_mb_s)()
    return facts


def verdict(facts, env=None):
    """Why this machine cannot finish a boot, or None. Only what would fail the boot anyway is a reason."""
    env = os.environ if env is None else env
    if number(env, "COMFY_EXAM", 1) == 0:
        return None
    ram, disk = facts.get("ram_gb"), facts.get("disk_free_gb")
    if ram is not None and ram < number(env, "COMFY_EXAM_RAM_GB", NEED["ram_gb"]):
        return "memory"
    if disk is not None and disk < number(env, "COMFY_EXAM_DISK_GB", NEED["disk_free_gb"]):
        return "disk"
    speed, gigabytes = facts.get("down_mb_s"), facts.get("boot_gb") or 0
    if speed is not None and gigabytes > 0:
        if gigabytes * 1024 / max(speed, .001) > number(env, "COMFY_EXAM_DOWNLOAD_S", NEED["boot_download_s"]):
            return "download"
    return None


FIELDS = ("machine_id", "host_id", "instance", "place", "ram_gb", "disk_free_gb", "disk_mb_s", "boot_gb",
          "down_mb_s", "up_mb_s", "boot_s")


def public(facts):
    """The numbers as one JSON text: Vast's ids, the place and the measures; no address, no key."""
    return json.dumps({name: facts[name] for name in FIELDS if facts.get(name) is not None}, sort_keys=True)


def marker(facts, reason=None):
    """One log line of the numbers alone."""
    return ("COMFY_EXAM_FAILED " + reason + " " if reason else "COMFY_EXAM ") + public(facts)


def from_environment(env=None):
    """The machine's numbers as the model server was given them; anything that is not a short plain value is dropped."""
    env = os.environ if env is None else env
    try:
        raw = json.loads(env.get("COMFY_MACHINE", "") or "{}")
    except ValueError:
        return {}
    if not isinstance(raw, dict):
        return {}
    kept = {}
    for name in FIELDS:
        value = raw.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == value and abs(value) < 1e12:
            kept[name] = value
        elif isinstance(value, str) and name == "place":
            kept[name] = value[:60]
    return kept
