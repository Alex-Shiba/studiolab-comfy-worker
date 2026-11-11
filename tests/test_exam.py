import io
import json

import pytest

import exam
import worker

DATA = {"repository": "org/model", "revision": "abc", "files": [
    {"path": "diffusion_models/big.safetensors", "size": 20 * exam.GIB, "sha256": "x"},
    {"path": "vae/small.safetensors", "size": 2 * exam.GIB, "sha256": "y", "repository": "other/repo", "revision": "def",
     "source_path": "comfy/small.safetensors"}]}


def facts(**change):
    return {"ram_gb": 125.0, "disk_free_gb": 500.0, "disk_mb_s": 1800.0, "boot_gb": 42.0, "down_mb_s": 180.0, "up_mb_s": 60.0, **change}


def test_only_what_would_fail_the_boot_is_a_reason():
    assert exam.verdict(facts(), {}) is None
    assert exam.verdict(facts(ram_gb=31.0), {}) == "memory"
    assert exam.verdict(facts(disk_free_gb=80.0), {}) == "disk"
    # 42 GB at 30 MB/s is 24 minutes: past the twenty the boot may spend on its models.
    assert exam.verdict(facts(down_mb_s=30.0), {}) == "download"
    assert exam.verdict(facts(down_mb_s=40.0), {}) is None
    # A slow upload, a slow disk, a number that could not be taken: measured, never a reason.
    assert exam.verdict(facts(up_mb_s=.2, disk_mb_s=40.0), {}) is None
    assert exam.verdict(facts(ram_gb=None, disk_free_gb=None, down_mb_s=None), {}) is None
    # Models already on the disk: the line no longer matters.
    assert exam.verdict(facts(down_mb_s=1.0, boot_gb=0.0), {}) is None


def test_thresholds_come_from_the_template_and_the_exam_can_be_switched_off():
    assert exam.verdict(facts(ram_gb=90.0), {"COMFY_EXAM_RAM_GB": "96"}) == "memory"
    assert exam.verdict(facts(down_mb_s=30.0), {"COMFY_EXAM_DOWNLOAD_S": "1800"}) is None
    assert exam.verdict(facts(ram_gb=8.0, disk_free_gb=1.0, down_mb_s=.1), {"COMFY_EXAM": "0"}) is None
    assert exam.verdict(facts(ram_gb=31.0), {"COMFY_EXAM_RAM_GB": "oops"}) == "memory"


def test_memory_is_the_smaller_of_the_host_and_the_container_limit():
    files = {"/proc/meminfo": "MemTotal:       131072000 kB\nMemFree: 1 kB\n", "/sys/fs/cgroup/memory.max": str(64 * exam.GIB)}

    def read(path):
        if path not in files:
            raise OSError(path)
        return files[path]

    assert exam.memory_gb(read) == 64.0
    files["/sys/fs/cgroup/memory.max"] = "max\n"
    assert exam.memory_gb(read) == 125.0
    assert exam.memory_gb(lambda path: (_ for _ in ()).throw(OSError())) is None


def test_disk_write_is_measured_and_leaves_nothing(tmp_path):
    ticks = iter([10.0, 10.5])
    assert exam.disk_write_mb_s(tmp_path, megabytes=8, clock=lambda: next(ticks)) == 16.0
    assert list(tmp_path.iterdir()) == []
    assert exam.disk_write_mb_s(tmp_path / "missing" / "deeper", megabytes=4) is None


class Response:
    def __init__(self, status, body=b"", headers=None):
        self.status, self.stream, self.headers = status, io.BytesIO(body), headers or {}

    def read(self, size=-1):
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_reads_parallel_ranges_of_the_real_source_and_counts_the_bytes():
    seen = []

    def opener(request, timeout):
        seen.append((request.full_url, request.get_header("Range")))
        return Response(206, b"x" * (3 * exam.MIB))

    clock = iter([0.0] + [2.0] * 50)
    speed = exam.download_mb_s("https://huggingface.co/org/model/resolve/abc/big", 600 * exam.MIB, connections=3,
                               opener=opener, clock=lambda: next(clock))
    assert speed == 4.5  # nine megabytes over the two seconds the clock shows
    assert sorted(r for _, r in seen) == ["bytes=0-209715199", "bytes=209715200-419430399", "bytes=419430400-629145599"]
    assert {u for u, _ in seen} == {"https://huggingface.co/org/model/resolve/abc/big"}
    # A source that refuses ranges, or a dead line, gives no number, not a zero.
    assert exam.download_mb_s("https://x/y", 600 * exam.MIB, opener=lambda r, timeout: Response(200, b"page")) is None
    assert exam.download_mb_s("https://x/y", 600 * exam.MIB, opener=lambda r, timeout: (_ for _ in ()).throw(OSError())) is None


def test_upload_is_measured_and_never_raises():
    sent = {}

    def opener(request, timeout):
        sent["bytes"], sent["method"] = len(request.data), request.get_method()
        return Response(200, b"ok")

    clock = iter([5.0, 7.0])
    assert exam.upload_mb_s(megabytes=4, opener=opener, clock=lambda: next(clock)) == 2.0
    assert sent == {"bytes": 4 * exam.MIB, "method": "POST"}
    assert exam.upload_mb_s(megabytes=1, opener=lambda r, timeout: Response(503)) is None
    assert exam.upload_mb_s(megabytes=1, opener=lambda r, timeout: (_ for _ in ()).throw(OSError())) is None


def test_identity_is_asked_with_the_instance_key_and_keeps_only_ids_and_the_place():
    asked = {}

    def opener(request, timeout):
        asked["url"], asked["auth"] = request.full_url, request.get_header("Authorization")
        return Response(200, json.dumps({"instances": {"machine_id": 136645, "host_id": 370354, "geolocation": "Australia, AU",
                                                       "public_ipaddr": "203.0.113.9", "ssh_host": "secret.example", "jupyter_token": "t"}}).encode())

    env = {"CONTAINER_ID": "60261830", "CONTAINER_API_KEY": "instance-key"}
    assert exam.machine_identity(env, opener) == {"machine_id": 136645, "host_id": 370354, "place": "Australia, AU", "instance": 60261830}
    assert asked == {"url": "https://console.vast.ai/api/v0/instances/60261830/", "auth": "Bearer instance-key"}
    # No key, a refusal, a strange id: no question is asked that could fail the boot.
    assert exam.machine_identity({"CONTAINER_ID": "60261830"}, opener) == {"instance": 60261830}
    assert exam.machine_identity(env, lambda r, timeout: (_ for _ in ()).throw(OSError())) == {"instance": 60261830}
    assert exam.machine_identity({"CONTAINER_ID": "../x", "CONTAINER_API_KEY": "k"}, opener) == {}


def test_measure_probes_the_largest_missing_file_and_counts_what_the_boot_still_needs(tmp_path):
    models = tmp_path / "models"
    (models / "vae").mkdir(parents=True)
    probed = {}

    def download(url, size):
        probed["url"], probed["size"] = url, size
        return 150.0

    probes = {"identity": lambda env: {"machine_id": 7}, "memory": lambda: 125.0, "disk": lambda folder: 900.0, "download": download, "upload": lambda: 40.0}
    got = exam.measure(tmp_path / "work", DATA, models, env={}, probes=probes)
    assert got["boot_gb"] == 22.0 and got["down_mb_s"] == 150.0 and got["machine_id"] == 7 and got["up_mb_s"] == 40.0
    assert probed == {"url": "https://huggingface.co/org/model/resolve/abc/diffusion_models/big.safetensors", "size": 20 * exam.GIB}
    assert got["disk_free_gb"] > 0
    # The small file's own repository and path are used when it is the one still missing.
    assert exam.source_url(DATA, DATA["files"][1]) == "https://huggingface.co/other/repo/resolve/def/comfy/small.safetensors"


def test_the_line_and_the_state_answer_carry_numbers_only():
    all_facts = {**facts(), "machine_id": 136645, "host_id": 370354, "instance": 60261830, "place": "Australia, AU", "boot_s": 412,
                 "public_ipaddr": "203.0.113.9", "note": "x" * 500}
    line = exam.marker(all_facts)
    assert line.startswith("COMFY_EXAM {") and "203.0.113.9" not in line and "note" not in line
    assert exam.marker(all_facts, "download").startswith("COMFY_EXAM_FAILED download {")
    env = {"COMFY_MACHINE": exam.public(all_facts)}
    assert exam.from_environment(env) == {k: all_facts[k] for k in exam.FIELDS}
    # Whatever reaches the model server's environment, only short plain values of the known names pass.
    hostile = {"COMFY_MACHINE": json.dumps({"machine_id": "1; rm", "ram_gb": True, "place": "A" * 500, "down_mb_s": 1e99, "extra": {"a": 1}, "up_mb_s": 3.5})}
    assert exam.from_environment(hostile) == {"place": "A" * 60, "up_mb_s": 3.5}
    assert exam.from_environment({"COMFY_MACHINE": "not json"}) == {} and exam.from_environment({}) == {}


def test_a_machine_that_cannot_boot_is_refused_before_the_install():
    lines = []
    good = worker.examine(lines.append, measure=lambda *args, env=None: facts(), env={})
    assert good["ram_gb"] == 125.0 and lines[-1].startswith("COMFY_EXAM {")
    with pytest.raises(worker.ExamFailed, match="download"):
        worker.examine(lines.append, measure=lambda *args, env=None: facts(down_mb_s=5.0), env={})
    assert lines[-1].startswith("COMFY_EXAM_FAILED download {")
