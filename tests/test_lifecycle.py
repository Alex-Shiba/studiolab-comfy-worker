import io
from pathlib import Path
import subprocess
from unittest.mock import AsyncMock

import pytest

import boot
from diagnostics import Diagnostics, Redactor, model_environment
import self_destruct
import worker


@pytest.fixture
def harness(tmp_path, monkeypatch):
    events, children, sleeps = [], [], []
    log = tmp_path / "worker.log"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("COMFY_LOG_FILE", str(log))
    monkeypatch.setenv("COMFY_DIAGNOSTIC_LOG", str(tmp_path / "comfyui.log"))
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("CONTAINER_API_KEY", "fake-container-secret")
    monkeypatch.setenv("MASTER_TOKEN", "fake-master-secret")
    monkeypatch.setenv("OTHER_MASTER_TOKEN", "fake-other-secret")
    monkeypatch.setenv("COMFY_SELF_DESTRUCT_IDLE_S", "540")
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.BootWatchdog, "start", lambda self: events.append("watchdog"))
    monkeypatch.setattr(worker.BootWatchdog, "cancel", lambda self: None)

    class Child:
        def __init__(self, args, **kwargs):
            self.name = "proxy" if args[-1] == "--proxy" else "server" if args[-1].endswith("server.py") else "comfy"
            self.env, self.args = kwargs["env"], args
            assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.STDOUT
            assert "CONTAINER_API_KEY" not in self.env
            if self.name != "proxy":
                assert "MASTER_TOKEN" not in self.env and "OTHER_MASTER_TOKEN" not in self.env
                assert self.env["COMFY_JOB_ROOT"] == str(runtime / "jobs")
            else:
                assert self.env["MASTER_TOKEN"] == "fake-master-secret"
            self.stdout = io.StringIO(self.name + " diagnostic output\n")
            self.stopped = False
            children.append(self)
            events.append(self.name)

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True
            events.append("stop:" + self.name)

        def wait(self, timeout=None):
            return 0

    def prepare(stage):
        assert children[0].name == "proxy" and not children[0].stopped
        events.append("prepare")
        stage("install")
        stage("download")
        return tmp_path / "ComfyUI", Path("python"), runtime

    def sleep(seconds):
        sleeps.append(seconds)
        if seconds != 15:
            raise SystemExit(0)
        assert children and not children[0].stopped

    delete = AsyncMock(return_value=204)
    monkeypatch.setattr(self_destruct, "delete_instance", delete)
    monkeypatch.setattr(worker, "prepare", prepare)
    monkeypatch.setattr(worker, "wait_ready", lambda *a, **k: events.append("ready"))
    monkeypatch.setattr(worker.subprocess, "Popen", Child)
    monkeypatch.setattr(worker.time, "sleep", sleep)
    return events, children, sleeps, log, runtime, delete


def test_proxy_precedes_provisioning_and_children_have_no_keys(harness):
    events, children, _, log, runtime, delete = harness
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 0
    assert events[:8] == ["watchdog", "proxy", "prepare", "comfy", "ready", "server", "ready", "stop:server"]
    assert events[-1] == "stop:proxy"
    assert all(child.stopped for child in children) and not runtime.exists()
    assert "COMFY_READY" in log.read_text()
    assert all("COMFY_STAGE " + stage in log.read_text() for stage in ("install", "download", "comfy", "server"))
    delete.assert_not_awaited()


@pytest.mark.parametrize("stage", ["install", "download", "comfy", "server"])
def test_fatal_boot_reports_stage_waits_and_deletes_once(harness, monkeypatch, stage):
    events, children, sleeps, log, _, delete = harness
    if stage in ("install", "download"):
        def fail(stage):
            stage(failure_stage)
            raise RuntimeError("fake-container-secret private detail")
        failure_stage = stage
        monkeypatch.setattr(worker, "prepare", fail)
    else:
        def fail(url, *args, **kwargs):
            if ("8188" in url) == (stage == "comfy"):
                raise RuntimeError("private detail")
        monkeypatch.setattr(worker, "wait_ready", fail)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1
    assert f"COMFY_FATAL {stage} RuntimeError" in log.read_text()
    assert "private" not in log.read_text() and "secret" not in log.read_text()
    assert "COMFY_SELF_DESTRUCT fatal status=204" in log.read_text()
    assert sleeps == [15] and events[-1] == "stop:proxy"
    assert all(child.stopped for child in children)
    delete.assert_awaited_once_with("123", "fake-container-secret")


def test_fatal_boot_watchdog_uses_same_deletion_path(harness, monkeypatch):
    _, _, sleeps, log, _, delete = harness
    def hang(stage):
        stage("download")
        worker.BootWatchdog().expired(None, None)
    monkeypatch.setattr(worker, "prepare", hang)
    with pytest.raises(SystemExit):
        worker.main()
    assert "COMFY_FATAL download BootTimeout" in log.read_text()
    assert sleeps == [15]
    delete.assert_awaited_once()


def test_watchdog_arms_25_minutes_and_disarms(monkeypatch):
    calls = []
    monkeypatch.setattr(worker.signal, "SIGALRM", 14, raising=False)
    monkeypatch.setattr(worker.signal, "ITIMER_REAL", 0, raising=False)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: 0)
    monkeypatch.setattr(worker.signal, "setitimer", lambda *args: calls.append(args), raising=False)
    watchdog = worker.BootWatchdog()
    watchdog.start()
    watchdog.cancel()
    watchdog.cancel()
    assert calls == [(0, 1500), (0, 0)]


def test_readiness_does_not_swallow_watchdog():
    def timed_out(*args, **kwargs):
        worker.BootWatchdog().expired(None, None)
    with pytest.raises(worker.BootTimeout):
        boot.wait_ready("unused", [], opener=timed_out)


def test_download_watchdog_does_not_wait_for_pool(tmp_path, monkeypatch):
    import download
    shutdown = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def map(self, *args):
            worker.BootWatchdog().expired(None, None)
        def shutdown(self, **kwargs):
            shutdown.append(kwargs)
    monkeypatch.setattr(download, "ThreadPoolExecutor", Pool)
    with pytest.raises(worker.BootTimeout):
        download.download_file("unused", tmp_path / "model", 10, "unused")
    assert shutdown == [{"wait": False, "cancel_futures": True}]


@pytest.mark.parametrize("child", ["comfy", "server", "proxy"])
def test_crashed_child_is_fatal_and_deleted(harness, monkeypatch, child):
    _, children, _, log, _, delete = harness
    def sleep(seconds):
        if seconds == 1:
            next(c for c in children if c.name == child).stopped = True
    monkeypatch.setattr(worker.time, "sleep", sleep)
    with pytest.raises(SystemExit):
        worker.main()
    assert f"COMFY_FATAL {child} ChildProcessError" in log.read_text()
    delete.assert_awaited_once()


def test_proxy_crash_before_ready_reports_proxy(harness, monkeypatch):
    _, children, _, log, _, delete = harness
    def fail(*args, **kwargs):
        children[0].stopped = True
        raise ChildProcessError("child exited before ready")
    monkeypatch.setattr(worker, "wait_ready", fail)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    with pytest.raises(SystemExit):
        worker.main()
    assert "COMFY_FATAL proxy ChildProcessError" in log.read_text()
    delete.assert_awaited_once()


def test_missing_kitchen_cuda_emits_install_fatal(harness, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    _, _, _, log, _, delete = harness
    monkeypatch.setattr(boot, "WORKSPACE", tmp_path)
    monkeypatch.setattr(boot.sys, "platform", "linux")
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.14.1+cu130", version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True)))
    checked = []
    def unavailable(name):
        checked.append(name)
        return False
    monkeypatch.setitem(sys.modules, "comfy_kitchen", SimpleNamespace(
        registry=SimpleNamespace(is_available=unavailable)))
    def command(args):
        if "-c" in args and "import torch" in str(args[-1]):
            exec(args[-1])
        return boot.manifest()["comfyui"]["commit"] if "rev-parse" in args else ""
    monkeypatch.setattr(boot, "command", command)
    monkeypatch.setattr(boot, "download_models", lambda *a: pytest.fail("download after failed CUDA check"))
    monkeypatch.setattr(worker, "prepare", boot.prepare)
    with pytest.raises(SystemExit):
        worker.main()
    assert checked == ["cuda"]
    assert "COMFY_FATAL install AssertionError" in log.read_text()
    delete.assert_awaited_once()


def test_fatal_deletion_opt_out(harness, monkeypatch):
    *_, delete = harness
    monkeypatch.setenv("COMFY_SELF_DESTRUCT_IDLE_S", "0")
    monkeypatch.setattr(worker, "prepare", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(SystemExit):
        worker.main()
    delete.assert_not_awaited()


def test_fatal_delete_retries_to_exactly_one_success(harness, monkeypatch):
    _, children, _, _, _, delete = harness
    delete.side_effect = [OSError("fake-container-secret"), 403, 429, 503, 500, 204]
    backoff = AsyncMock()
    async def retire(log, reason):
        assert not children[0].stopped
        await self_destruct.retire(log, reason=reason, sleep=backoff)
    monkeypatch.setattr(worker, "retire", retire)
    monkeypatch.setattr(worker, "prepare", lambda **kwargs: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(SystemExit):
        worker.main()
    assert delete.await_count == 6
    assert [call.args[0] for call in backoff.await_args_list] == [1, 2, 4, 8, 16]


def test_log_disk_failure_still_deletes(harness, monkeypatch):
    *_, delete = harness
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    # The real marker fallback does not propagate filesystem errors.
    monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    with pytest.raises(SystemExit):
        worker.main()
    delete.assert_awaited_once()


def test_idle_request_is_deleted_by_parent(harness, monkeypatch):
    _, children, _, log, runtime, delete = harness
    def sleep(seconds):
        (runtime / "retire").touch()
    monkeypatch.setattr(worker.time, "sleep", sleep)
    worker.main()
    delete.assert_awaited_once_with("123", "fake-container-secret")
    assert all(child.stopped for child in children)
    assert "COMFY_SELF_DESTRUCT idle status=204" in log.read_text()


def test_rotated_diagnostics_tee_and_streaming_secret_redaction(tmp_path):
    stdout = io.StringIO()
    path = tmp_path / "comfyui.log"
    diag = Diagnostics(path, env={"MASTER_TOKEN": "top-secret", "CONTAINER_API_KEY": "container-secret"},
                       stdout=stdout, max_bytes=256)
    for _ in range(100):
        diag.pump(io.StringIO("Prompt from owner\n" + "top-secret container-secret\n"))
    diag.close()
    files = list(tmp_path.glob("comfyui.log*"))
    assert len(files) == 4 and sum(p.stat().st_size for p in files) <= 4 * 256
    assert "Prompt from owner" in stdout.getvalue() and "[REDACTED]" in stdout.getvalue()
    for text in [stdout.getvalue(), *(p.read_text() for p in files)]:
        assert "top-secret" not in text and "container-secret" not in text
    redactor = Redactor(["top-secret"])
    assert redactor.feed("top-sec") + redactor.feed("ret\n", final=True) == "[REDACTED]\n"


def test_model_environment_drops_credentials():
    assert model_environment({"CONTAINER_API_KEY": "fake", "MASTER_TOKEN": "fake", "VAST_MASTER_TOKEN": "fake",
                              "HF_TOKEN": "fake", "PATH": "path", "CUDA_VISIBLE_DEVICES": "0"}) == {
                                  "PATH": "path", "CUDA_VISIBLE_DEVICES": "0"}
