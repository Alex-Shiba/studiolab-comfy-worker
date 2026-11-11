"""Own ComfyUI, the model server and Vast proxy; start_server.sh owns public TLS."""

import asyncio
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from aiohttp import web
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from boot import WORKSPACE, comfy_command, comfy_environment, prepare, wait_ready
from contract import MAX_BODY_BYTES
from download import manifest
import exam
from diagnostics import Diagnostics, model_environment
from self_destruct import retire
from server import log_marker


def build_config(benchmark_id=None):
    benchmark_id = benchmark_id or os.environ["COMFY_BENCHMARK_ID"]
    return WorkerConfig(
        model_server_url="http://127.0.0.1", model_server_port=18001,
        model_log_file=os.environ.get("COMFY_LOG_FILE", "/var/log/portal/comfy-worker.log"),
        model_healthcheck_url="/health",
        handlers=[
            HandlerConfig(route="/clip/start", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 100.0),
            HandlerConfig(route="/clip/state", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 1.0,
                          benchmark_config=BenchmarkConfig(generator=lambda: {"id": benchmark_id},
                                                           runs=1, concurrency=1, do_warmup=False)),
            HandlerConfig(route="/clip/file", allow_parallel_requests=True, max_queue_time=30,
                          workload_calculator=lambda payload: 1.0),
        ],
        log_action_config=LogActionConfig(on_load=["COMFY_READY"], on_error=[
            "COMFY_FATAL install", "COMFY_FATAL download", "COMFY_FATAL comfy",
            "COMFY_FATAL server", "COMFY_FATAL proxy", "COMFY_FATAL"]),
    )


def bounded_route(handler):
    async def limited(request):
        # SDK 1.8.3 otherwise constructs an unlimited public aiohttp application.
        bounded = request.clone(client_max_size=MAX_BODY_BYTES + 65536)
        try:
            return await handler(bounded)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "invalid", "detail": "request exceeds size limit"}, status=400)
    return limited


def run_proxy():
    proxy = Worker(build_config())
    # SDK sets DEBUG during construction and logs full payloads at that level.
    logging.disable(logging.DEBUG)
    close_sessions(proxy.backend)
    proxy.routes = [web.post(route.path, bounded_route(route.handler)) for route in proxy.routes]
    proxy.run()


async def unknown_session(request):
    # Do not read/decompress JSON or invoke any SDK session machinery.
    return web.json_response({"error": "unknown"}, status=404)


def close_sessions(backend):
    # SDK 1.8.3 reads these attributes for both its public and webhook apps.
    for name in ("create", "get", "end", "health"):
        setattr(backend, f"session_{name}_handler", unknown_session)


class BootTimeout(RuntimeError):
    # Not OSError/TimeoutError: readiness and download retries must not swallow it.
    pass


class ExamFailed(RuntimeError):
    """The machine cannot finish a boot: it is given back before anything is installed."""


def examine(log, *, measure=exam.measure, env=None):
    """The first minute's look at the machine; returns its numbers or raises ExamFailed. The line it logs carries only numbers."""
    facts = measure(WORKSPACE, manifest(), WORKSPACE / "studiolab-comfy" / "ComfyUI" / "models", env=env)
    reason = exam.verdict(facts, env)
    log(exam.marker(facts, reason))
    if reason:
        raise ExamFailed(reason)
    return facts


class BootWatchdog:
    def __init__(self, seconds=25 * 60):
        self.seconds = seconds
        self.previous = None

    def expired(self, signum, frame):
        raise BootTimeout("boot deadline exceeded")

    def start(self):
        # Provisioning is Linux-only. SIGALRM interrupts even a blocked install/download.
        if hasattr(signal, "setitimer"):
            self.previous = signal.signal(signal.SIGALRM, self.expired)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def cancel(self):
        if self.previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
            self.previous = None


def stop_children(children):
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    for child in reversed(children):
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    children, runtime, diagnostics = [], None, None
    watchdog = BootWatchdog()
    current_stage = "install"

    def stage(name):
        nonlocal current_stage
        current_stage = name
        log_marker("COMFY_STAGE " + name)

    def launch(args, env, **kwargs):
        child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1,
                                 env=env, **kwargs)
        children.append(child)
        diagnostics.attach(child)
        return child

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        os.umask(0o077)
        log = Path(os.environ.get("COMFY_LOG_FILE", "/var/log/portal/comfy-worker.log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")
        diagnostics = Diagnostics()
        watchdog.start()
        benchmark = uuid.uuid4().hex
        proxy_env = {**os.environ, "COMFY_BENCHMARK_ID": benchmark, "PYTHONUNBUFFERED": "1"}
        proxy_env.pop("CONTAINER_API_KEY", None)
        stage("proxy")
        launch([sys.executable, str(Path(__file__)), "--proxy"], proxy_env)
        # Before the install and the tens of gigabytes of models: a machine that cannot finish the boot leaves now.
        stage("exam")
        booted = time.monotonic()
        facts = examine(log_marker)
        stage("install")
        source, python, runtime = prepare(stage=stage)
        retire_file = runtime / "retire"
        # What the studio keeps about this machine travels with every state answer: numbers and Vast's own ids only.
        facts["boot_s"] = round(time.monotonic() - booted)
        child_env = {**model_environment(os.environ), "COMFY_JOB_ROOT": str(runtime / "jobs"),
                     "COMFY_MODEL_ROOT": str(source / "models"), "COMFY_READY_FILE": str(runtime / "ready"),
                     "COMFY_RETIRE_FILE": str(retire_file), "COMFY_BENCHMARK_ID": benchmark,
                     "COMFY_MACHINE": exam.public(facts), "PYTHONUNBUFFERED": "1"}
        stage("comfy")
        comfy = launch(comfy_command(source, python, runtime), comfy_environment(child_env), cwd=source)
        wait_ready("http://127.0.0.1:8188/object_info", children, nodes=True)
        stage("server")
        launch([sys.executable, str(Path(__file__).with_name("server.py"))], child_env)
        wait_ready("http://127.0.0.1:18001/health", children, timeout=60)
        log_marker("COMFY_READY")
        (runtime / "ready").touch(mode=0o600)
        watchdog.cancel()
        while True:
            for child in children:
                if child.poll() is not None:
                    current_stage = "proxy" if child is children[0] else "comfy" if child is comfy else "server"
                    raise ChildProcessError("child exited")
            if retire_file.exists():
                asyncio.run(retire(log_marker, reason="idle"))
                return
            time.sleep(1)
    except Exception as error:
        watchdog.cancel()
        if children and children[0].poll() is not None:
            current_stage = "proxy"
        log_marker(f"COMFY_FATAL {current_stage} {type(error).__name__}")
        # Stop generation, but leave the proxy alive through its 10 s metrics cadence.
        stop_children(children[1:])
        time.sleep(15)
        asyncio.run(retire(log_marker, reason="fatal"))
        raise SystemExit(1) from None
    finally:
        watchdog.cancel()
        stop_children(children)
        if diagnostics is not None:
            diagnostics.close()
        if runtime is not None:
            shutil.rmtree(runtime, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--proxy"]:
        run_proxy()
    else:
        main()
