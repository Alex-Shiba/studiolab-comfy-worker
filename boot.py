"""Linux-only provisioning, invoked only by worker.main on the rented machine."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from download import download_models, manifest
from graph import REQUIRED_NODES
from diagnostics import model_environment

WORKSPACE = Path("/workspace")
COMFY_PYTHON = "3.13.9"


def command(argv, **kwargs):
    # subprocess exceptions contain commands; callers report only fixed markers.
    result = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            check=True, timeout=1800, env=model_environment(os.environ), **kwargs)
    return result.stdout.decode("utf-8").strip()


def prepare(stage=lambda name: None):
    stage("install")
    if sys.platform != "linux":
        raise RuntimeError("GPU provisioning requires Linux")
    os.umask(0o077)
    pin = manifest()["comfyui"]
    root = WORKSPACE / "studiolab-comfy"
    root.mkdir(parents=True, exist_ok=True)
    source = root / "ComfyUI"
    if not source.exists():
        command(["git", "clone", "--depth", "1", "--branch", pin["tag"], pin["repository"], source])
    if command(["git", "-C", source, "rev-parse", "HEAD"]) != pin["commit"]:
        raise RuntimeError("ComfyUI commit mismatch")
    if command(["git", "-C", source, "rev-parse", pin["tag"] + "^{commit}"]) != pin["commit"]:
        raise RuntimeError("ComfyUI tag mismatch")
    command(["git", "-C", source, "diff", "--exit-code", "HEAD", "--"])
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("base image must provide uv")
    # A versioned path cannot accidentally reuse the previous Python 3.10 venv.
    environment = root / ("venv-" + COMFY_PYTHON)
    python = environment / "bin/python"
    if not python.exists():
        command([uv, "venv", "--managed-python", "--python", COMFY_PYTHON, environment])
    command([python, "-c", "import sys; assert sys.version_info[:3] == (3, 13, 9)"])
    command([uv, "pip", "install", "--python", python,
             "--index-url", "https://download.pytorch.org/whl/cu130",
             "torch==2.14.1+cu130", "torchvision==0.29.1+cu130"])
    command([uv, "pip", "install", "--python", python,
             "-r", source / "requirements.txt", "-c", Path(__file__).with_name("runtime-constraints.txt")])
    # Fail instead of starting an accidental CPU model server.
    command([python, "-c", "import torch; import comfy_kitchen; "
             "assert torch.__version__ == '2.14.1+cu130'; "
             "assert torch.version.cuda == '13.0'; assert torch.cuda.is_available(); "
             "assert comfy_kitchen.registry.is_available('cuda')"])
    stage("download")
    download_models(source / "models")
    private = root / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    # Stale payloads from an interrupted previous process are never restored.
    for old in private.glob("run-*"):
        if old.is_dir() and not old.is_symlink() and old.resolve().parent == private.resolve():
            shutil.rmtree(old)
    runtime = Path(tempfile.mkdtemp(prefix="run-", dir=private))
    for name in ("jobs", "temp", "user"):
        (runtime / name).mkdir(mode=0o700)
    return source, python, runtime


def comfy_command(source, python, runtime):
    return [str(python), str(source / "main.py"), "--listen", "127.0.0.1", "--port", "8188",
            "--input-directory", str(runtime / "jobs"), "--output-directory", str(runtime / "jobs"),
            "--temp-directory", str(runtime / "temp"), "--user-directory", str(runtime / "user"),
            "--disable-metadata", "--disable-all-custom-nodes", "--offline",
            "--preview-method", "none", "--disable-auto-launch"]


def wait_ready(url, children, *, nodes=False, timeout=600, clock=time.monotonic,
               sleep=time.sleep, opener=urlopen):
    deadline = clock() + timeout
    while clock() < deadline:
        if any(child.poll() is not None for child in children):
            raise RuntimeError("child exited before ready")
        try:
            with opener(url, timeout=5) as response:
                body = json.load(response)
                if response.status == 200 and (not nodes or REQUIRED_NODES <= body.keys()):
                    return
        except (OSError, ValueError):
            pass
        sleep(1)
    raise RuntimeError("readiness timeout")
