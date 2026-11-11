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

from download import download_models, manifest, valid_file
from graph import REQUIRED_NODES
from diagnostics import model_environment
from regen import enabled as regen_enabled

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
    if regen_enabled():
        install_regen(source, python, uv)
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
    args = [str(python), str(source / "main.py"), "--listen", "127.0.0.1", "--port", "8188",
            "--input-directory", str(runtime / "jobs"), "--output-directory", str(runtime / "jobs"),
            "--temp-directory", str(runtime / "temp"), "--user-directory", str(runtime / "user"),
            "--disable-metadata", "--disable-all-custom-nodes", "--offline",
            "--preview-method", "none", "--disable-auto-launch"]
    if regen_enabled():
        args += ["--lowvram", "--whitelist-custom-nodes"] + [p["name"] for p in manifest()["regen_nodes"]]
    return args


def install_regen(source, python, uv):
    """Only called during the install stage on a dedicated regeneration worker."""
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError("regen requires local media tools")
    checkout_root = source.parent / "regen-sources"
    checkout_root.mkdir(exist_ok=True)
    custom = source / "custom_nodes"
    custom.mkdir(exist_ok=True)
    for pin in manifest()["regen_nodes"]:
        checkout = checkout_root / pin["name"]
        if not checkout.exists():
            command(["git", "clone", "--filter=blob:none", "--no-checkout", pin["repository"], checkout])
            command(["git", "-C", checkout, "checkout", "--detach", pin["commit"]])
        if command(["git", "-C", checkout, "rev-parse", "HEAD"]) != pin["commit"]:
            raise RuntimeError("regen node commit mismatch")
        command(["git", "-C", checkout, "diff", "--exit-code", "HEAD", "--"])
        package = checkout / pin["subdirectory"]
        for asset in manifest().get("regen_node_assets", []):
            if asset["node"] == pin["name"] and not valid_file(package / asset["path"], asset["size"], asset["sha256"]):
                raise RuntimeError("regen node asset checksum mismatch")
        if pin["requirements"]:
            requirements = package / pin["requirements"]
            # Check in the ComfyUI venv. Preserve every upstream version constraint;
            # do not reinstall av/safetensors already supplied by the pinned stack.
            missing = command([python, "-c", REQUIREMENTS_CHECK, requirements])
            if missing != "satisfied":
                command([uv, "pip", "install", "--python", python, "-r", requirements,
                         "-c", Path(__file__).with_name("runtime-constraints.txt")])
                if command([python, "-c", REQUIREMENTS_CHECK, requirements]) != "satisfied":
                    raise RuntimeError("regen node dependencies unavailable")
        link = custom / pin["name"]
        if link.is_symlink():
            if link.resolve() != package.resolve():
                raise RuntimeError("regen node path mismatch")
        elif link.exists():
            raise RuntimeError("regen node path occupied")
        else:
            link.symlink_to(package, target_is_directory=True)


REQUIREMENTS_CHECK = """
import importlib.metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement
ok = True
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.split('#', 1)[0].strip()
    if not line:
        continue
    req = Requirement(line)
    if req.marker and not req.marker.evaluate():
        continue
    try:
        ok = ok and importlib.metadata.version(req.name) in req.specifier
    except importlib.metadata.PackageNotFoundError:
        ok = False
print('satisfied' if ok else 'missing')
"""


def comfy_environment(env):
    if not regen_enabled():
        return env
    # These are LynnReal's own controls: keep whole-clip pixels in host RAM,
    # decode spatial tiles singly, and avoid the larger compile working set.
    return {**env, "LYNNREAL_NO_DEVICE_VAE": "1", "LYNNREAL_NO_BATCH_VAE": "1",
            "LYNNREAL_VAE_TILE_BATCH": "1", "LYNNREAL_NO_COMPILE_VAE": "1"}


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
