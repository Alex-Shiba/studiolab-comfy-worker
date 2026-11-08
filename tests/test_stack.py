import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import boot


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    calls, stages = [], []
    pin = boot.manifest()["comfyui"]
    monkeypatch.setattr(boot, "WORKSPACE", tmp_path)
    monkeypatch.setattr(boot.sys, "platform", "linux")
    monkeypatch.setattr(boot.os, "umask", lambda value: None)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")

    def command(args):
        calls.append(list(map(str, args)))
        return pin["commit"] if "rev-parse" in args else ""

    monkeypatch.setattr(boot, "command", command)
    monkeypatch.setattr(boot, "download_models", lambda path: stages.append("download_models"))
    source, python, runtime = boot.prepare(stage=stages.append)
    return calls, stages, source, python, runtime


def test_cuda13_pins_and_boot_assertions(prepared, monkeypatch):
    calls, stages, _, _, runtime = prepared
    assert stages == ["install", "download", "download_models"]
    install = next(call for call in calls if "--index-url" in call)
    assert "https://download.pytorch.org/whl/cu130" in install
    assert "torch==2.14.1+cu130" in install and "torchvision==0.29.1+cu130" in install
    check = next(call[-1] for call in calls if "-c" in call and "import torch" in call[-1])
    fake = SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(cuda="13.0"),
                           cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "comfy_kitchen", SimpleNamespace())
    exec(check)
    fake.__version__ = "2.10.0+cu130"
    with pytest.raises(AssertionError):
        exec(check)
    fake.__version__ = "2.14.1+cu130"
    fake.version.cuda = "12.8"
    with pytest.raises(AssertionError):
        exec(check)
    fake.version.cuda = "13.0"
    fake.cuda.is_available = lambda: False
    with pytest.raises(AssertionError):
        exec(check)
    assert all((runtime / name).is_dir() for name in ("jobs", "temp", "user"))


def test_metadata_evidence_matches_pins_and_template():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "stack-evidence.json").read_text())
    constraints = (root / "runtime-constraints.txt").read_text()
    for package, version in (("torch", "2.14.1+cu130"), ("torchvision", "0.29.1+cu130")):
        assert f"{package}=={version}" in constraints
        assert evidence["wheels"][package]["version"] == version
        assert "cp310-cp310-manylinux_2_28_x86_64.whl" in evidence["wheels"][package]["wheel"]
    assert "comfy-kitchen==0.2.40" in constraints and "comfy-aimdo==0.6.0" in constraints
    assert evidence["docker"]["lowest_cuda"] == "13.0.2"
    assert evidence["docker"]["image"] in (root / "README.md").read_text(encoding="utf-8")
    assert evidence["source"]["node_requires_triton"] is False


def test_first_trial_filter_and_bearer_secret_documented():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    needs = json.loads((root / "needs.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in needs["conductor"]}
    assert entries["first-trial-filter"]["filter"] == {
        "vram_gb_min": 24, "compute_capability_min": 8.6, "cuda_max_good_min": 13.0,
        "disk_gb_min": 120, "fast_download": True}
    assert "auth_data" in entries["route-secret"]["need"]
    assert "bearer" in readme and "24 GB" in readme and "720p" in readme
    assert "cuda_max_good" in readme and "13.0" in readme and "8.6" in readme
    assert "120 GB" in readme and "/var/log/portal/comfyui.log" in readme
