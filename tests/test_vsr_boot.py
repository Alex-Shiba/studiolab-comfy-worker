import asyncio
import json
from pathlib import Path
import re

import pytest

import boot
import download
from reference_models import VsrModels
from test_regen_boot import provision


@pytest.mark.parametrize("switch", [None, "0", "false"])
def test_disabled_boot_unchanged(provision, monkeypatch, tmp_path, switch):
    monkeypatch.delenv("COMFY_REGEN", raising=False)
    if switch is None:
        monkeypatch.delenv("COMFY_VSR", raising=False)
    else:
        monkeypatch.setenv("COMFY_VSR", switch)
    source, python, runtime = boot.prepare(stage=provision[1].append)
    legacy = json.loads(Path(__file__).with_name("fixtures").joinpath("regen-disabled.json").read_text())
    def normalize(arg):
        s = str(arg).replace(str(runtime), "<runtime>").replace(str(tmp_path), "<workspace>").replace("\\", "/")
        return "<worker>/runtime-constraints.txt" if s.endswith("runtime-constraints.txt") else s
    assert [[normalize(a) for a in c] for c in provision[0]] == legacy["calls"]
    assert [normalize(a) for a in boot.comfy_command(source, python, runtime)] == legacy["launch"]
    assert provision[1] == legacy["stages"] and not provision[2]
    with pytest.raises(RuntimeError, match="disabled"):
        download.download_vsr_models(tmp_path)


@pytest.fixture
def vsr_provision(provision, monkeypatch):
    monkeypatch.setenv("COMFY_VSR", "1")
    monkeypatch.setenv("COMFY_VSR_KERNELS", "fast")
    original = boot.command
    def command(args):
        result = original(args)
        if "rev-parse" in args and any("vsr-sources" in str(x) for x in args):
            return download.manifest()["vsr_nodes"][0]["commit"]
        if any("get_device_capability" in str(x) for x in args):
            return "fast"
        if boot.VSR_VERSIONS_CHECK in args:
            return "satisfied"
        return result
    monkeypatch.setattr(boot, "command", command)
    return provision


@pytest.mark.parametrize("regen", [False, True])
def test_pinned_install_and_combined_allowlist(vsr_provision, monkeypatch, regen):
    monkeypatch.setenv("COMFY_REGEN", "1" if regen else "0")
    source, python, runtime = boot.prepare()
    pins = (download.manifest()["regen_nodes"] if regen else []) + download.manifest()["vsr_nodes"]
    calls, stages, links = vsr_provision
    assert len(links) == len(pins)
    for pin in pins:
        clone = next(c for c in calls if "clone" in c and pin["repository"] in c)
        assert ["git", "-C", clone[-1], "checkout", "--detach", pin["commit"]] in calls
        assert ["git", "-C", clone[-1], "rev-parse", "HEAD"] in calls
        assert ["git", "-C", clone[-1], "diff", "--exit-code", "HEAD", "--"] in calls
    launch = boot.comfy_command(source, python, runtime)
    assert launch.count("--whitelist-custom-nodes") == 1
    assert launch[launch.index("--whitelist-custom-nodes") + 1:] == [p["name"] for p in pins]
    assert ("--lowvram" in launch) == regen
    assert any(boot.REQUIREMENTS_CHECK in c and "vsr-sources" in c[-1] and c[0] == str(python) for c in calls)
    assert stages == ["download_models"]


def test_missing_packages_use_comfy_constraints(vsr_provision, monkeypatch):
    original = boot.command
    checked = []
    def command(args):
        result = original(args)
        if boot.REQUIREMENTS_CHECK in args and "vsr-sources" in str(args[-1]):
            checked.append(args)
            return "missing" if len(checked) == 1 else "satisfied"
        return result
    monkeypatch.setattr(boot, "command", command)
    _, python, _ = boot.prepare()
    installs = [c for c in vsr_provision[0] if "pip" in c and any("vsr-sources" in x for x in c)]
    assert len(installs) == 1 and len(checked) == 2
    assert installs[0][installs[0].index("--python") + 1] == str(python)
    assert installs[0][-4:] == ["-c", str(Path(boot.__file__).with_name("runtime-constraints.txt")),
                              "-c", str(Path(boot.__file__).with_name("vsr-constraints.txt"))]


@pytest.mark.parametrize("kernels", ["fast", "exact"])
def test_install_selects_gpu_kernels(vsr_provision, monkeypatch, kernels):
    original = boot.command
    def command(args):
        result = original(args)
        if any("get_device_capability" in str(x) for x in args):
            assert any("cc >= (7, 5)" in str(x) and "cc >= (8, 0)" in str(x) for x in args)
            # Fast kernels only when comfy-kitchen really has both fused kernels the pack calls without a fallback.
            assert any("hasattr(ck, name)" in str(x) and "'window_attention'" in str(x) and "'apply_rope1'" in str(x)
                       and "and fused" in str(x) for x in args)
            return kernels
        return result
    monkeypatch.setattr(boot, "command", command)
    boot.prepare()
    assert boot.os.environ["COMFY_VSR_KERNELS"] == kernels


def test_occupied_node_path_fails(vsr_provision, tmp_path):
    path = tmp_path / "studiolab-comfy" / "ComfyUI" / "custom_nodes" / "ComfyUI-ReCaVSR"
    path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="path occupied"):
        boot.prepare()
    assert not vsr_provision[1]


def test_unsatisfied_dependencies_fail(vsr_provision, monkeypatch):
    original = boot.command
    def command(args):
        result = original(args)
        if boot.REQUIREMENTS_CHECK in args and "vsr-sources" in str(args[-1]):
            return "missing"
        return result
    monkeypatch.setattr(boot, "command", command)
    with pytest.raises(RuntimeError, match="dependencies unavailable"):
        boot.prepare()
    assert not vsr_provision[1]


@pytest.mark.parametrize("failure", ["clone", "checkout", "pin", "requirements"])
def test_install_fails_before_download(vsr_provision, monkeypatch, failure):
    original = boot.command
    def command(args):
        result = original(args)
        if any("vsr-sources" in str(x) for x in args):
            if failure in args or (failure == "requirements" and boot.REQUIREMENTS_CHECK in args):
                raise RuntimeError("private failure")
            if failure == "pin" and "rev-parse" in args:
                return "wrong"
        return result
    monkeypatch.setattr(boot, "command", command)
    with pytest.raises(RuntimeError):
        boot.prepare(stage=vsr_provision[1].append)
    assert vsr_provision[1] == ["install"]


def test_manifest_metadata_and_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_VSR", "1")
    m = download.manifest()
    assert len(m["vsr_nodes"]) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", m["vsr_nodes"][0]["commit"])
    files = m["vsr_files"]
    assert {p["path"] for p in files} == {"recavsr/" + n + ".safetensors" for n in (
        "transformer_int8_convrot", "flashdecoder", "prompt")}
    assert sum(p["size"] for p in files) == 6_499_853_216
    for p in files:
        assert re.fullmatch(r"[0-9a-f]{40}", p["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", p["sha256"])
        assert p["repository"] and p["source_path"] and p["size"] > 0
    evidence = json.loads(Path(__file__).parents[1].joinpath("vsr-evidence.json").read_text())
    assert evidence["nodes"] == m["vsr_nodes"] and evidence["weights"] == files
    assert evidence["requirements"] == ["triton", "opencv-python-headless"]
    calls = []
    monkeypatch.setattr(download, "download_file", lambda *a, **kw: calls.append((a, kw)))
    download.download_vsr_models(tmp_path)
    for p, (args, kw) in zip(files, calls, strict=True):
        assert args == (f"https://huggingface.co/{p['repository']}/resolve/{p['revision']}/{p['source_path']}",
                        tmp_path / p["path"], p["size"], p["sha256"])
        assert kw["max_attempts"] == 5


async def test_lazy_shared_transfer_and_retry(tmp_path):
    ready = tmp_path / "ready"
    calls = []
    def downloader(root, progress, cancelled):
        calls.append(root)
        progress(.5)
        if len(calls) == 1:
            raise OSError("private download failure")
    models = VsrModels(tmp_path, ready, downloader)
    assert models.task is None and not calls
    one = asyncio.create_task(models.wait(lambda _: None))
    two = asyncio.create_task(models.wait(lambda _: None))
    await asyncio.sleep(.02)
    assert not calls
    ready.touch()
    assert all(isinstance(e, RuntimeError) for e in await asyncio.gather(one, two, return_exceptions=True))
    assert len(calls) == 1
    await models.wait(lambda _: None)
    assert len(calls) == 2
    await models.close()
