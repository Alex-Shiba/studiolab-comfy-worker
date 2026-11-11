import asyncio
import json
from pathlib import Path
import re
import threading
import sys
from unittest.mock import patch

import pytest

import boot
import download
from contract import Invalid, validate_start
from conftest import payload, encoded_image
from reference_models import RegenModels
from test_lifecycle import harness


@pytest.fixture
def provision(tmp_path, monkeypatch):
    calls, stages, links = [], [], []
    monkeypatch.setattr(boot, "WORKSPACE", tmp_path)
    monkeypatch.setattr(boot.sys, "platform", "linux")
    monkeypatch.setattr(boot.os, "umask", lambda _: None)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv" if name == "uv" else name)
    monkeypatch.setattr(boot, "download_models", lambda _: stages.append("download_models"))
    monkeypatch.setattr(boot, "valid_file", lambda *a: True)
    monkeypatch.setattr(Path, "symlink_to", lambda self, target, **kw: links.append((self, target)))

    def command(args):
        args = list(map(str, args))
        calls.append(args)
        if "clone" in args:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
        if "rev-parse" in args:
            return next((p["commit"] for p in boot.manifest()["regen_nodes"] if args[2].endswith(p["name"])),
                        boot.manifest()["comfyui"]["commit"])
        if boot.REQUIREMENTS_CHECK in args:
            return "satisfied"
        return ""
    monkeypatch.setattr(boot, "command", command)
    return calls, stages, links


@pytest.mark.parametrize("switch", [None, "0", "false"])
def test_disabled_boot_download_and_modes_exact_legacy_snapshot(provision, tmp_path, monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv("COMFY_REGEN", raising=False)
    else:
        monkeypatch.setenv("COMFY_REGEN", switch)
    calls, stages, links = provision
    source, python, runtime = boot.prepare(stage=stages.append)
    legacy = json.loads(Path(__file__).with_name("fixtures").joinpath("regen-disabled.json").read_text())
    def normalize(arg):
        return str(arg).replace(str(runtime), "<runtime>").replace(str(tmp_path), "<workspace>").replace("\\", "/")
    normalized = [[normalize(arg) for arg in call] for call in calls]
    for call in normalized:
        for i, arg in enumerate(call):
            if arg.endswith("runtime-constraints.txt"):
                call[i] = "<worker>/runtime-constraints.txt"
    assert normalized == legacy["calls"]
    assert stages == legacy["stages"]
    assert [normalize(arg) for arg in boot.comfy_command(source, python, runtime)] == legacy["launch"]
    assert not links
    assert boot.comfy_environment({"A": "B"}) == {"A": "B"}
    m = download.manifest()
    assert {key: m[key] for key in legacy["manifest"]} == legacy["manifest"]
    seen = []
    monkeypatch.setattr(download, "download_file", lambda *a, **kw: seen.append((a, kw)))
    download.download_models(tmp_path)
    assert len(seen) == len(legacy["manifest"]["files"]) == 5
    assert sum(item["size"] for item in m["files"]) == 42_030_014_335
    for mode, extra in [("text", {}), ("frames", {"first": encoded_image()}), ("refs", {"refs": [encoded_image()]})]:
        assert validate_start(payload(mode=mode, **extra)).mode == mode
    for mode in ("regen", "ic", "refine", "unknown", None, ""):
        with pytest.raises(Invalid):
            validate_start(payload(mode=mode))
    with pytest.raises(RuntimeError, match="disabled"):
        download.download_regen_models(tmp_path)


def test_enabled_pinned_clone_checkout_allowlist_and_requirements(provision, monkeypatch):
    monkeypatch.setenv("COMFY_REGEN", "1")
    calls, stages, links = provision
    source, python, runtime = boot.prepare(stage=stages.append)
    pins = download.manifest()["regen_nodes"]
    assert len(pins) == len(links) == 4
    assert {p["name"] for p in pins} == {"ComfyUI-OpenH3Regenerate2KIC", "ComfyUI-LynnReal", "h3regen", "Comfyui_Minimax_h3_latent_Upscaler"}
    for p in pins:
        assert re.fullmatch(r"[0-9a-f]{40}", p["commit"])
        clone = next(c for c in calls if "clone" in c and p["repository"] in c)
        assert "--no-checkout" in clone
        assert ["git", "-C", clone[-1], "checkout", "--detach", p["commit"]] in calls
        assert ["git", "-C", clone[-1], "rev-parse", "HEAD"] in calls
        assert ["git", "-C", clone[-1], "diff", "--exit-code", "HEAD", "--"] in calls
        assert any(a.name == p["name"] and b == Path(clone[-1]) / p["subdirectory"] for a, b in links)
    checks = [c for c in calls if boot.REQUIREMENTS_CHECK in c]
    assert len(checks) == 2 and all(c[0] == str(python) for c in checks)
    assert not any("pip" in c and any("regen-sources" in x for x in c) for c in calls)
    launch = boot.comfy_command(source, python, runtime)
    assert "--lowvram" in launch and "--disable-all-custom-nodes" in launch
    assert launch[launch.index("--whitelist-custom-nodes") + 1:] == [p["name"] for p in pins]
    env = boot.comfy_environment({})
    assert env["LYNNREAL_NO_DEVICE_VAE"] == env["LYNNREAL_NO_BATCH_VAE"] == "1"
    assert env["LYNNREAL_VAE_TILE_BATCH"] == env["LYNNREAL_NO_COMPILE_VAE"] == "1"
    assert stages == ["install", "download", "download_models"]


def test_missing_requirements_installed_only_in_comfy_venv(provision, monkeypatch):
    monkeypatch.setenv("COMFY_REGEN", "1")
    original = boot.command
    checks = []
    def command(args):
        result = original(args)
        if boot.REQUIREMENTS_CHECK in args:
            checks.append(args[-1])
            return "missing" if len(checks) == 1 else "satisfied"
        return result
    monkeypatch.setattr(boot, "command", command)
    _, python, _ = boot.prepare()
    installs = [c for c in provision[0] if "pip" in c and any("regen-sources" in x for x in c)]
    assert len(installs) == 1
    assert installs[0][installs[0].index("--python") + 1] == str(python)
    assert "-c" in installs[0] and "--upgrade" not in installs[0]


@pytest.mark.parametrize("failure", ["clone", "checkout", "requirements", "asset", "pin"])
def test_install_failure_precedes_download(provision, monkeypatch, failure):
    monkeypatch.setenv("COMFY_REGEN", "1")
    original = boot.command
    def command(args):
        result = original(args)
        if any("regen-sources" in str(x) for x in args):
            if failure in args or (failure == "requirements" and boot.REQUIREMENTS_CHECK in args):
                raise RuntimeError("install failed")
            if failure == "pin" and "rev-parse" in args:
                return "wrong"
        return result
    monkeypatch.setattr(boot, "command", command)
    if failure == "asset":
        monkeypatch.setattr(boot, "valid_file", lambda *a: False)
    with pytest.raises(RuntimeError):
        boot.prepare(stage=provision[1].append)
    assert provision[1] == ["install"]


def test_regen_install_failure_reports_comfy_fatal_install(harness, provision, monkeypatch):
    import worker
    monkeypatch.setenv("COMFY_REGEN", "1")
    monkeypatch.setattr(worker, "prepare", boot.prepare)
    def fail(*args):
        raise RuntimeError("private install detail")
    monkeypatch.setattr(boot, "install_regen", fail)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1
    assert "COMFY_FATAL install RuntimeError" in harness[3].read_text()
    assert "private install" not in harness[3].read_text()
    assert provision[1] == []  # download_models never started


def test_all_regen_weights_pinned_and_downloaded_with_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_REGEN", "1")
    m = download.manifest()
    assert len(m["regen_files"]) == 8
    assert sum(x["size"] for x in m["regen_files"]) == 38_344_709_904
    for item in m["regen_files"] + m["regen_node_assets"]:
        assert re.fullmatch(r"[0-9a-f]{40}", item["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert item["repository"] and item["size"] > 0
    calls = []
    monkeypatch.setattr(download, "download_file", lambda *args, **kw: calls.append((args, kw)))
    download.download_regen_models(tmp_path)
    for item, (args, kwargs) in zip(m["regen_files"], calls, strict=True):
        assert args == (f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{item['source_path']}",
                        tmp_path / item["path"], item["size"], item["sha256"])
        assert kwargs["max_attempts"] == 5


async def test_regen_transfer_waits_for_ready_shared_progress_retry(tmp_path):
    ready = tmp_path / "ready"
    entered, release = threading.Event(), threading.Event()
    calls = []
    def downloader(root, progress, stopped):
        calls.append(root)
        progress(.4)
        entered.set()
        assert release.wait(3)
        if len(calls) == 1:
            raise OSError("test failure")
    models = RegenModels(tmp_path, ready, downloader)
    models.start()
    await asyncio.sleep(.12)
    assert not calls
    ready.touch()
    updates = []
    first = asyncio.create_task(models.wait(updates.append))
    second = asyncio.create_task(models.wait(updates.append))
    assert await asyncio.to_thread(entered.wait, 3)
    await asyncio.sleep(.12)
    assert .4 in updates
    release.set()
    errors = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(e, RuntimeError) for e in errors) and len(calls) == 1
    await models.wait(updates.append)
    assert len(calls) == 2
    await models.close()


def test_requirement_check_honors_pins_and_markers(tmp_path, monkeypatch, capsys):
    path = tmp_path / "requirements.txt"
    path.write_text("# upstream requirements\nav==19.0.2\nsafetensors>=0.8\nabsent; python_version < '3.0'\n")
    monkeypatch.setattr(sys, "argv", ["check", str(path)])
    with patch("importlib.metadata.version", side_effect=lambda name: {"av": "19.0.2", "safetensors": "0.8.0"}[name]):
        exec(boot.REQUIREMENTS_CHECK, {})
    assert capsys.readouterr().out.strip() == "satisfied"
    with patch("importlib.metadata.version", return_value="18.0.0"):
        exec(boot.REQUIREMENTS_CHECK, {})
    assert capsys.readouterr().out.strip() == "missing"


def test_network_evidence_matches_every_pin_and_lora_shape_check():
    evidence = json.loads(Path(__file__).parents[1].joinpath("regen-evidence.json").read_text())
    m = download.manifest()
    for group, evidence_group in (("regen_files", "weights"), ("regen_nodes", "nodes")):
        assert [{k: v for k, v in item.items() if k != "source"} for item in evidence[evidence_group]] == m[group]
    assert [c["checked_lora_modules"] for c in evidence["compatibility"]] == [208, 208, 258]
    assert all(not c["missing_or_mismatched"] for c in evidence["compatibility"])
    assert evidence["upstream"]["ic_sigmas"] == [0.834942, 0.696852, 0.468753, 0.0]
    assert evidence["upstream"]["refine_release_sigmas"] == [0.909375, 6 / 7, 2 / 3, 0.0]
    assert evidence["upstream"]["refine_grid_shift"] == 6.0
