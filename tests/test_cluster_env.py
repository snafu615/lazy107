"""Tests for the conda environment layer (GPU/CUDA integrity)."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lazy107.cluster.env import (
    env_commands,
    env_exists,
    env_name_for,
    find_conda,
    gpu_build_check,
    list_envs,
    missing_deps,
    sanitize_env_name,
)
from lazy107.core.defaults import write_requirements


def test_find_conda_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lazy107.cluster.env.shutil.which", lambda _: "/usr/bin/conda")
    assert find_conda() == "/usr/bin/conda"


def test_find_conda_fallback_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lazy107.cluster.env.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    exe = "conda.exe" if os.name == "nt" else "conda"
    (tmp_path / "miniconda3" / "bin").mkdir(parents=True)
    (tmp_path / "miniconda3" / "bin" / exe).touch()
    assert find_conda() == str(tmp_path / "miniconda3" / "bin" / exe)


def test_find_conda_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lazy107.cluster.env.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert find_conda() is None


def test_env_name_for_sanitizes(tmp_path: Path) -> None:
    project = tmp_path / "my project!"
    project.mkdir()
    assert env_name_for(project) == "my_project_"


def test_sanitize_env_name() -> None:
    assert sanitize_env_name("my env!") == "my_env_"
    assert sanitize_env_name("cifar-10.v2") == "cifar-10.v2"


def test_sanitize_env_name_rejects_empty() -> None:
    with pytest.raises(ValueError):
        sanitize_env_name("")


def test_env_name_for_rejects_empty() -> None:
    with pytest.raises(ValueError):
        env_name_for(Path(""))


def test_env_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        envs = {"envs": ["/home/u/miniconda3/envs/demo", "/home/u/miniconda3/envs/other"]}
        return SimpleNamespace(stdout=json.dumps(envs), returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert env_exists("conda", "demo") is True
    assert env_exists("conda", "missing") is False


def test_env_commands_create_only(tmp_path: Path) -> None:
    cmds = env_commands(tmp_path, "demo")
    assert cmds[0].startswith("module load miniconda/py312")
    assert "conda create -y -n demo python=3.12" in cmds
    assert len(cmds) == 2


def test_env_commands_gpu_torch_requirements(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("torch\nnumpy\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", gpu=1)
    assert any("pip install -r requirements.txt" in c for c in cmds)
    assert any("pip install torch torchvision" in c for c in cmds)


def test_env_commands_gpu_jax(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("jax\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", gpu=1)
    assert any('"jax[cuda12]"' in c for c in cmds)


def test_env_commands_gpu_environment_yml_gets_override(tmp_path: Path) -> None:
    (tmp_path / "environment.yml").write_text("name: demo\ndependencies:\n  - torch\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", gpu=1)
    assert any("conda env update -n demo -f environment.yml" in c for c in cmds)
    assert any("pip install torch torchvision" in c for c in cmds)


def test_env_commands_cpu_environment_yml(tmp_path: Path) -> None:
    (tmp_path / "environment.yml").write_text("name: demo\ndependencies:\n  - numpy\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", gpu=0)
    assert any("conda env update -n demo -f environment.yml" in c for c in cmds)
    assert not any("torch" in c for c in cmds)


def test_env_commands_cpu_requirements(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("scikit-learn\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo")
    assert any("pip install -r requirements.txt" in c for c in cmds)


def test_env_commands_cpu_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    cmds = env_commands(tmp_path, "demo")
    assert any("pip install -e ." in c for c in cmds)


def test_env_commands_both_manifests_installed(tmp_path: Path) -> None:
    (tmp_path / "environment.yml").write_text(
        "name: demo\ndependencies:\n  - numpy\n", encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text("scikit-learn\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo")
    assert any("conda env update -n demo -f environment.yml" in c for c in cmds)
    assert any("pip install -r requirements.txt" in c for c in cmds)


def test_env_commands_scanned_imports_via_requirements(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    write_requirements(tmp_path, {"torch"})
    cmds = env_commands(tmp_path, "demo", gpu=1)
    assert any("pip install -r requirements.txt" in c for c in cmds)
    assert any("pip install torch torchvision" in c for c in cmds)


def test_env_commands_requirements_pending_preview(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("import numpy\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", requirements_pending=True)
    assert any("pip install -r requirements.txt" in c for c in cmds)
    assert not (tmp_path / "requirements.txt").exists()


def test_env_commands_gpu_without_gpu_deps_stays_cpu_path(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("scikit-learn\n", encoding="utf-8")
    cmds = env_commands(tmp_path, "demo", gpu=1)
    assert not any("torch" in c for c in cmds)


def test_env_exists_timeout_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("conda", 30)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert env_exists("conda", "demo") is False


def test_list_envs_parses_names(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        out = (
            "# packages in environment at /home/u/miniconda3/envs:\n"
            "#\n"
            "base                 *  /home/u/miniconda3\n"
            "run107                  /home/u/miniconda3/envs/run107\n"
            "torch2                  /home/u/miniconda3/envs/torch2\n"
        )
        return SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert list_envs("conda") == ["base", "run107", "torch2"]


def test_list_envs_timeout_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("conda", 30)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert list_envs("conda") == []


def test_missing_deps_probe_all_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="'torch' True\n'scikit-learn' True\n", returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert missing_deps("conda", "demo", {"torch", "scikit-learn"}) == []
    assert calls[0][:4] == ["conda", "run", "-n", "demo"]
    assert "find_spec('sklearn')" in calls[0][-1]  # pip name mapped back to its import


def test_missing_deps_reports_failed_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        return SimpleNamespace(stdout="'torch' True\n'scikit-learn' False\n", returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert missing_deps("conda", "demo", {"torch", "scikit-learn"}) == ["scikit-learn"]


def test_missing_deps_skips_unmappable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        raise AssertionError("tf-keras cannot be probed; no subprocess call expected")

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert missing_deps("conda", "demo", {"tf-keras"}) == []


def test_missing_deps_probe_failure_reports_all(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert missing_deps("conda", "demo", {"torch", "numpy"}) == ["numpy", "torch"]


def test_missing_deps_timeout_reports_all(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired("conda", 30)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert missing_deps("conda", "demo", {"torch", "numpy"}) == ["numpy", "torch"]


def test_gpu_build_check_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="12.6", returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert gpu_build_check("conda", "demo") == "12.6"


def test_gpu_build_check_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert gpu_build_check("conda", "demo") == ""


def test_gpu_build_check_torch_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert gpu_build_check("conda", "demo") is None


def test_gpu_build_check_timeout_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("conda", 30)

    monkeypatch.setattr("lazy107.cluster.env.subprocess.run", fake_run)
    assert gpu_build_check("conda", "demo") is None
