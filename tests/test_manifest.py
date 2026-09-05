"""Tests for the 107.toml manifest (3-layer config merge)."""

from pathlib import Path

import pytest

from lazy107 import manifest
from lazy107.core.plan import RunPlan
from lazy107.manifest import entry_path, load_manifest, resolve_plan


def test_load_manifest_missing(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == {}


def test_load_manifest_flat_and_run_section(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('partition = "P1"\n[run]\ncpus = 8\n', encoding="utf-8")
    assert load_manifest(tmp_path) == {"cpus": 8}


def test_resolve_plan_defaults(tmp_path: Path) -> None:
    plan = resolve_plan(tmp_path)
    assert plan == RunPlan(entry="")
    assert plan.partition == "Students"


def test_resolve_plan_manifest_overrides_defaults(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text(
        'partition = "GPU-A100"\ngpu = 2\nunknown_key = "ignored"\n',
        encoding="utf-8",
    )
    plan = resolve_plan(tmp_path)
    assert plan.partition == "GPU-A100"
    assert plan.gpu == 2
    assert plan.cpus == 4  # untouched default


def test_resolve_plan_derived_under_manifest(tmp_path: Path) -> None:
    """Explicit manifest values win over dependency-derived defaults (GPU inference)."""
    (tmp_path / "107.toml").write_text('gpu = 0\n', encoding="utf-8")
    plan = resolve_plan(tmp_path, derived={"gpu": 1, "cpus": 4, "mem": "16G", "time": "2:00:00"})
    assert plan.gpu == 0  # manifest wins over derived
    assert plan.time == "2:00:00"  # derived fills the rest


def test_resolve_plan_global_under_manifest_over_defaults(tmp_path: Path) -> None:
    """Global config wins over defaults; the project 107.toml wins over global."""
    manifest.global_config_path().write_text(
        'account = "competition"\npartition = "P107-RTX5090"\nqos = "qos_p107-rtx5090"\n',
        encoding="utf-8",
    )
    (tmp_path / "107.toml").write_text('qos = "qos_stu_default"\n', encoding="utf-8")
    plan = resolve_plan(tmp_path)
    assert plan.partition == "P107-RTX5090"  # global wins over defaults
    assert plan.account == "competition"
    assert plan.qos == "qos_stu_default"  # project manifest wins over global


def test_load_global_invalid_file_ignored(tmp_path: Path) -> None:
    manifest.global_config_path().write_text('not valid toml [[[\n', encoding="utf-8")
    plan = resolve_plan(tmp_path)
    assert plan.partition == "Students"


def test_resolve_plan_env_over_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "107.toml").write_text('partition = "P1"\n', encoding="utf-8")
    monkeypatch.setenv("LAZY107_PARTITION", "P2")
    monkeypatch.setenv("LAZY107_GPU", "2")
    monkeypatch.setenv("LAZY107_ACCOUNT", "competition")
    plan = resolve_plan(tmp_path)
    assert plan.partition == "P2"
    assert plan.gpu == 2  # env strings coerced to int
    assert plan.account == "competition"


def test_resolve_plan_env_coerces_nodes_ntasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZY107_NODES", "2")
    monkeypatch.setenv("LAZY107_NTASKS", "2")
    plan = resolve_plan(tmp_path)
    assert plan.nodes == 2  # env strings coerced to int
    assert plan.ntasks == 2


def test_resolve_plan_manifest_nodes_and_command(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text(
        'nodes = 2\ncommand = "python -m trainer.main --config cfg.yaml"\n',
        encoding="utf-8",
    )
    plan = resolve_plan(tmp_path)
    assert plan.nodes == 2
    assert plan.command == "python -m trainer.main --config cfg.yaml"


def test_resolve_plan_array_from_manifest_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "107.toml").write_text('array = "1-5%2"\n', encoding="utf-8")
    assert resolve_plan(tmp_path).array == "1-5%2"
    monkeypatch.setenv("LAZY107_ARRAY", "0,2,4")
    assert resolve_plan(tmp_path).array == "0,2,4"  # env wins over manifest


def test_resolve_plan_flags_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "107.toml").write_text('partition = "P1"\n', encoding="utf-8")
    monkeypatch.setenv("LAZY107_PARTITION", "P2")
    plan = resolve_plan(tmp_path, flags={"partition": "P3"})
    assert plan.partition == "P3"


def test_entry_path(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    resolved = entry_path(tmp_path, "train.py")
    assert resolved == (tmp_path / "train.py").resolve()
    assert resolved.is_file()


def test_entry_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="entry not found"):
        entry_path(tmp_path, "nope.py")


def test_wire_conda_env_creates_minimal_manifest(tmp_path: Path) -> None:
    status = manifest.wire_conda_env(tmp_path, "demo")
    assert "wired" in status
    assert load_manifest(tmp_path)["conda_env"] == "demo"


def test_wire_conda_env_appends_without_clobbering(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('account = "competition"\n', encoding="utf-8")
    status = manifest.wire_conda_env(tmp_path, "demo")
    assert "wired" in status
    data = load_manifest(tmp_path)
    assert data["conda_env"] == "demo"
    assert data["account"] == "competition"


def test_wire_conda_env_replaces_empty_template_line(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('gpu = 0\nconda_env = ""\n', encoding="utf-8")
    status = manifest.wire_conda_env(tmp_path, "demo")
    assert "wired" in status
    data = load_manifest(tmp_path)
    assert data["conda_env"] == "demo"
    assert data["gpu"] == 0


def test_wire_conda_env_keeps_existing_value(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('conda_env = "other"\n', encoding="utf-8")
    status = manifest.wire_conda_env(tmp_path, "demo")
    assert "kept" in status
    assert load_manifest(tmp_path)["conda_env"] == "other"


def test_wire_entry_creates_minimal_manifest(tmp_path: Path) -> None:
    status = manifest.wire_entry(tmp_path, "train.py")
    assert "pinned" in status
    assert load_manifest(tmp_path)["entry"] == "train.py"


def test_wire_entry_appends_without_clobbering(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('account = "competition"\n', encoding="utf-8")
    status = manifest.wire_entry(tmp_path, "src/train.py")
    assert "pinned" in status
    data = load_manifest(tmp_path)
    assert data["entry"] == "src/train.py"
    assert data["account"] == "competition"


def test_wire_entry_replaces_existing_pin(tmp_path: Path) -> None:
    (tmp_path / "107.toml").write_text('entry = "old.py"\ngpu = 1\n', encoding="utf-8")
    status = manifest.wire_entry(tmp_path, "new.py")
    assert "pinned" in status
    data = load_manifest(tmp_path)
    assert data["entry"] == "new.py"
    assert data["gpu"] == 1
