"""Tests for Slurm association discovery (sacctmgr/scontrol parsing)."""

from pathlib import Path

import pytest

from lazy107.cluster import discover

ASSOC_OUT = (
    "ai1001a02||qos_p107-a100,qos_p107-rtx5090,qos_stu_default\n"
    "competition||qos_p107-a100,qos_p107-rtx5090,qos_stu_default\n"
    "stu||qos_p107-a100,qos_p107-rtx5090,qos_stu_default\n"
    "stu|students|qos_p107-a100,qos_p107-rtx5090,qos_stu_default"
)

PARTITIONS_OUT = (
    "PartitionName=P107-RTX5090\n"
    "   AllowGroups=ALL AllowAccounts=competition AllowQos=qos_p107-rtx5090\n"
    "   Default=YES MaxTime=2-00:00:00 MaxMemPerNode=192000\n"
    "\n"
    "PartitionName=P107-A100\n"
    "   AllowAccounts=competition AllowQos=qos_p107-a100\n"
    "   Default=NO\n"
    "\n"
    "PartitionName=Students\n"
    "   AllowAccounts=stu,stu001 AllowQos=qos_stu001,qos_stu_default\n"
    "   Default=NO MaxTime=01:00:00 MaxMemPerNode=184320\n"
)


def _fake_slurm(default_account: str = "competition"):
    def fake_run(argv: list[str]) -> str:
        if argv[0] == "sacctmgr" and "assoc" in argv:
            return ASSOC_OUT
        if argv[0] == "sacctmgr" and "user" in argv:
            return f"pb24061316|{default_account}\n"
        if argv[0] == "scontrol":
            return PARTITIONS_OUT
        return ""

    return fake_run


def test_slurm_associations_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    assoc = discover.slurm_associations("pb24061316")
    assert len(assoc[""]) == 3  # empty partition applies everywhere
    assert assoc["students"] == [("stu", ["qos_p107-a100", "qos_p107-rtx5090", "qos_stu_default"])]


def test_slurm_partitions_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    partitions = discover.slurm_partitions()
    assert partitions["P107-RTX5090"] == (
        {"competition"},
        {"qos_p107-rtx5090"},
        True,
        "2-00:00:00",
        "192000",
    )
    assert partitions["Students"] == (
        {"stu", "stu001"},
        {"qos_stu001", "qos_stu_default"},
        False,
        "01:00:00",
        "184320",
    )


def test_resolve_association_picks_default_account_and_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    assert discover.resolve_association("pb24061316") == (
        "competition",
        "P107-RTX5090",
        "qos_p107-rtx5090",
    )


def test_resolve_association_falls_back_to_stu(monkeypatch: pytest.MonkeyPatch) -> None:
    """With default account `stu`, P107 partitions reject it; Students allows it."""
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm(default_account="stu"))
    assert discover.resolve_association("pb24061316") == ("stu", "Students", "qos_stu_default")


def test_resolve_association_none_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", lambda argv: "")
    assert discover.resolve_association("nobody") is None


def test_submission_check_default_account_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-world failure: default account `competition` on Students."""
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    error = discover.submission_check("", "Students", "qos_stu_default", "pb24061316")
    assert error is not None
    assert "account competition is not allowed on partition Students" in error
    assert "discover" in error


def test_submission_check_valid_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    assert (
        discover.submission_check("competition", "P107-RTX5090", "qos_p107-rtx5090", "pb24061316")
        is None
    )


def test_submission_check_qos_not_assigned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    error = discover.submission_check("competition", "P107-RTX5090", "qos_stu_default", "pb24061316")
    assert error is not None
    assert "is not allowed on partition" in error


def test_submission_check_fails_open_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", lambda argv: "")
    assert discover.submission_check("", "Students", "qos_stu_default", "nobody") is None


def test_parse_duration() -> None:
    assert discover._parse_duration("2-00:00:00") == 2880
    assert discover._parse_duration("01:00:00") == 60
    assert discover._parse_duration("30:00") == 30
    assert discover._parse_duration("INFINITE") is None
    assert discover._parse_duration("") is None


def test_parse_mem() -> None:
    assert discover._parse_mem("16G") == 16384
    assert discover._parse_mem("184320") == 184320
    assert discover._parse_mem("UNLIMITED") is None
    assert discover._parse_mem("") is None


def test_submission_check_time_exceeds_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    error = discover.submission_check(
        "competition", "P107-RTX5090", "qos_p107-rtx5090", "pb24061316", time="3-00:00:00"
    )
    assert error is not None
    assert "exceeds partition P107-RTX5090 MaxTime" in error


def test_submission_check_mem_exceeds_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    error = discover.submission_check(
        "competition", "P107-RTX5090", "qos_p107-rtx5090", "pb24061316", mem="512G"
    )
    assert error is not None
    assert "exceeds partition P107-RTX5090 MaxMemPerNode" in error


def test_submission_check_within_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    assert (
        discover.submission_check(
            "competition",
            "P107-RTX5090",
            "qos_p107-rtx5090",
            "pb24061316",
            time="2:00:00",
            mem="16G",
        )
        is None
    )


def test_submission_check_fails_open_without_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """P107-A100 has no MaxTime/MaxMemPerNode in the fixture: no positive
    evidence, so oversized requests pass (fail open)."""
    monkeypatch.setattr(discover, "_slurm_run", _fake_slurm())
    assert (
        discover.submission_check(
            "competition",
            "P107-A100",
            "qos_p107-a100",
            "pb24061316",
            time="999:00:00",
            mem="9999G",
        )
        is None
    )


def test_write_global_config(tmp_path: Path) -> None:
    path = discover.write_global_config("competition", "P107-RTX5090", "qos_p107-rtx5090", tmp_path / "cfg.toml")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert 'account = "competition"' in text
    assert 'partition = "P107-RTX5090"' in text
    assert 'qos = "qos_p107-rtx5090"' in text
