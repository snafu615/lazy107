"""Tests for workflow-state checks and the check dashboard."""

from lazy107.core.workflow import workflow_checks, workflow_status


def test_wired_env_missing_is_a_blocking_error() -> None:
    errors, warnings = workflow_checks("proj", "proj", set(), lambda name: False)
    assert len(errors) == 1
    assert "not created" in errors[0]
    assert warnings == []


def test_wired_env_present_is_clean() -> None:
    assert workflow_checks("proj", "proj", {"torch"}, lambda name: True) == ([], [])


def test_missing_conda_fails_open() -> None:
    assert workflow_checks("proj", "proj", {"torch"}, None) == ([], [])


def test_unwired_missing_env_warns_with_dep_sample() -> None:
    errors, warnings = workflow_checks("proj", "", {"torch", "numpy", "seaborn"}, lambda name: False)
    assert errors == []
    assert len(warnings) == 1
    assert "system Python" in warnings[0]
    assert "torch" in warnings[0]


def test_unwired_missing_env_warns_even_without_deps() -> None:
    _, warnings = workflow_checks("proj", "", set(), lambda name: False)
    assert len(warnings) == 1
    assert "system Python" in warnings[0]


def test_env_exists_but_unwired_warns() -> None:
    errors, warnings = workflow_checks("proj", "", set(), lambda name: name == "proj")
    assert errors == []
    assert "exists but conda_env is unset" in warnings[0]


def test_status_recommends_init_without_entry() -> None:
    lines = workflow_status(None, "proj", "", lambda name: False, [])
    assert "entry: not found" in lines[0]
    assert lines[-1].startswith("next: lazy107 init")


def test_status_next_steps_env_then_submit_then_watch() -> None:
    lines = workflow_status("train.py", "proj", "", lambda name: False, [])
    assert lines[-1] == "next: lazy107 env --yes"

    lines = workflow_status("train.py", "proj", "proj", lambda name: True, [])
    assert "env: 'proj' ready" in lines
    assert lines[-1] == "next: lazy107 submit"

    lines = workflow_status("train.py", "proj", "proj", lambda name: True, ["42", "43"])
    assert lines[-1] == "next: lazy107 watch 43"


def test_status_declares_conda_unknown() -> None:
    lines = workflow_status("train.py", "proj", "", None, [])
    assert any("conda not on PATH" in line for line in lines)


def test_status_env_exists_but_unwired() -> None:
    lines = workflow_status("train.py", "proj", "", lambda name: True, [])
    assert any("exists but not wired" in line for line in lines)
    assert lines[-1] == "next: lazy107 env --yes"
