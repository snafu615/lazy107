"""Tests for contract checks and injection safety."""

from lazy107.core.plan import RunPlan
from lazy107.core.validate import gpu_preflight_error, is_safe, validate_plan


def test_is_safe() -> None:
    assert is_safe("train.py")
    assert is_safe("my_env")
    assert not is_safe("")
    assert not is_safe("train.py; rm -rf /")
    assert not is_safe("my env")
    assert not is_safe("$(whoami)")
    assert not is_safe("a&b")


def test_validate_plan_ok() -> None:
    assert validate_plan(RunPlan(entry="train.py")) == []


def test_validate_plan_rejects_unsafe_fields() -> None:
    errors = validate_plan(RunPlan(entry="train.py", conda_env="my env; evil"))
    assert any("conda_env" in e for e in errors)


def test_validate_plan_rejects_bad_resources() -> None:
    errors = validate_plan(RunPlan(entry="train.py", gpu=-1, cpus=0))
    assert any("gpu" in e for e in errors)
    assert any("cpus" in e for e in errors)


def test_validate_plan_rejects_zero_nodes_ntasks() -> None:
    errors = validate_plan(RunPlan(entry="train.py", nodes=0, ntasks=0))
    assert any("nodes" in e for e in errors)
    assert any("ntasks" in e for e in errors)


def test_validate_plan_ddp_multi_node_requires_ntasks_match() -> None:
    plan = RunPlan(entry="train.py", gpu=2, nodes=2, ntasks=1)
    errors = validate_plan(plan, ddp=True)
    assert any("must equal nodes" in e for e in errors)
    assert validate_plan(RunPlan(entry="train.py", gpu=2, nodes=2, ntasks=2), ddp=True) == []
    # Single-node DDP has no ntasks constraint.
    assert validate_plan(RunPlan(entry="train.py", gpu=2, nodes=1, ntasks=1), ddp=True) == []


def test_validate_plan_array_syntax() -> None:
    for good in ("1", "1-5", "1-5:2", "1-5%2", "0,2,4", "0-7,16-23%8"):
        assert validate_plan(RunPlan(entry="train.py", array=good)) == []
    for bad in ("1-", "a-b", "1 2", "1-5; rm -rf /", "1-5%", "1-5%0", "1-5:0"):
        assert any("array" in e for e in validate_plan(RunPlan(entry="train.py", array=bad)))


def test_validate_plan_rejects_empty_entry() -> None:
    assert any("entry" in e for e in validate_plan(RunPlan(entry="")))


def test_gpu_preflight_error() -> None:
    assert gpu_preflight_error(0, "") is None
    assert gpu_preflight_error(1, None) is None
    assert gpu_preflight_error(1, "12.6") is None
    assert gpu_preflight_error(1, "") is not None
