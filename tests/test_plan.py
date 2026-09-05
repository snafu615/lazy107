"""Tests for RunPlan."""

from lazy107.core.plan import RunPlan


def test_default_job_name_from_entry() -> None:
    assert RunPlan(entry="src/train.py").effective_job_name == "train"


def test_explicit_job_name_wins() -> None:
    assert RunPlan(entry="train.py", job_name="my_job").effective_job_name == "my_job"
