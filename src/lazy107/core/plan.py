"""RunPlan: every submission decision as data (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPlan:
    entry: str = ""
    partition: str = "Students"
    qos: str = "qos_stu_default"
    account: str = ""
    cpus: int = 4
    mem: str = "16G"
    gpu: int = 0
    nodes: int = 1
    ntasks: int = 1
    time: str = "1:00:00"
    log_dir: str = "logs"
    conda_env: str = ""
    job_name: str = ""
    command: str = ""
    array: str = ""

    @property
    def effective_job_name(self) -> str:
        return self.job_name or Path(self.entry).stem
