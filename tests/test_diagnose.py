"""Tests for the failure-diagnosis signature library.

Signatures are the verbatim output harvested on the 107 cluster
(examples/failures/SIGNATURES.md, 2026-09-03).
"""

from lazy107.core.diagnose import diagnose, exit_code_diagnosis

MODULE_NOT_FOUND = """\
Traceback (most recent call last):
  File "/home/scc/pb24061316/failures/01-missing-module/train.py", line 7, in <module>
    import nonexistent_module_xyz  # noqa: F401
ModuleNotFoundError: No module named 'nonexistent_module_xyz'
"""

TIME_LIMIT = (
    "[2026-09-03T13:35:06.005] error: *** JOB 53074 ON anode01 CANCELLED AT "
    "2026-09-03T13:35:06 DUE TO TIME LIMIT ***\n"
)

CUDA_OOM = """\
[W903 13:41:51.709566905 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 33136967680, total: 33668857856).
Traceback (most recent call last):
  File "/home/scc/pb24061316/failures/07-cuda-oom/train.py", line 9, in <module>
    x = torch.empty(40 * 1024**3, dtype=torch.uint8, device="cuda")
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 31.36 GiB of which 30.86 GiB is free. Including non-PyTorch memory, this process has 498.00 MiB memory in use.
"""


def test_module_not_found() -> None:
    diag = diagnose(MODULE_NOT_FOUND)
    assert diag is not None
    assert diag.cause == "missing dependency 'nonexistent_module_xyz'"
    assert "lazy107 env --yes" in diag.fix
    assert "ModuleNotFoundError" in diag.evidence


def test_file_not_found() -> None:
    diag = diagnose("FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'\n")
    assert diag is not None
    assert diag.cause == "missing data file 'data/train.csv'"
    assert "transfer" in diag.fix


def test_import_error() -> None:
    diag = diagnose("ImportError: cannot import name 'resnet18' from 'model'\n")
    assert diag is not None
    assert diag.cause == "cannot import 'resnet18' from 'model'"


def test_time_limit_line_without_exit_code() -> None:
    diag = diagnose(TIME_LIMIT)
    assert diag is not None
    assert diag.cause == "hit the wall-time limit"
    assert "checkpointing" in diag.fix


def test_time_limit_exit_code_only() -> None:
    diag = diagnose("", "0:15")
    assert diag is not None
    assert "SIGTERM" in diag.cause
    assert "raise `time`" in diag.fix


def test_oom_kill_line() -> None:
    diag = diagnose("slurmstepd: error: Detected 2 oom-kill event(s) in step 53075.batch cgroup.\n")
    assert diag is not None
    assert diag.cause == "OOM-killed by Slurm (memory limit)"
    assert "raise `mem`" in diag.fix


def test_cuda_oom_parses_sizes() -> None:
    diag = diagnose(CUDA_OOM)
    assert diag is not None
    assert "40.00 GiB" in diag.cause
    assert "31.36 GiB" in diag.cause
    assert "30.86 GiB" in diag.cause
    assert "batch size" in diag.fix


def test_generic_exception_fallback() -> None:
    diag = diagnose("RuntimeError: CUDA error: device-side assert triggered\n")
    assert diag is not None
    assert diag.cause == "RuntimeError: CUDA error: device-side assert triggered"


def test_exit_code_fallbacks() -> None:
    assert exit_code_diagnosis("1:0") is not None
    assert "SIGKILL" in exit_code_diagnosis("0:9").cause
    assert "SIGTERM" in exit_code_diagnosis("0:15").cause
    assert exit_code_diagnosis("7:0") is None
    assert exit_code_diagnosis(None) is None


def test_log_signature_beats_exit_code() -> None:
    # A time-limited job whose log also carries a traceback: the specific
    # log signature should win over the generic SIGTERM explanation.
    diag = diagnose(MODULE_NOT_FOUND, "0:15")
    assert diag is not None
    assert diag.cause == "missing dependency 'nonexistent_module_xyz'"
