"""Deliberate failure #06: allocate past the requested memory limit.

Run with mem = "4G" in 107.toml; the cgroup OOM killer terminates the job.
numpy is available because the demo env's torch install depends on it.

Expected log signature: slurmstepd: error: Detected 1 oom-kill event(s) ...
    (or an uncaught numpy MemoryError traceback — both are valid harvests)
sacct ExitCode: 0:9 (SIGKILL)
"""

import numpy as np

chunks = []
while True:
    chunks.append(np.ones((10**8,), dtype=np.uint8))  # ~100 MB per chunk

print("unreachable")
