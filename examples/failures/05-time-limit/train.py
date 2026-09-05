"""Deliberate failure #05: sleep past the requested wall time.

Run with time = "0:02:00" in 107.toml; Slurm kills the job at 2 minutes.

Expected log signature: slurmstepd: error: *** JOB <id> ON <node> CANCELLED AT ... DUE TO TIME LIMIT ***
sacct ExitCode: 0:15 (SIGTERM)
"""

import time

time.sleep(300)

print("unreachable")
