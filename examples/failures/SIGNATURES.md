# SIGNATURES.md — live failure harvest results (2026-09-03)

What the four failure jobs actually produced on the 107 cluster. Feeds the
debug command's signature library; see README.md for how each project was built.

## Key operational fact

`render.py` emits both `--output=logs/%x_%j.out` and `--error=logs/%x_%j.err`.
Every harvested failure wrote to stderr only → all signatures below came from
the `.err` files; every `.out` was empty. The debug command must read both,
and `logs`/`watch` (monitor.py) currently tail only `.out` — needs a fix.

## Verified signatures

### 01 ModuleNotFoundError (exit 1:0)

```
Traceback (most recent call last):
  File "/home/scc/pb24061316/failures/01-missing-module/train.py", line 7, in <module>
    import nonexistent_module_xyz  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'nonexistent_module_xyz'
```

- Match: last traceback line `ModuleNotFoundError: No module named '<name>'`
- Also parse `File "...", line N` for the culprit location
- Fix hint: add `<name>` to requirements/environment.yml, re-run `lazy107 env --yes`

### 05 time limit (exit 0:15)

```
[2026-09-03T13:35:06.005] error: *** JOB 53074 ON anode01 CANCELLED AT 2026-09-03T13:35:06 DUE TO TIME LIMIT ***
```

- Format note: timestamp-prefixed `[ISO-ts .ms] error:`, **no** `slurmstepd:` prefix (newer Slurm format)
- Match: `\*\*\* JOB (\d+) ON (\S+) CANCELLED AT .* DUE TO TIME LIMIT \*\*\*`
- Fix hint: raise `time`, add checkpointing, or reduce the workload

### 06 memory limit — DID NOT FIRE (unexpected)

Job 53075 (`mem = 4G`, ~100 MB/chunk growth loop) survived the full 5:00 and
was killed by the TIME LIMIT, not by memory. So `--mem=4G` was not enforced on
P107-RTX5090 (or the job was swap-throttled instead of killed). The classic
`Detected N oom-kill event(s)` slurmstepd line remains **UNVERIFIED** on this
cluster.

Follow-up: submit a probe with `mem = 4G`, watch `scontrol show job` (Mem/RSS
fields) while it runs; if memory limits are never enforced, deprioritize the
oom-kill matcher and treat time-limit as the observable resource-exhaustion
signal on P107.

### 07 CUDA OOM (exit 1:0)

```
[W903 13:41:51.709566905 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 33136967680, total: 33668857856).
Traceback (most recent call last):
  File "/home/scc/pb24061316/failures/07-cuda-oom/train.py", line 9, in <module>
    x = torch.empty(40 * 1024**3, dtype=torch.uint8, device="cuda")
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 31.36 GiB of which 30.86 GiB is free. ...
```

- Match: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X. GPU 0 has a total capacity of Y GiB of which Z GiB is free`
- Parse X/Y/Z → fix hint: requested X vs free Z; reduce batch size / model size
- Also present: the allocator warning line and torch's own `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` hint

## Exit-code table (verified)

| Case | State | ExitCode |
|---|---|---|
| 01 missing module | FAILED | 1:0 |
| 05 time limit | TIMEOUT (CANCELLED) | 0:15 |
| 06 mem limit (not enforced) | TIMEOUT (CANCELLED) | 0:15 |
| 07 CUDA OOM | FAILED | 1:0 |

## Open items

- Slurm version unknown — run `scontrol show config | grep -i slurmversion` and tie the timestamped error prefix to it.
- oom-kill line wording unverified (see 06).
- `logs`/`watch` now tail both `.out` and `.err`; `debug` command built (`lazy107 debug <job_id>`, core/diagnose.py + cluster/monitor.py).
