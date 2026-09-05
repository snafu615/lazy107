"""Deliberate failure #07: allocate more VRAM than an RTX 5090 has (32 GB).

Expected log signature: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB ...
sacct ExitCode: 1:0
"""

import torch

x = torch.empty(40 * 1024**3, dtype=torch.uint8, device="cuda")  # 40 GiB > 32 GiB

print("unreachable")
