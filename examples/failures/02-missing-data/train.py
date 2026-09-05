"""Deliberate failure #02: read a data file that was never uploaded.

Expected log signature: FileNotFoundError: [Errno 2] No such file or directory: 'data/raw.pt'
sacct ExitCode: 1:0
"""

data = open("data/raw.pt", "rb").read()

print("unreachable")
