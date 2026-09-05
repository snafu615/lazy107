"""Deliberate failure #04: import a name that does not exist in model.py.

Expected log signature: ImportError: cannot import name 'resnet18' from 'model'
sacct ExitCode: 1:0
"""

from model import resnet18  # noqa: F401

print("unreachable")
