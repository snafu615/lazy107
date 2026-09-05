"""Deliberate failure #01: import a package that is not installed.

Expected log signature: ModuleNotFoundError: No module named 'nonexistent_module_xyz'
sacct ExitCode: 1:0
"""

import nonexistent_module_xyz  # noqa: F401

print("unreachable")
