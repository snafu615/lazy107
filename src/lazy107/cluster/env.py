"""Conda environment management (cluster side; subprocess faked in tests)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from lazy107.core.defaults import (
    GPU_PACKAGES,
    IMPORT_BY_PIP,
    TORCH_PACKAGES,
    detect_dependencies,
)

MODULE_LOAD = "module load miniconda/py312  # no-op if conda is already on PATH"

_PROBE_TIMEOUT = 30


def find_conda() -> str | None:
    """Path to conda (PATH first, then ~/miniconda3), or None."""
    on_path = shutil.which("conda")
    if on_path:
        return on_path
    candidate = Path.home() / "miniconda3" / "bin" / ("conda.exe" if os.name == "nt" else "conda")
    return str(candidate) if candidate.exists() else None


def sanitize_env_name(name: str) -> str:
    """Replace characters conda env names cannot contain; raises when the result is empty."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if not cleaned:
        raise ValueError(f"cannot derive a conda env name from {name!r}")
    return cleaned


def env_name_for(project_root: Path) -> str:
    """Sanitized project-dir name; raises when the result would be empty."""
    return sanitize_env_name(project_root.name)


def env_exists(conda: str, env_name: str) -> bool:
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False
    try:
        envs = json.loads(result.stdout).get("envs", [])
    except json.JSONDecodeError:
        return False
    return any(Path(e).name == env_name for e in envs)


def list_envs(conda: str) -> list[str]:
    """Names of existing conda envs, parsed from `conda env list` ([] when unavailable)."""
    try:
        result = subprocess.run(
            [conda, "env", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return []
    names: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = re.match(r"^(\S+)\s+\*?\s+/", line)
        if match:
            names.append(match.group(1))
    return sorted(set(names))


def _import_candidates(pip_name: str) -> list[str]:
    """Plausible import names for a pip package; [] when it cannot be mapped."""
    mapped = IMPORT_BY_PIP.get(pip_name)
    candidates = [mapped] if mapped else []
    if "-" not in pip_name and pip_name not in candidates:
        candidates.append(pip_name)
    return candidates


def missing_deps(conda: str, env_name: str, packages: set[str]) -> list[str]:
    """Pip deps not importable in the env ([] = all present).

    Probing uses importlib.util.find_spec (no imports executed), one
    `conda run` call for all deps. Names that cannot be mapped to an
    import (e.g. `tf-keras`) are skipped rather than wrongly reported
    missing; any probe failure reports everything missing (fail closed).
    """
    probes = {name: _import_candidates(name) for name in sorted(packages)}
    probes = {name: candidates for name, candidates in probes.items() if candidates}
    if not probes:
        return []
    lines = ["import importlib.util as _u"]
    for name, candidates in probes.items():
        checks = " or ".join(f"_u.find_spec({c!r}) is not None" for c in candidates)
        lines.append(f"print({name!r}, {checks})")
    try:
        result = subprocess.run(
            [conda, "run", "-n", env_name, "python", "-c", "\n".join(lines)],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return list(probes)
    if result.returncode != 0:
        return list(probes)
    missing: list[str] = []
    for line in result.stdout.splitlines():
        if line.endswith(" False"):
            missing.append(line.rsplit(" ", 1)[0].strip("'"))
    return missing


def _gpu_overrides(env_name: str, deps: set[str]) -> list[str]:
    """pip overrides guaranteeing CUDA builds (PyPI torch/tf wheels bundle CUDA; jax does not)."""
    run = f"conda run -n {env_name} pip install"
    cmds = []
    if deps & TORCH_PACKAGES:
        cmds.append(f"{run} torch torchvision  # PyPI default wheel bundles CUDA")
    if deps & {"jax", "jaxlib"}:
        cmds.append(f'{run} "jax[cuda12]"  # default jax is CPU-only')
    if deps & {"tensorflow", "tf-keras", "keras"}:
        cmds.append(f"{run} tensorflow  # PyPI default wheel is GPU-enabled")
    return cmds


def env_commands(
    project_root: Path,
    env_name: str,
    gpu: int = 0,
    requirements_pending: bool = False,
) -> list[str]:
    """Deterministic, printable commands to prepare the project's conda env.

    requirements_pending previews the command list as if cmd_env had just
    written requirements.txt (dry-run accuracy without any file writes).
    """
    cmds = [MODULE_LOAD, f"conda create -y -n {env_name} python=3.12"]
    deps = detect_dependencies(project_root)
    if (project_root / "environment.yml").exists():
        cmds.append(f"conda env update -n {env_name} -f environment.yml")
    if requirements_pending or (project_root / "requirements.txt").exists():
        cmds.append(f"conda run -n {env_name} pip install -r requirements.txt")
    elif (project_root / "pyproject.toml").exists():
        cmds.append(f"conda run -n {env_name} pip install -e .")
    if gpu > 0 and deps & GPU_PACKAGES:
        # Never let conda channels (CPU torch) be the final install for GPU jobs.
        cmds.extend(_gpu_overrides(env_name, deps))
    return cmds


def gpu_build_check(conda: str, env_name: str) -> str | None:
    """torch.version.cuda of the env's torch: None = not installed, '' = CPU-only build."""
    try:
        result = subprocess.run(
            [conda, "run", "-n", env_name, "python", "-c", "import torch; print(torch.version.cuda or '')"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or ""
