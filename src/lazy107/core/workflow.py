"""Workflow-state checks: loud messages for wrong-order and skipped steps.

Pure logic only; every cluster probe (env existence) is injected as a
callable so the module stays unit-testable without a cluster. The submit
command turns errors into a blocking preflight and warnings into stderr
hints; `check` renders the same state as a status dashboard.
"""

from __future__ import annotations

from collections.abc import Callable


def workflow_checks(
    project_env_name: str,
    conda_env: str,
    third_party_deps: set[str],
    env_exists: Callable[[str], bool] | None,
) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the project's workflow state.

    errors are deterministic failures that make the submitted job die
    immediately; warnings are loud hints for skipped steps. env_exists is
    None when conda is unavailable, which fails open (no checks).
    """
    if env_exists is None:
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    if conda_env and not env_exists(conda_env):
        errors.append(
            f"conda env {conda_env!r} is configured in 107.toml but not created; "
            "run `lazy107 env --yes` before submitting"
        )
    elif not conda_env:
        if env_exists(project_env_name):
            warnings.append(
                f"conda env {project_env_name!r} exists but conda_env is unset; "
                "this job will not activate it (run `lazy107 env --yes` to wire it)"
            )
        elif third_party_deps:
            sample = ", ".join(sorted(third_party_deps)[:3])
            warnings.append(
                "no conda env wired or created; the job will run with system Python and "
                f"imports like {sample} will likely fail (run `lazy107 env --yes`)"
            )
        else:
            warnings.append(
                "no conda env wired or created; the job will run with system Python "
                "(run `lazy107 env --yes`)"
            )
    return errors, warnings


def workflow_status(
    entry: str | None,
    project_env_name: str,
    conda_env: str,
    env_exists: Callable[[str], bool] | None,
    recorded: list[str],
) -> list[str]:
    """Status lines for `lazy107 check`, ending with a single next step."""
    lines: list[str] = []
    if entry:
        lines.append(f"entry: {entry}")
    else:
        lines.append("entry: not found (create train.py, or `lazy107 init <name>`)")

    env_ok = False
    if conda_env:
        if env_exists is not None and not env_exists(conda_env):
            lines.append(
                f"env: {conda_env!r} configured in 107.toml but not created — "
                "run `lazy107 env --yes`"
            )
        else:
            env_ok = True
            lines.append(f"env: {conda_env!r} ready")
    elif env_exists is None:
        lines.append("env: unknown (conda not on PATH — run `module load miniconda/py312`)")
    elif env_exists(project_env_name):
        lines.append(
            f"env: {project_env_name!r} exists but not wired — run `lazy107 env --yes` to wire it"
        )
    else:
        lines.append("env: not prepared — run `lazy107 env --yes`")

    if recorded:
        lines.append(f"runs: {len(recorded)} recorded (latest: {recorded[-1]})")
    else:
        lines.append("runs: none recorded")

    if not entry:
        lines.append("next: lazy107 init <name>  # or create an entry (train.py)")
    elif not env_ok:
        lines.append("next: lazy107 env --yes")
    elif not recorded:
        lines.append("next: lazy107 submit")
    else:
        lines.append(f"next: lazy107 watch {recorded[-1]}")
    return lines
