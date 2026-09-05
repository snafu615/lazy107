"""Command-line interface (only place with user I/O)."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

from lazy107 import __version__
from lazy107 import template as template_mod
from lazy107 import transfer as transfer_mod
from lazy107.cluster import discover as discover_mod
from lazy107.cluster import env as env_mod
from lazy107.cluster.monitor import collect_logs, monitor_commands, sacct_row
from lazy107.cluster.submit import SubmitError, slurm_check, submit_sbatch
from lazy107.core.defaults import (
    TORCH_PACKAGES,
    derive_defaults,
    detect_ddp,
    detect_dependencies,
    scan_imports,
    write_requirements,
)
from lazy107.core.detect import detect_entries, recommend_entry
from lazy107.core.diagnose import diagnose
from lazy107.core.notebook import convert_notebook
from lazy107.core.plan import RunPlan
from lazy107.core.record import (
    find_array,
    read_history,
    record_command,
    record_run,
    recorded_runs,
)
from lazy107.core.render import render_sbatch, write_sbatch
from lazy107.core.validate import gpu_preflight_error, validate_plan
from lazy107.core.workflow import workflow_checks, workflow_status
from lazy107.manifest import (
    MANIFEST_NAME,
    entry_path,
    load_manifest,
    resolve_plan,
    wire_conda_env,
    wire_entry,
)

_MANIFEST_TEMPLATE = """\
# lazy107 project config. Any field may be omitted; LAZY107_* env vars and
# CLI flags override these values.
entry = ""
partition = "Students"
qos = "qos_stu_default"
account = ""
cpus = 4
mem = "16G"
gpu = 0
nodes = 1
ntasks = 1
time = "1:00:00"
log_dir = "logs"
conda_env = ""
job_name = ""
command = ""
array = ""
"""

_INSTALL_TIMEOUT = 900  # seconds per environment-setup command

_MAX_MENU_ENTRIES = 9


def _stdin_interactive() -> bool:
    """True when stdin is a terminal; prompts are skipped otherwise."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, OSError):
        return False


def _prompt_entry(project_root: Path) -> str | None:
    """Interactive entry picker: numbered menu, Enter takes the recommendation.

    Runs only when the entry is neither flagged nor pinned and stdin is a
    TTY. Returns a forward-slash path relative to project_root, or None to
    keep the detected recommendation (single entry or EOF).
    """
    entries = detect_entries(project_root)
    if len(entries) < 2:
        return None
    relative = [p.relative_to(project_root).as_posix() for p in entries]
    print("detected entry files:", file=sys.stderr)
    for i, name in enumerate(relative[: _MAX_MENU_ENTRIES], 1):
        mark = "  <- default" if i == 1 else ""
        print(f"  {i}) {name}{mark}", file=sys.stderr)
    if len(relative) > _MAX_MENU_ENTRIES:
        print(f"  ... and {len(relative) - _MAX_MENU_ENTRIES} more (type its path to pick)", file=sys.stderr)
    while True:
        try:
            answer = input(
                f"choose entry [1-{len(relative)}, Enter={relative[0]}]: "
            ).strip()
        except EOFError:
            return None
        if not answer:
            return relative[0]
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(relative):
                return relative[index - 1]
            print(f"lazy107: no entry numbered {answer} (pick 1-{len(relative)})", file=sys.stderr)
            continue
        if '"' in answer or "\\" in answer or ".." in Path(answer).parts:
            print("lazy107: entry must be a plain relative path", file=sys.stderr)
            continue
        if (project_root / answer).is_file():
            return answer.replace("\\", "/")
        print(f"lazy107: no such file: {answer}", file=sys.stderr)


def _print_edit_hint(project_root: Path) -> None:
    """Point the user at the one file where every plan value can be edited."""
    path = project_root / MANIFEST_NAME
    if path.exists():
        print(
            f"edit any value above in {path} — open the project folder in the "
            "Web Shell GUI (Files) and edit 107.toml, then `lazy107 plan` to re-check"
        )
    else:
        print(
            f"hint: Slurm params live in {path} — create it in the Web Shell GUI "
            "file manager or run `lazy107 config --init`; `lazy107 plan` re-reads it after edits"
        )


def _prompt_reuse_env(project_root: Path, plan: RunPlan) -> int | None:
    """Interactive reuse of an existing conda env; None = keep the install path.

    Lists the user's conda envs, validates a pick against the project's
    detected dependencies (one importability probe per pick, plus the
    CUDA build check for GPU jobs), and pins it as conda_env in
    107.toml. Invalid picks reprompt; Enter declines. Returns an exit
    code when an env was wired, None when the caller should fall through
    to the normal install flow.
    """
    conda = env_mod.find_conda()
    if conda is None:
        print("lazy107: conda not found; cannot list existing environments", file=sys.stderr)
        return None
    envs = env_mod.list_envs(conda)
    if not envs:
        print("lazy107: no existing conda environments to reuse", file=sys.stderr)
        return None
    deps = detect_dependencies(project_root)
    print("existing conda environments:", file=sys.stderr)
    for i, name in enumerate(envs[: _MAX_MENU_ENTRIES], 1):
        print(f"  {i}) {name}", file=sys.stderr)
    if len(envs) > _MAX_MENU_ENTRIES:
        print(
            f"  ... and {len(envs) - _MAX_MENU_ENTRIES} more (type its name to pick)",
            file=sys.stderr,
        )
    while True:
        try:
            answer = input(f"reuse an env [1-{len(envs)}, name, Enter=install fresh]: ").strip()
        except EOFError:
            return None
        if not answer:
            return None
        chosen: str | None = None
        if answer.isdigit():
            index = int(answer)
            chosen = envs[index - 1] if 1 <= index <= len(envs) else None
        elif answer in envs:
            chosen = answer
        if chosen is None:
            print(
                f"lazy107: no conda env numbered or named {answer!r} (pick 1-{len(envs)})",
                file=sys.stderr,
            )
            continue
        missing = env_mod.missing_deps(conda, chosen, deps)
        if missing:
            print(
                f"lazy107: env {chosen!r} is missing {len(missing)} needed "
                f"package(s): {', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            print(
                "lazy107: pick another env, or press Enter to install a fresh env",
                file=sys.stderr,
            )
            continue
        if plan.gpu > 0 and deps & TORCH_PACKAGES:
            build = env_mod.gpu_build_check(conda, chosen)
            if not build:  # None = probe failed, "" = CPU-only torch
                print(
                    f"lazy107: env {chosen!r} cannot run this GPU job "
                    "(torch CUDA build missing or CPU-only); pick another env or press Enter",
                    file=sys.stderr,
                )
                continue
        else:
            build = None
        print(wire_conda_env(project_root, chosen, force=True))
        suffix = f" (torch CUDA build {build})" if build else ""
        print(f"validated env {chosen!r}: all detected dependencies importable{suffix}")
        print(f"edit `conda_env` in {project_root / MANIFEST_NAME} to switch later (Web Shell GUI Files manager)")
        return 0


def _fail(msg: str) -> int:
    print(f"lazy107: {msg}", file=sys.stderr)
    return 1


def _entry(args) -> str | None:
    """Entry as the --entry flag, else the 107.toml pin, else detection."""
    root = Path.cwd()
    raw = getattr(args, "entry", None)
    if not raw:
        raw = load_manifest(root).get("entry")  # a pinned choice wins over detection
    if not raw:
        raw = recommend_entry(root)
    if not raw:
        return None
    path = Path(raw)
    return path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()


def _plan_for(project_root: Path, entry: str | None, array: str | None = None) -> RunPlan:
    flags = {}
    if entry:
        flags["entry"] = entry
    if array:
        flags["array"] = array
    return resolve_plan(project_root, flags, derive_defaults(project_root))


def _warn_unknown_job(project_root: Path, job_id: str) -> None:
    """Loud warnings when the ledger disagrees with the job id (wrong id, or watch-before-submit)."""
    ids = recorded_runs(project_root / "notes" / "runs.md")
    if not ids:
        print(
            "lazy107: warning: no recorded runs in notes/runs.md — submit first (`lazy107 submit`)",
            file=sys.stderr,
        )
    elif job_id not in ids:
        print(f"lazy107: warning: job {job_id} not found in notes/runs.md", file=sys.stderr)


def _record_history(project_root: Path, argv: list[str] | None, rc: int) -> None:
    """Append the invocation to notes/history.md when run inside a project.

    Gate on the project signature (107.toml, notes/, or the scaffold's
    environment.yml) so stray runs from $HOME don't pollute it. init
    records into the new project instead (see cmd_init). History must
    never break the command, so write errors are swallowed.
    """
    if argv is None or not any(
        (project_root / name).exists() for name in (MANIFEST_NAME, "notes", "environment.yml")
    ):
        return
    try:
        record_command(project_root / "notes" / "history.md", argv, rc)
    except OSError:
        pass


def cmd_init(project_root: Path, args) -> int:
    target = project_root / args.name
    try:
        template_mod.scaffold(target, args.name)
    except FileExistsError as err:
        return _fail(str(err))
    record_command(target / "notes" / "history.md", ["init", args.name], 0)
    print(f"created project at {target}")
    return 0


def cmd_plan(project_root: Path, args) -> int:
    entry = _entry(args)
    # Interactive step: when the entry is neither flagged nor pinned and
    # several candidates exist, let the user pick (Enter = recommendation)
    # and pin the choice in 107.toml so every later command honors it.
    if args.entry is None and not load_manifest(project_root).get("entry") and _stdin_interactive():
        chosen = _prompt_entry(project_root)
        if chosen:
            entry = chosen
            print(wire_entry(project_root, chosen))
    plan = _plan_for(project_root, entry, getattr(args, "array", None))
    errors = validate_plan(plan, detect_ddp(project_root))
    if errors:
        return _fail("; ".join(errors))
    for f in fields(RunPlan):
        if f.name != "effective_job_name":
            print(f"{f.name}={getattr(plan, f.name)}")
    _print_edit_hint(project_root)
    return 0


def cmd_render(project_root: Path, args) -> int:
    plan = _plan_for(project_root, _entry(args), getattr(args, "array", None))
    ddp = detect_ddp(project_root)
    errors = validate_plan(plan, ddp)
    if errors:
        return _fail("; ".join(errors))
    entry_path(project_root, plan.entry)
    if args.dry_run:
        print(render_sbatch(plan, ddp), end="")
        return 0
    path = write_sbatch(project_root, plan, ddp=ddp)
    print(f"wrote {path}")
    return 0


def cmd_env(project_root: Path, args) -> int:
    env_name = env_mod.sanitize_env_name(args.name) if args.name else env_mod.env_name_for(project_root)
    plan = _plan_for(project_root, _entry(args))
    scanned = scan_imports(project_root)
    autowrite = (
        bool(scanned)
        and not (project_root / "requirements.txt").exists()
        and not (project_root / "pyproject.toml").exists()
    )
    if args.dry_run:
        if autowrite:
            print(f"# would write requirements.txt (imports: {', '.join(sorted(scanned))})")
        if not plan.conda_env:
            print(f'# would set conda_env = "{env_name}" in {MANIFEST_NAME}')
        elif plan.conda_env != env_name:
            print(f"# note: conda_env is already set to {plan.conda_env!r}; this run prepares {env_name!r}")
        cmds = env_mod.env_commands(project_root, env_name, plan.gpu, requirements_pending=autowrite)
        print("\n".join(cmds))
        return 0
    if not args.yes:
        # Interactive shortcut: reuse an existing env (validated against the
        # project's deps) instead of installing one per project. Explicit
        # --name keeps managing that one env, so no picker is offered.
        if args.name is None and _stdin_interactive():
            code = _prompt_reuse_env(project_root, plan)
            if code is not None:
                return code
        return _fail("env runs install commands; use --yes to execute (or --dry-run to preview)")
    if autowrite:
        path = write_requirements(project_root, scanned)
        print(f"wrote {path} (imports: {', '.join(sorted(scanned))})")
    conda = env_mod.find_conda()
    if conda is None:
        return _fail("conda not found: run `module load miniconda/py312` or install under ~/miniconda3")
    cmds = env_mod.env_commands(project_root, env_name, plan.gpu)
    if env_mod.env_exists(conda, env_name):
        cmds = [c for c in cmds if "conda create" not in c]
        print(f"note: conda env {env_name!r} already exists; skipping create (the update steps are idempotent)")
    for cmd in cmds:
        print(f"$ {cmd}")
        try:
            result = subprocess.run(["bash", "-lc", cmd], check=False, timeout=_INSTALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _fail(f"command timed out after {_INSTALL_TIMEOUT}s: {cmd}")
        if result.returncode != 0:
            return _fail(f"command failed (exit {result.returncode}): {cmd}")
    if plan.gpu > 0 and detect_dependencies(project_root) & TORCH_PACKAGES:
        build = env_mod.gpu_build_check(conda, env_name)
        if build == "":
            return _fail(
                "torch in this env is a CPU-only build; "
                "reinstall via `lazy107 env --yes` (PyPI wheel bundles CUDA)"
            )
        if build is None:
            return _fail(
                f"torch is a project dependency but not importable in env {env_name!r}; "
                f"check `conda run -n {env_name} python -c 'import torch'`"
            )
        print(f"torch CUDA build verified (torch.version.cuda={build})")
    if plan.conda_env:
        if plan.conda_env != env_name:
            print(
                f"note: conda_env is already set to {plan.conda_env!r}, "
                f"but this run prepared {env_name!r}; submissions will activate {plan.conda_env!r}"
            )
    else:
        print(wire_conda_env(project_root, env_name))
    return 0


def cmd_submit(project_root: Path, args) -> int:
    plan = _plan_for(project_root, _entry(args), getattr(args, "array", None))
    ddp = detect_ddp(project_root)
    errors = validate_plan(plan, ddp)
    if errors:
        return _fail("; ".join(errors))
    if plan.entry.endswith(".ipynb"):
        py = project_root / plan.entry.replace(".ipynb", ".py")
        convert_notebook(project_root / plan.entry, py)
        plan = replace(plan, entry=py.relative_to(project_root).as_posix())
    entry_path(project_root, plan.entry)
    # Workflow-state checks: wrong-order and skipped-step mistakes surface loudly
    # before queueing. Blocking errors honor --skip-check; warnings always print.
    conda = env_mod.find_conda()
    probe = (lambda name: env_mod.env_exists(conda, name)) if conda else None
    errors, warnings = workflow_checks(
        env_mod.env_name_for(project_root),
        plan.conda_env,
        detect_dependencies(project_root),
        probe,
    )
    for warning in warnings:
        print(f"lazy107: warning: {warning}", file=sys.stderr)
    if errors and not args.skip_check:
        return _fail("; ".join(errors))
    if not args.skip_check:
        if not slurm_check(plan.partition):
            return _fail(f"partition {plan.partition} failed preflight; use --skip-check to override")
        user = os.environ.get("USER", "") or getpass.getuser()
        error = discover_mod.submission_check(plan.account, plan.partition, plan.qos, user, plan.time, plan.mem)
        if error:
            return _fail(f"{error}; use --skip-check to override")
    if plan.gpu > 0:
        conda = env_mod.find_conda()
        if conda is None:
            return _fail("conda not found; cannot verify GPU build (run `module load miniconda/py312`)")
        build = env_mod.gpu_build_check(conda, plan.conda_env or env_mod.env_name_for(project_root))
        error = gpu_preflight_error(plan.gpu, build)
        if error:
            return _fail(error)
    script = render_sbatch(plan, ddp)
    if args.dry_run:
        print(script, end="")
        return 0
    if not args.yes:
        try:
            answer = input(f"submit {plan.effective_job_name} on {plan.partition} (gpu={plan.gpu})? [y/N] ").strip().lower()
        except EOFError:
            return _fail("aborted: no input available; pass --yes for non-interactive submission")
        if answer != "y":
            return _fail("aborted")
    sbatch_path = write_sbatch(project_root, plan, ddp=ddp)
    job_id = submit_sbatch(project_root, sbatch_path)
    record_run(project_root / "notes" / "runs.md", plan, job_id, sbatch_path)
    print(f"submitted job {job_id} ({plan.effective_job_name})")
    print(f"watch: lazy107 watch {job_id}")
    return 0


def cmd_watch(project_root: Path, args) -> int:
    plan = _plan_for(project_root, None)
    job_name = args.job_name or plan.job_name or "*"
    # The ledger wins: it records the array actually submitted (e.g. a
    # `submit --array` one-off), which 107.toml may not reflect.
    array = find_array(project_root / "notes" / "runs.md", args.job_id) or plan.array
    _warn_unknown_job(project_root, args.job_id)
    for cmd in monitor_commands(args.job_id, job_name, plan.log_dir, array):
        print(cmd)
    return 0


def cmd_status(project_root: Path, args) -> int:
    print("squeue --me")
    if args.job_id:
        print(f"scontrol show job {args.job_id}")
    return 0


def cmd_logs(project_root: Path, args) -> int:
    plan = _plan_for(project_root, None)
    job_name = args.job_name or plan.job_name or "*"
    array = find_array(project_root / "notes" / "runs.md", args.job_id) or plan.array
    _warn_unknown_job(project_root, args.job_id)
    # Both halves matter: normal output goes to .out, failures (tracebacks,
    # slurmstepd errors) go to .err.
    for cmd in monitor_commands(args.job_id, job_name, plan.log_dir, array)[-2:]:
        print(cmd)
    return 0


def cmd_debug(project_root: Path, args) -> int:
    """Diagnose a job: sacct exit code + .out/.err logs -> cause and fix."""
    plan = _plan_for(project_root, None)
    job_name = args.job_name or plan.job_name or "*"
    array = find_array(project_root / "notes" / "runs.md", args.job_id) or plan.array
    _warn_unknown_job(project_root, args.job_id)
    row = sacct_row(args.job_id)
    state, exit_code = row if row else ("UNKNOWN", None)
    text = collect_logs(project_root, plan.log_dir, job_name, args.job_id, array)
    diag = diagnose(text, exit_code)
    print(f"job {args.job_id}: state={state} exit={exit_code or 'n/a'}")
    if diag is None:
        if not text.strip():
            print("no log output found (job may still be queued; check --job-name or logs/)")
        elif exit_code == "0:0":
            print("no failure detected (exit 0:0)")
        else:
            print("no recognized failure signature; last log lines:")
            for line in text.strip().splitlines()[-5:]:
                print(f"  {line}")
        return 0
    print(f"cause: {diag.cause}")
    print(f"fix: {diag.fix}")
    print(f"evidence: {diag.evidence}")
    return 0


def cmd_check(project_root: Path, args) -> int:
    entry = _entry(args)
    plan = _plan_for(project_root, entry)
    conda = env_mod.find_conda()
    probe = (lambda name: env_mod.env_exists(conda, name)) if conda else None
    ids = recorded_runs(project_root / "notes" / "runs.md")
    history = read_history(project_root / "notes" / "history.md")
    lines = workflow_status(entry, env_mod.env_name_for(project_root), plan.conda_env, probe, ids)
    if history:
        lines.insert(-1, f"commands: {len(history)} recorded (latest: {history[-1]})")
    else:
        lines.insert(-1, "commands: none recorded")
    for line in lines:
        print(line)
    return 0


def cmd_transfer(project_root: Path, args) -> int:
    print("\n".join(transfer_mod.checklist(project_root)))
    return 0


def cmd_discover(project_root: Path, args) -> int:
    user = os.environ.get("USER", "") or getpass.getuser()
    resolved = discover_mod.resolve_association(user)
    if resolved is None:
        return _fail(
            "could not determine a valid account/partition/qos from sacctmgr/scontrol; "
            f"set them manually in {discover_mod.manifest.global_config_path()}"
        )
    account, partition, qos = resolved
    print(f"user={user}")
    print(f"account={account}")
    print(f"partition={partition}")
    print(f"qos={qos}")
    if args.dry_run:
        print(f"# dry-run: not writing {discover_mod.manifest.global_config_path()}")
        return 0
    path = discover_mod.write_global_config(account, partition, qos)
    print(f"wrote {path}")
    return 0


def cmd_config(project_root: Path, args) -> int:
    if args.init:
        path = project_root / MANIFEST_NAME
        if path.exists():
            return _fail(f"{MANIFEST_NAME} already exists")
        path.write_text(_MANIFEST_TEMPLATE, encoding="utf-8")
        print(f"wrote {path}")
        return 0
    plan = resolve_plan(project_root, None, derive_defaults(project_root))
    manifest = project_root / MANIFEST_NAME
    global_path = discover_mod.manifest.global_config_path()
    print(f"manifest: {manifest} ({'present' if manifest.exists() else 'absent'})")
    print(f"global: {global_path} ({'present' if global_path.exists() else 'absent'})")
    for f in fields(RunPlan):
        if f.name != "effective_job_name":
            print(f"{f.name}={getattr(plan, f.name)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lazy107", description="Slurm submission CLI for USTC 107 (cluster side)")
    parser.add_argument("--version", action="version", version=f"lazy107 {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="scaffold a new project from the default template")
    p.add_argument("name", help="project name (also the conda env name)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("plan", help="print the resolved run plan (asks which entry when several exist)")
    p.add_argument("--entry", help="entry file (default: detected train.py/main.py, or the 107.toml pin)")
    p.add_argument("--array", help="job array spec (e.g. 1-5%%2 or 0,2,4)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("render", help="render the sbatch script")
    p.add_argument("--entry")
    p.add_argument("--array", help="job array spec (e.g. 1-5%%2 or 0,2,4)")
    p.add_argument("--dry-run", action="store_true", help="print the script without writing")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("env", help="prepare a conda environment (or reuse an existing one when interactive)")
    p.add_argument("--name", help="conda env name (default: sanitized project dir name)")
    p.add_argument("--entry")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--yes", action="store_true", help="run the commands non-interactively")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("submit", help="render, validate, and submit")
    p.add_argument("--entry")
    p.add_argument("--array", help="job array spec (e.g. 1-5%%2 or 0,2,4)")
    p.add_argument("--dry-run", action="store_true", help="print the script; do not write or submit")
    p.add_argument("--skip-check", action="store_true", help="skip the sinfo preflight")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("watch", help="print monitoring commands for a running job")
    p.add_argument("job_id")
    p.add_argument("--job-name")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("status", help="print queue status commands")
    p.add_argument("job_id", nargs="?")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs", help="print the tail commands for a job's output and error logs")
    p.add_argument("job_id")
    p.add_argument("--job-name")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("debug", help="diagnose a failed job from its logs and sacct exit code")
    p.add_argument("job_id")
    p.add_argument("--job-name")
    p.set_defaults(func=cmd_debug)

    p = sub.add_parser("transfer", help="print the large-file upload checklist (GUI, print-only)")
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("discover", help="query Slurm and write the per-user global config")
    p.add_argument("--dry-run", action="store_true", help="print the resolved values without writing")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("config", help="show effective config or write a 107.toml")
    p.add_argument("--init", action="store_true", help="write a 107.toml manifest")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("check", help="show workflow status and the next step")
    p.set_defaults(func=cmd_check)

    from lazy107 import wizard  # local import: wizard re-imports cli

    p = sub.add_parser("everything", help="guided end-to-end: entry, env, slurm presets, submit")
    p.add_argument("--entry")
    p.add_argument("--yes", action="store_true", help="run non-interactively (fresh env, direct submit)")
    p.set_defaults(func=wizard.cmd_everything)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    argv = sys.argv[1:] if argv is None else argv  # for the history ledger
    if not getattr(args, "command", None):
        parser.print_help()
        print(
            "\nlazy107: no command given; typical flow: "
            "init -> plan -> render --dry-run -> env --yes -> submit -> watch/debug",
            file=sys.stderr,
        )
        return 1
    try:
        rc = args.func(Path.cwd(), args)
        if args.command != "init":  # init records into the new project instead
            _record_history(Path.cwd(), argv, rc)
        return rc
    except (ValueError, FileNotFoundError, FileExistsError, SubmitError) as err:
        print(f"lazy107: {err}", file=sys.stderr)
        if args.command != "init":
            _record_history(Path.cwd(), argv, 1)
        return 1
