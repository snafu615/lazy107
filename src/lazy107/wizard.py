"""`lazy107 everything`: guided end-to-end pipeline, wired as an external scaffold.

Each stage reuses the existing cli commands (cmd_* functions called with
SimpleNamespace args) and the cli module's interactive helpers; nothing here
reimplements plan/env/submit logic. The module is imported lazily from
cli.build_parser so the wizard -> cli import cycle stays one-directional at
module-load time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from lazy107 import cli as cli_mod
from lazy107 import manifest

# Slurm resource presets offered before the per-field editor. cpu-light and
# gpu-light mirror derive_defaults' two profiles, so the menu is never a
# downgrade from the plan's current values.
PRESETS = (
    ("cpu-light", {"gpu": 0, "cpus": 2, "mem": "4G", "time": "1:00:00"}),
    ("cpu-heavy", {"gpu": 0, "cpus": 8, "mem": "32G", "time": "8:00:00"}),
    ("gpu-light", {"gpu": 1, "cpus": 4, "mem": "16G", "time": "2:00:00"}),
    ("gpu-heavy", {"gpu": 1, "cpus": 8, "mem": "64G", "time": "12:00:00"}),
    ("gpu-multi", {"gpu": 2, "cpus": 16, "mem": "64G", "time": "24:00:00"}),
)


def _ask(question: str) -> bool:
    """y/N prompt; Enter or EOF means no."""
    try:
        answer = input(question).strip().lower()
    except EOFError:
        return False
    return answer == "y"


def _edit_slurm(plan) -> dict:
    """Pick a resource preset or customize each field; Enter keeps current values.

    A preset wires all four keys at once (values equal to the current plan
    are dropped, so nothing is rewritten pointlessly); `c` falls through to
    the per-field prompts with common-value hints. gpu/cpus must parse as
    ints (reprompt otherwise); mem/time pass through as strings. Range/safety
    errors are left to validate_plan, which runs right after the edits are
    wired.
    """
    print("slurm resources (Enter keeps the current values):", file=sys.stderr)
    for index, (name, values) in enumerate(PRESETS, start=1):
        print(
            f"  {index}) {name:<9} gpu={values['gpu']} cpus={values['cpus']} "
            f"mem={values['mem']} time={values['time']}",
            file=sys.stderr,
        )
    while True:
        try:
            answer = input(
                f"pick a preset [1-{len(PRESETS)}], c to customize each field: "
            ).strip().lower()
        except EOFError:
            return {}
        if answer in ("", "c", "custom"):
            break
        if answer.isdigit() and 1 <= int(answer) <= len(PRESETS):
            preset = PRESETS[int(answer) - 1][1]
            return {
                key: value
                for key, value in preset.items()
                if value != getattr(plan, key)
            }
        print(
            f"lazy107: pick a number 1-{len(PRESETS)} or c (got {answer!r})",
            file=sys.stderr,
        )
    updates: dict = {}
    for key, label, kind in (
        ("gpu", "number of GPUs (common: 0, 1, 2)", "int"),
        ("cpus", "CPUs per task (common: 2, 4, 8)", "int"),
        ("mem", "memory (e.g. 16G; common: 4G, 16G, 32G)", "str"),
        ("time", "time limit (e.g. 1:00:00; common: 1:00:00, 2:00:00)", "str"),
    ):
        current = getattr(plan, key)
        while True:
            try:
                answer = input(f"  {label} [{current}] ").strip()
            except EOFError:
                return updates
            if not answer:
                break
            if kind == "int":
                try:
                    value: object = int(answer)
                except ValueError:
                    print(
                        f"lazy107: {key} must be a whole number (got {answer!r})",
                        file=sys.stderr,
                    )
                    continue
            else:
                value = answer
            if value != current:
                updates[key] = value
            break
    return updates


def cmd_everything(project_root: Path, args) -> int:
    """Guided end-to-end: entry -> env -> slurm presets -> plan -> submit."""
    # 1. Entry picker (mirrors cmd_plan's interactive gate; pins the pick)
    if (
        args.entry is None
        and not args.yes
        and not manifest.load_manifest(project_root).get("entry")
        and cli_mod._stdin_interactive()
    ):
        chosen = cli_mod._prompt_entry(project_root)
        if chosen:
            print(manifest.wire_entry(project_root, chosen))
    # Resolve the concrete entry (flag > pin > detection) so every later
    # cmd_* call is handed the value and never re-prompts.
    entry = cli_mod._entry(SimpleNamespace(entry=args.entry))
    plan = cli_mod._plan_for(project_root, entry)

    # 2. Env: reuse an existing one, else offer a fresh install
    if plan.conda_env:
        print(f"note: conda_env = {plan.conda_env!r} already wired; skipping env setup")
    elif args.yes or not cli_mod._stdin_interactive():
        rc = cli_mod.cmd_env(
            project_root,
            SimpleNamespace(name=None, entry=entry, dry_run=False, yes=True),
        )
        if rc != 0:
            return rc
    else:
        code = cli_mod._prompt_reuse_env(project_root, plan)
        if code is None:
            if _ask("no reusable env picked; install a fresh conda env now? [y/N] "):
                rc = cli_mod.cmd_env(
                    project_root,
                    SimpleNamespace(name=None, entry=entry, dry_run=False, yes=True),
                )
                if rc != 0:
                    return rc
            else:
                print(
                    "lazy107: skipping env setup; submit will warn if the env is missing",
                    file=sys.stderr,
                )

    # 3. Slurm parameter editor
    if not args.yes and cli_mod._stdin_interactive():
        updates = _edit_slurm(plan)
        if updates:
            print(manifest.wire_fields(project_root, updates))
            plan = cli_mod._plan_for(project_root, entry)

    # 4. Plan print (validates the edits; aborts on error)
    rc = cli_mod.cmd_plan(project_root, SimpleNamespace(entry=entry, array=None))
    if rc != 0:
        return rc

    # 5. Optional sbatch preview
    if not args.yes and cli_mod._stdin_interactive() and _ask("preview the sbatch script? [y/N] "):
        rc = cli_mod.cmd_render(
            project_root, SimpleNamespace(entry=entry, array=None, dry_run=True)
        )
        if rc != 0:
            return rc

    # 6. Submit (its own confirm prompt is the final gate)
    return cli_mod.cmd_submit(
        project_root,
        SimpleNamespace(
            entry=entry, array=None, dry_run=False, skip_check=False, yes=args.yes
        ),
    )
