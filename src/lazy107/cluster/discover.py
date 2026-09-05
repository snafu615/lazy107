"""Query live Slurm associations to derive a valid account/partition/QoS
and to validate resource limits against partition maxima.

sacctmgr/scontrol output is machine-parsed (`--parsable2`), so column
truncation never corrupts values. Everything is read-only except
`write_global_config`, which only writes the user's own ~/.config file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from lazy107 import manifest

_NAME_RE = re.compile(r"PartitionName=(\S+)")
_ACCOUNTS_RE = re.compile(r"AllowAccounts=(\S+)")
_QOS_RE = re.compile(r"AllowQos=(\S+)")
_DEFAULT_RE = re.compile(r"Default=(\S+)")
_MAXTIME_RE = re.compile(r"MaxTime=(\S+)")
_MAXMEM_NODE_RE = re.compile(r"MaxMemPerNode=(\S+)")
_MAXMEM_CPU_RE = re.compile(r"MaxMemPerCPU=(\S+)")


def _slurm_run(argv: list[str]) -> str:
    """stdout of a Slurm CLI call; '' when missing or failing."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def slurm_associations(user: str) -> dict[str, list[tuple[str, list[str]]]]:
    """partition -> [(account, qos list)] from sacctmgr; rows with an empty
    partition apply to every partition."""
    out = _slurm_run(
        [
            "sacctmgr",
            "show",
            "assoc",
            f"user={user}",
            "format=Account,Partition,QOS",
            "--parsable2",
            "--noheader",
        ]
    )
    assoc: dict[str, list[tuple[str, list[str]]]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        account = parts[0].strip()
        partition = parts[1].strip()
        qos = [q.strip() for q in parts[2].split(",") if q.strip()]
        assoc.setdefault(partition, []).append((account, qos))
    return assoc


def slurm_partitions() -> dict[str, tuple[set[str], set[str], bool, str, str]]:
    """partition -> (allow_accounts, allow_qos, is_default, max_time, max_mem)
    from scontrol; limit strings are raw ('' when absent), parsed later so
    callers can fail open."""
    out = _slurm_run(["scontrol", "show", "partition"])
    partitions: dict[str, tuple[set[str], set[str], bool, str, str]] = {}
    for block in out.split("\n\n"):
        name = ""
        allow_accounts: set[str] = set()
        allow_qos: set[str] = set()
        is_default = False
        max_time = ""
        max_mem_node = ""
        max_mem_cpu = ""
        for line in block.splitlines():
            # scontrol packs AllowGroups/AllowAccounts/AllowQos/Default onto
            # one line, so search each line instead of prefix-matching.
            for match in _NAME_RE.finditer(line):
                name = match.group(1)
            for match in _ACCOUNTS_RE.finditer(line):
                allow_accounts.update(v.strip() for v in match.group(1).split(",") if v.strip())
            for match in _QOS_RE.finditer(line):
                allow_qos.update(v.strip() for v in match.group(1).split(",") if v.strip())
            for match in _DEFAULT_RE.finditer(line):
                is_default = "YES" in match.group(1)
            for match in _MAXTIME_RE.finditer(line):
                max_time = match.group(1)
            for match in _MAXMEM_NODE_RE.finditer(line):
                max_mem_node = match.group(1)
            for match in _MAXMEM_CPU_RE.finditer(line):
                max_mem_cpu = match.group(1)
        if name:
            partitions[name] = (
                allow_accounts,
                allow_qos,
                is_default,
                max_time,
                max_mem_node or max_mem_cpu,
            )
    return partitions


def slurm_default_account(user: str) -> str:
    """The user's default account from sacctmgr; '' when undeterminable."""
    out = _slurm_run(
        ["sacctmgr", "show", "user", user, "format=User,DefaultAccount", "--parsable2", "--noheader"]
    )
    parts = out.strip().split("|")
    return parts[1].strip() if len(parts) >= 2 else ""


def resolve_association(user: str) -> tuple[str, str, str] | None:
    """Best (account, partition, qos) allowed for the user, or None.

    Deterministic preference: the default account, then the default
    partition, then a row explicitly naming the partition; QoS is the first
    assigned-and-allowed one in sacctmgr order.
    """
    assoc = slurm_associations(user)
    partitions = slurm_partitions()
    if not assoc or not partitions:
        return None
    default_account = slurm_default_account(user)
    best: tuple | None = None
    for part_order, (
        part_name,
        (allow_accounts, allow_qos, is_default, _max_time, _max_mem),
    ) in enumerate(partitions.items()):
        rows = assoc.get(part_name, []) + assoc.get("", [])
        for row_num, (account, qos_list) in enumerate(rows):
            if allow_accounts and "ALL" not in allow_accounts and account not in allow_accounts:
                continue
            allowed = [q for q in qos_list if not allow_qos or q in allow_qos]
            if not allowed:
                continue
            candidate = (
                (4 if account == default_account else 0)
                + (2 if is_default else 0)
                + (1 if part_name in assoc else 0),
                -part_order,
                -row_num,
                account,
                part_name,
                allowed[0],
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    return best[3], best[4], best[5]


_MEM_RE = re.compile(r"^(\d+)\s*([KMGTP])?$", re.IGNORECASE)
_INFINITE = {"INFINITE", "UNLIMITED"}


def _parse_duration(value: str) -> int | None:
    """Slurm duration ('2-01:30:00', '01:00:00', '30:00' = MM:SS, '30' =
    minutes) -> minutes. None when infinite or unparseable, so callers
    fail open."""
    value = value.strip()
    if not value or value.upper() in _INFINITE:
        return None
    days, rest = ("0", value)
    if "-" in value:
        days, rest = value.split("-", 1)
    parts = rest.split(":")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts) or not days.isdigit():
        return None
    # pad to [days, hours, minutes, seconds]; a bare number means minutes
    nums = [int(days)] + [int(p) for p in parts]
    nums = [0] * (4 - len(nums)) + nums
    d, h, m, s = nums
    return d * 1440 + h * 60 + m + (1 if s else 0)


def _parse_mem(value: str) -> int | None:
    """Memory size ('16G', '184320' = MB) -> MB; None when infinite or
    unparseable, so callers fail open."""
    value = value.strip()
    if not value or value.upper() in _INFINITE:
        return None
    match = _MEM_RE.match(value)
    if not match:
        return None
    number = int(match.group(1))
    unit = (match.group(2) or "M").upper()
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}.get(unit, 1)
    return int(number * factor)


def submission_check(
    account: str, partition: str, qos: str, user: str, time: str = "", mem: str = ""
) -> str | None:
    """Actionable error when the plan's combo is not allowed for the user
    or exceeds the partition's MaxTime/MaxMemPerNode.

    None = allowed or undeterminable (fail open; the sinfo preflight still
    guards partition existence). Only positive evidence of a bad combo or
    an exceeded limit blocks submission.
    """
    assoc = slurm_associations(user)
    partitions = slurm_partitions()
    if not assoc or not partitions:
        return None
    part = partitions.get(partition)
    if part is None:
        return None
    allow_accounts, allow_qos, _is_default, max_time, max_mem = part
    effective = account or slurm_default_account(user)
    if effective and allow_accounts and "ALL" not in allow_accounts and effective not in allow_accounts:
        return (
            f"account {effective} is not allowed on partition {partition} "
            f"(AllowAccounts: {', '.join(sorted(allow_accounts))}); "
            f"set account=... in 107.toml or run `lazy107 discover`"
        )
    my_qos: set[str] = set()
    for row_account, qos_list in assoc.get(partition, []) + assoc.get("", []):
        if effective and row_account != effective:
            continue
        my_qos.update(qos_list)
    if my_qos and qos not in my_qos:
        return (
            f"qos {qos} is not assigned to your account; assigned: {', '.join(sorted(my_qos))}; "
            f"run `lazy107 discover`"
        )
    if allow_qos and qos not in allow_qos:
        return (
            f"qos {qos} is not allowed on partition {partition} "
            f"(AllowQos: {', '.join(sorted(allow_qos))}); run `lazy107 discover`"
        )
    requested_min = _parse_duration(time)
    limit_min = _parse_duration(max_time)
    if requested_min is not None and limit_min is not None and requested_min > limit_min:
        return f"time {time} exceeds partition {partition} MaxTime ({max_time})"
    requested_mb = _parse_mem(mem)
    limit_mb = _parse_mem(max_mem)
    if requested_mb is not None and limit_mb is not None and requested_mb > limit_mb:
        return f"mem {mem} exceeds partition {partition} MaxMemPerNode ({max_mem})"
    return None


def write_global_config(account: str, partition: str, qos: str, path: Path | None = None) -> Path:
    """Write the per-user global config; returns its path."""
    path = path or manifest.global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# lazy107 global config — written by `lazy107 discover`.\n"
        "# Applies to every project for this user; a project 107.toml overrides it.\n"
        f'account = "{account}"\n'
        f'partition = "{partition}"\n'
        f'qos = "{qos}"\n',
        encoding="utf-8",
    )
    return path
