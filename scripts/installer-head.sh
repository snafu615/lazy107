#!/usr/bin/env bash
# lazy107 0.1.0 - single-file installer (USTC 107 cluster Slurm submission CLI).
# Upload this one file to the cluster, then:
#     bash lazy107-0.1.0-install.sh
# Options: --env NAME (conda env), --prefix DIR (venv), --python PY, -h.
# With no options: active conda env > python3 on PATH > new conda env
# "lazy107" > venv at ~/.lazy107. Safe to re-run.
#
# The wheel payload is appended after the marker line below by
# scripts/build-installer.sh; do not edit past the marker.

set -euo pipefail

MARKER="__LAZY107_WHEEL_BELOW__"
ENV_NAME="lazy107"
MODE="auto"
PY=""
PREFIX=""
ACTIVATE=""

USAGE='lazy107 0.1.0 single-file installer

usage: bash lazy107-0.1.0-install.sh [options]

options:
  --env NAME     install into conda env NAME (created if missing)
  --prefix DIR   install into a fresh venv at DIR (~/.lazy107 default)
  --python PY    install into python >= 3.11 at PY
  -h, --help     show this help and exit

with no options: active conda env, else python3 on PATH, else a new
conda env "lazy107", else a venv at ~/.lazy107. Safe to re-run.'

py_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "$USAGE"
            exit 0
            ;;
        --env) MODE="conda" ;;
        --prefix) MODE="venv" ;;
        --python) MODE="direct" ;;
        *)
            if [[ "$MODE" == "conda" && -z "${ENV_NAME_SET:-}" ]]; then
                ENV_NAME="$arg"; ENV_NAME_SET=1
            elif [[ "$MODE" == "venv" && -z "${PREFIX_SET:-}" ]]; then
                PREFIX="$arg"; PREFIX_SET=1
            elif [[ "$MODE" == "direct" && -z "${PY_SET:-}" ]]; then
                PY="$arg"; PY_SET=1
            else
                echo "lazy107 installer: unexpected argument '$arg'" >&2
                exit 2
            fi
            ;;
    esac
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
WHEEL="$TMP/lazy107.whl"

# the payload marker is the file's last line; the MARKER= assignment above
# also contains the string, so take the LAST occurrence
line_off="$(grep -aboF "$MARKER" "$0" | tail -n1 | cut -d: -f1)"
if [[ -z "$line_off" ]]; then
    echo "lazy107 installer: payload marker not found (corrupt file?)" >&2
    exit 1
fi
start=$((line_off + ${#MARKER} + 2))
tail -c "+$start" "$0" > "$WHEEL"

AUTO_FALLBACK=0
if [[ "$MODE" == "auto" ]]; then
    AUTO_FALLBACK=1
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] && py_ok "$CONDA_PREFIX/bin/python"; then
        PY="$CONDA_PREFIX/bin/python"
    elif command -v python3 >/dev/null 2>&1 && py_ok "$(command -v python3)"; then
        PY="$(command -v python3)"
    fi
    if [[ -n "$PY" ]]; then
        MODE="direct"
    elif command -v conda >/dev/null 2>&1; then
        MODE="conda"
    else
        MODE="venv"
    fi
fi

if [[ "$MODE" == "direct" ]]; then
    if [[ -z "$PY" ]] || [[ ! -x "$PY" ]]; then
        echo "lazy107 installer: --python needs an executable path" >&2
        exit 2
    fi
    if ! py_ok "$PY"; then
        echo "lazy107 installer: $PY is too old (need python >= 3.11)" >&2
        exit 2
    fi
    echo "lazy107 installer: installing into the environment of $PY"
    if ! "$PY" -m pip install --no-deps --disable-pip-version-check -q "$WHEEL" 2>"$TMP/pip.err"; then
        cat "$TMP/pip.err" >&2
        if [[ "$AUTO_FALLBACK" == "1" ]]; then
            echo "lazy107 installer: that python is not writable, using a dedicated environment instead"
            if command -v conda >/dev/null 2>&1; then
                MODE="conda"
            else
                MODE="venv"
            fi
        else
            echo "lazy107 installer: install failed (try --env or --prefix)" >&2
            exit 1
        fi
    else
        BIN="$(dirname "$PY")/lazy107"
    fi
fi

if [[ "$MODE" == "conda" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "lazy107 installer: conda not found on PATH" >&2
        exit 1
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "lazy107 installer: creating conda env '$ENV_NAME' (python 3.11)..."
        conda create -y -n "$ENV_NAME" python=3.11 -q
    fi
    echo "lazy107 installer: installing into conda env '$ENV_NAME'"
    conda run -n "$ENV_NAME" python -m pip install --no-deps --disable-pip-version-check -q "$WHEEL"
    BIN="$(conda run -n "$ENV_NAME" python -c 'import os, sys; print(os.path.join(os.path.dirname(sys.executable), "lazy107"))')"
    ACTIVATE="conda activate $ENV_NAME"
fi

if [[ "$MODE" == "venv" ]]; then
    PREFIX="${PREFIX:-$HOME/.lazy107}"
    BASE=""
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        BASE="$CONDA_PREFIX/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        BASE="$(command -v python3)"
    fi
    if [[ -z "$BASE" ]] || ! py_ok "$BASE"; then
        echo "lazy107 installer: no python >= 3.11 available to create a venv (use --env with conda)" >&2
        exit 1
    fi
    if [[ ! -x "$PREFIX/bin/python" ]]; then
        echo "lazy107 installer: creating venv at $PREFIX"
        "$BASE" -m venv "$PREFIX"
    fi
    echo "lazy107 installer: installing into venv at $PREFIX"
    "$PREFIX/bin/python" -m pip install --no-deps --disable-pip-version-check -q "$WHEEL"
    BIN="$PREFIX/bin/lazy107"
    ACTIVATE="export PATH=$PREFIX/bin:\$PATH"
fi

if [[ -z "${BIN:-}" ]]; then
    echo "lazy107 installer: internal error (no install target)" >&2
    exit 1
fi
if ! "$BIN" --version >/dev/null 2>&1; then
    echo "lazy107 installer: verification failed at $BIN" >&2
    exit 1
fi

echo
echo "lazy107 installer: done - $("$BIN" --version)"
echo
if [[ -n "$ACTIVATE" ]]; then
    echo "    $ACTIVATE"
fi
echo "    lazy107 --help    (14 commands, incl. the 'everything' wizard)"
echo
__LAZY107_WHEEL_BELOW__
