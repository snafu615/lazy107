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
SHA="__WHEEL_SHA256__"          # substituted at build time; skipped if still a placeholder
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

pip_install() {
    # $1 = python binary; installs $WHEEL, logs everything, returns 0/1
    local log="$TMP/pip.log"
    if ! "$1" -m pip install --no-deps "$WHEEL" >"$log" 2>&1; then
        echo "lazy107 installer: pip install failed - full output:" >&2
        cat "$log" >&2
        return 1
    fi
    rm -f "$log"
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

# extract the wheel appended after the marker line (last occurrence of the
# marker: the MARKER= assignment above also contains the string)
line_off="$(grep -aboF "$MARKER" "$0" | tail -n1 | cut -d: -f1)"
if [[ -z "$line_off" ]]; then
    echo "lazy107 installer: payload marker not found (corrupt file?)" >&2
    exit 1
fi
start=$((line_off + ${#MARKER} + 2))
tail -c "+$start" "$0" > "$WHEEL"

# verify the payload against the sha256 embedded at build time - catches
# uploads that were mangled by copy-paste or a text-mode transfer
if [[ "$SHA" != "__WHEEL_SHA256__" ]]; then
    got="$(sha256sum "$WHEEL" | awk '{print $1}')"
    if [[ "$got" != "$SHA" ]]; then
        echo "lazy107 installer: payload checksum mismatch ($got != $SHA)" >&2
        echo "  the file was corrupted in transfer - re-upload it (binary mode)" >&2
        exit 1
    fi
fi

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
    if ! pip_install "$PY"; then
        if [[ "$AUTO_FALLBACK" == "1" ]]; then
            echo "lazy107 installer: that python is not usable, using a dedicated environment instead"
            if command -v conda >/dev/null 2>&1; then
                MODE="conda"
            else
                MODE="venv"
            fi
        else
            echo "lazy107 installer: install failed - manual retry:" >&2
            echo "    $PY -m pip install --no-deps $WHEEL" >&2
            trap - EXIT
            exit 1
        fi
    else
        BIN="$(dirname "$PY")/lazy107"
    fi
fi

# conda helpers: env dir from `conda env list`, no reliance on `conda run`
conda_env_dir() {
    command -v conda >/dev/null 2>&1 || return 1
    conda env list 2>/dev/null | awk -v n="$1" '$1 == n {print $NF; exit}'
}
conda_base_dir() {
    command -v conda >/dev/null 2>&1 || return 1
    conda env list 2>/dev/null | awk '$1 == "base" {print $NF; exit}'
}

if [[ "$MODE" == "conda" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "lazy107 installer: conda not found on PATH" >&2
        exit 1
    fi
    if [[ -z "$(conda_env_dir "$ENV_NAME")" ]]; then
        echo "lazy107 installer: creating conda env '$ENV_NAME' (python 3.11 + pip)..."
        conda create -y -n "$ENV_NAME" python=3.11 pip -q
    fi
    ENVDIR="$(conda_env_dir "$ENV_NAME")"
    if [[ -z "$ENVDIR" || ! -x "$ENVDIR/bin/python" ]]; then
        echo "lazy107 installer: env '$ENV_NAME' exists but is broken at $ENVDIR" >&2
        echo "  fix: conda env remove -n $ENV_NAME, then re-run this installer" >&2
        exit 1
    fi
    PY="$ENVDIR/bin/python"
    if ! py_ok "$PY"; then
        echo "lazy107 installer: conda env '$ENV_NAME' has an unsupported python ($("$PY" --version 2>&1))" >&2
        echo "  fix: conda env remove -n $ENV_NAME, then re-run this installer" >&2
        exit 1
    fi
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
        echo "lazy107 installer: pip missing in '$ENV_NAME', bootstrapping with ensurepip..."
        "$PY" -m ensurepip --upgrade >"$TMP/ensurepip.log" 2>&1 || {
            cat "$TMP/ensurepip.log" >&2
            echo "lazy107 installer: could not bootstrap pip; fix: conda install -n $ENV_NAME pip" >&2
            exit 1
        }
    fi
    echo "lazy107 installer: installing into conda env '$ENV_NAME'"
    if ! pip_install "$PY"; then
        echo "lazy107 installer: manual retry (the wheel is kept for you):" >&2
        cp "$WHEEL" "$HOME/lazy107-0.1.0-py3-none-any.whl"
        echo "    $PY -m pip install --no-deps $HOME/lazy107-0.1.0-py3-none-any.whl" >&2
        trap - EXIT
        exit 1
    fi
    BIN="$ENVDIR/bin/lazy107"
    ACTIVATE="conda activate $ENV_NAME"
fi

if [[ "$MODE" == "venv" ]]; then
    PREFIX="${PREFIX:-$HOME/.lazy107}"
    BASE=""
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        BASE="$CONDA_PREFIX/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        BASE="$(command -v python3)"
    elif [[ -n "$(conda_base_dir)" ]] && [[ -x "$(conda_base_dir)/bin/python" ]]; then
        BASE="$(conda_base_dir)/bin/python"
    fi
    if [[ -z "$BASE" ]] || ! py_ok "$BASE"; then
        echo "lazy107 installer: no python >= 3.11 available to create a venv (use --env with conda)" >&2
        exit 1
    fi
    if [[ ! -x "$PREFIX/bin/python" ]]; then
        echo "lazy107 installer: creating venv at $PREFIX"
        "$BASE" -m venv "$PREFIX"
    fi
    PY="$PREFIX/bin/python"
    if ! py_ok "$PY"; then
        echo "lazy107 installer: existing venv at $PREFIX has an unsupported python" >&2
        echo "  fix: rm -rf $PREFIX, then re-run this installer" >&2
        exit 1
    fi
    echo "lazy107 installer: installing into venv at $PREFIX"
    if ! pip_install "$PY"; then
        echo "lazy107 installer: manual retry (the wheel is kept for you):" >&2
        cp "$WHEEL" "$HOME/lazy107-0.1.0-py3-none-any.whl"
        echo "    $PY -m pip install --no-deps $HOME/lazy107-0.1.0-py3-none-any.whl" >&2
        trap - EXIT
        exit 1
    fi
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
