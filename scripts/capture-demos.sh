#!/usr/bin/env bash
# capture-demos.sh — capture verified terminal output for the lazy107 demo
# series (docs/demos/). Run ON THE 107 LOGIN NODE, from the repo root:
#
#   ./scripts/capture-demos.sh 01        # run one demo
#   ./scripts/capture-demos.sh all       # run everything (about an hour, mostly waits)
#
# Output lands in captures/<demo>/*.txt, each file prefixed with the command
# that produced it. Projects are staged under ~/demos/ and reused, so re-runs
# are idempotent. Submitted jobs are polled via sacct until they reach a
# terminal state (timeout: $DEMO_TIMEOUT_SECS seconds, default 1800).
#
# Demos 02 and 04-06 submit real jobs on P107-RTX5090 (qos_p107-rtx5090,
# MaxJobsPU=4); the script runs one job at a time. Demo 02's `env --yes` is
# the only long step (~10 min, one time; every later demo reuses that env).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMOS="$HOME/demos"
CAPS="$ROOT/captures"
DEMO_TIMEOUT_SECS="${DEMO_TIMEOUT_SECS:-1800}"

say() { printf '\n== %s ==\n' "$*"; }
fail() { echo "capture-demos: $*" >&2; exit 1; }

# cap <demo> <name> -- <cmd...>   run a command, record it in captures/<demo>/<name>.txt
cap() {
    local demo="$1" name="$2"
    shift 2
    local capf="$CAPS/$demo/$name.txt"
    {
        echo "\$ $*"
        "$@" 2>&1
        echo "--- exit: $?"
    } | tee "$capf"
}

# Extract the job id from a captured lazy107 submit line
# ("submitted job 123 (name)").
submitted_id() { sed -n 's/.*submitted[^0-9]*\([0-9][0-9]*\).*/\1/p' "$1" | tail -n1; }

# wait_and_sacct <demo> <name> <job_id> <dir>   poll sacct until terminal, record the row
wait_and_sacct() {
    local demo="$1" name="$2" id="$3" capf="$CAPS/$demo/$name.txt"
    local state="" i=0 max
    max=$((DEMO_TIMEOUT_SECS / 10))
    while [ "$i" -lt "$max" ]; do
        state="$(sacct -j "$id" -n -o State --parsable2 2>/dev/null | head -n1 | tr -d '[:space:]')"
        case "$state" in
            COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|DEADLINE|NODE_FAIL|BOOT_FAIL|PREEMPTED)
                break ;;
        esac
        sleep 10
        i=$((i + 1))
    done
    {
        echo "--- job $id final state: ${state:-UNKNOWN}"
        sacct -j "$id" -o JobID,JobName,Partition,State,ExitCode,Elapsed --parsable2 2>/dev/null
    } | tee -a "$capf"
}

# submit_and_wait <demo> <name> <dir>   lazy107 submit --yes in <dir>, poll to completion
submit_and_wait() {
    local demo="$1" name="$2" dir="$3" capf="$CAPS/$demo/$name.txt" id
    cd "$dir" || fail "cannot cd $dir"
    say "submit in $dir"
    {
        echo "\$ lazy107 submit --yes"
        lazy107 submit --yes 2>&1
        echo "--- exit: $?"
    } | tee "$capf"
    id="$(submitted_id "$capf")"
    [ -n "$id" ] || { echo "capture-demos: submit produced no job id (see $capf)" >&2; return 1; }
    echo "job $id" >> "$capf"
    wait_and_sacct "$demo" "$name" "$id"
}

demo01() {
    local d="$CAPS/01"
    mkdir -p "$d"
    say "demo 01: install + discover (login node only, no jobs)"
    cap 01 version -- lazy107 --version
    cap 01 help -- lazy107 --help
    cap 01 discover-dry -- lazy107 discover --dry-run
    cap 01 discover -- lazy107 discover
    cap 01 config -- lazy107 config
}

demo02() {
    local d="$CAPS/02" proj="$DEMOS/demo" id
    mkdir -p "$d" "$DEMOS"
    [ -d "$proj" ] || cp -r "$ROOT/examples/resnet-cifar10" "$proj"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 02: happy path in $proj"
    cap 02 transfer -- lazy107 transfer
    cap 02 plan -- lazy107 plan
    cap 02 env-dry -- lazy107 env --dry-run
    say "env --yes (one-time, ~10 min first run; idempotent afterwards)"
    cap 02 env-yes -- lazy107 env --yes
    cap 02 cuda-proof -- bash -c 'conda run -n demo python -c "import torch; print(torch.version.cuda)"'
    cap 02 render-dry -- lazy107 render --dry-run
    submit_and_wait 02 happy "$proj" || return 1
    id="$(submitted_id "$d/happy.txt")"
    cap 02 watch-cmds -- lazy107 watch "$id"
    cap 02 logs-cmds -- lazy107 logs "$id"
    cap 02 debug-success -- lazy107 debug "$id"
    cap 02 metrics -- cat outputs/metrics.json
    cap 02 ledger -- cat notes/runs.md
}

demo03() {
    local d="$CAPS/03" proj="$DEMOS/demo"
    mkdir -p "$d"
    [ -d "$proj" ] || fail "run demo 02 first ($proj missing)"
    command -v script >/dev/null || fail "util-linux 'script' is required to drive the wizard prompts"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 03: everything wizard in $proj (0 GPU; the submit is declined)"
    # The wizard reads prompts from a TTY, so drive it through a pseudo-TTY
    # (script -qec); the pty echo of the typed answers lands in the capture.
    # 3a: entry 1, preset 4 (gpu-heavy) -> wires cpus/mem/time, preview y, decline submit.
    cp 107.toml 107.toml.bak 2>/dev/null || true
    cap 03 wizard-preset -- bash -c "printf '1\n4\ny\nn\n' | script -qec 'lazy107 everything' /dev/null"
    cap 03 manifest -- cat 107.toml
    # 3b: the custom path — c, then Enter on every field (nothing changes).
    cap 03 wizard-custom -- bash -c "printf 'c\n\n\n\n\n\nn\nn\n' | script -qec 'lazy107 everything' /dev/null"
    cap 03 plan -- lazy107 plan
    # Restore the project so demo 02's state survives for later demos.
    mv -f 107.toml.bak 107.toml 2>/dev/null || true
}

demo04() {
    local d="$CAPS/04" proj id
    mkdir -p "$d" "$DEMOS"
    for p in 01-missing-module 05-time-limit 06-oom-kill 07-cuda-oom; do
        proj="$DEMOS/fail-$p"
        [ -d "$proj" ] || cp -r "$ROOT/examples/failures/$p" "$proj"
        cd "$proj" || fail "cannot cd $proj"
        say "demo 04: failure project $p"
        cap 04 "plan-$p" -- lazy107 plan
        submit_and_wait 04 "$p" "$proj" || return 1
        id="$(submitted_id "$d/$p.txt")"
        cap 04 "debug-$p" -- lazy107 debug "$id"
        # Refresh the harvested fixtures (feeds tests/fixtures + SIGNATURES.md).
        mkdir -p "$ROOT/examples/failures/fixtures"
        cp -f logs/*.out logs/*.err "$ROOT/examples/failures/fixtures/" 2>/dev/null || true
    done
}

demo05() {
    local d="$CAPS/05" proj
    mkdir -p "$d" "$DEMOS"
    proj="$DEMOS/ddp"
    [ -d "$proj" ] || cp -r "$ROOT/examples/resnet-cifar10-ddp" "$proj"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 05a: DDP detection and two-GPU run"
    cap 05 ddp-plan -- lazy107 plan
    cap 05 ddp-render -- lazy107 render --dry-run
    submit_and_wait 05 ddp "$proj" || return 1
    cap 05 ddp-metrics -- cat outputs/metrics.json
    cap 05 ddp-logtail -- bash -c 'tail -n 10 logs/*.out'
    cap 05 demo02-metrics -- bash -c 'cat "$HOME/demos/demo/outputs/metrics.json"'

    proj="$DEMOS/sweep"
    [ -d "$proj" ] || cp -r "$ROOT/examples/sweep" "$proj"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 05b: job array (4 tasks, 2 concurrent)"
    cap 05 sweep-plan -- lazy107 plan
    cap 05 sweep-render -- lazy107 render --dry-run
    submit_and_wait 05 sweep "$proj" || return 1
    cap 05 sweep-logs -- bash -c 'ls logs'
    cap 05 sweep-tails -- bash -c 'for f in logs/*.out; do echo "== $f"; cat "$f"; done'

    proj="$DEMOS/ddp"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 05c: command override (dry-run only, no compute)"
    cp 107.toml 107.toml.bak
    printf '\ncommand = "python train.py --epochs 1 --batch-size 64"\n' >> 107.toml
    cap 05 cmd-override -- lazy107 render --dry-run
    mv -f 107.toml.bak 107.toml
}

demo06() {
    local d="$CAPS/06" proj id
    mkdir -p "$d" "$DEMOS"

    # 6a: workflow guard — conda_env set but the env was never created (0 GPU)
    proj="$DEMOS/guard"
    if [ ! -d "$proj" ]; then
        (cd "$DEMOS" && lazy107 init guard >/dev/null) || fail "init guard failed"
        printf '\nconda_env = "guard-never-created"\n' >> "$proj/107.toml"
    fi
    cd "$proj" || fail "cannot cd $proj"
    say "demo 06a: submit blocked before anything is queued (exit 1 expected)"
    cap 06 guard-block -- lazy107 submit --yes

    # 6b: the confirm prompt is honored (0 GPU)
    proj="$DEMOS/demo"
    [ -d "$proj" ] || fail "run demo 02 first ($proj missing)"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 06b: answering n at the submit prompt"
    cap 06 confirm-n -- bash -c 'echo n | lazy107 submit'

    # 6c: config layers (0 GPU)
    say "demo 06c: manifest vs env var"
    cp 107.toml 107.toml.bak
    printf '\ngpu = 0\n' >> 107.toml
    cap 06 layer-gpu0 -- lazy107 render --dry-run
    cap 06 layer-env -- bash -c 'LAZY107_GPU=2 lazy107 plan'
    mv -f 107.toml.bak 107.toml

    # 6d: runtime CUDA guard trip (1 short GPU job, manual sbatch)
    proj="$DEMOS/fail-08"
    [ -d "$proj" ] || cp -r "$ROOT/examples/failures/08-cuda-guard" "$proj"
    cd "$proj" || fail "cannot cd $proj"
    say "demo 06d: runtime CUDA guard fires on real hardware"
    cap 06 guard-render -- lazy107 render
    awk '{ if ($0 ~ /^python - <</) print "export CUDA_VISIBLE_DEVICES=\"\"" } { print }' \
        scripts/train.sbatch > scripts/train.sbatch.edited
    mv scripts/train.sbatch.edited scripts/train.sbatch
    {
        echo "\$ sbatch scripts/train.sbatch"
        sbatch scripts/train.sbatch 2>&1
        echo "--- exit: $?"
    } | tee "$d/guard-trip.txt"
    id="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' "$d/guard-trip.txt" | tail -n1)"
    [ -n "$id" ] || fail "sbatch produced no job id (see $d/guard-trip.txt)"
    wait_and_sacct 06 guard-trip "$id"
    cap 06 guard-debug -- lazy107 debug "$id"
}

main() {
    local demo="${1:-}"
    case "$demo" in
        01) demo01 ;;
        02) demo02 ;;
        03) demo03 ;;
        04) demo04 ;;
        05) demo05 ;;
        06) demo06 ;;
        all) demo01; demo02; demo03; demo04; demo05; demo06 ;;
        *)
            echo "usage: $0 <01..06|all>" >&2
            echo "captures land in captures/<demo>/; paste them into docs/demos/<demo>-*.md" >&2
            return 1
            ;;
    esac
    echo ""
    echo "captures written under $CAPS — copy each .txt into the matching docs/demos/*.md output block."
}

main "$@"
