#!/usr/bin/env bash
# Reproducible benchmark of caps-sa (Rust) vs upstream CaPS-SA (C++).
#
# Usage:
#   bench/run.sh <input.txt> <threads>
#
# Expects, in PATH or via env vars:
#   $CAPS_SA_UPSTREAM   — upstream C++ caps_sa binary (default: ./CaPS-SA-upstream/build/src/caps_sa)
#   $CAPS_SA_RUST       — rust caps-sa example binary (default: ./target/release/examples/caps_sa)
#   $UPSTREAM_LD_PATH   — extra LD_LIBRARY_PATH for the upstream binary (e.g. a newer libstdc++)

set -euo pipefail

INPUT="${1:?usage: run.sh <input.txt> <threads>}"
THREADS="${2:?usage: run.sh <input.txt> <threads>}"

CAPS_SA_UPSTREAM="${CAPS_SA_UPSTREAM:-./CaPS-SA-upstream/build/src/caps_sa}"
CAPS_SA_RUST="${CAPS_SA_RUST:-./target/release/examples/caps_sa}"
UPSTREAM_LD_PATH="${UPSTREAM_LD_PATH:-}"
WORK_DIR="$(mktemp -d -t capsbench-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

INPUT_BYTES=$(stat -c %s "$INPUT")

run_one() {
    # $1 = label, $2 = output path, rest = command
    local label="$1"; shift
    local output="$1"; shift
    rm -f "$output"
    # /usr/bin/time prints to stderr; capture wall + RSS.
    local time_fmt="wall_s=%e rss_kb=%M"
    local time_log
    time_log=$(mktemp)
    if /usr/bin/time -o "$time_log" -f "$time_fmt" "$@" 2>/dev/null; then
        local stat="$(cat "$time_log")"
        rm -f "$time_log"
        printf '%-32s  %s\n' "$label" "$stat"
    else
        rm -f "$time_log"
        printf '%-32s  FAILED\n' "$label"
    fi
}

printf 'input=%s bytes=%d threads=%d\n\n' "$INPUT" "$INPUT_BYTES" "$THREADS"

# Each run is fresh — output written to WORK_DIR. We don't compare bytes here
# (upstream uses 32-bit indices, ours uses 64-bit), but the algorithms
# produce equivalent suffix arrays modulo index width.

LD_LIBRARY_PATH="${UPSTREAM_LD_PATH}:${LD_LIBRARY_PATH-}" \
PARLAY_NUM_THREADS="$THREADS" \
run_one "upstream c++ (in-mem)"   "$WORK_DIR/upstream.sa" \
    "$CAPS_SA_UPSTREAM" "$INPUT" "$WORK_DIR/upstream.sa" --data-type t

LD_LIBRARY_PATH="${UPSTREAM_LD_PATH}:${LD_LIBRARY_PATH-}" \
PARLAY_NUM_THREADS="$THREADS" \
run_one "upstream c++ (ext-mem)"  "$WORK_DIR/upstream_ext.sa" \
    "$CAPS_SA_UPSTREAM" "$INPUT" "$WORK_DIR/upstream_ext.sa" --data-type t --ext-mem --collate-extmem-result

run_one "rust caps-sa (in-mem)"   "$WORK_DIR/rust.sa" \
    "$CAPS_SA_RUST" "$INPUT" "$WORK_DIR/rust.sa" --threads "$THREADS"

run_one "rust caps-sa (ext-mem)"  "$WORK_DIR/rust_ext.sa" \
    "$CAPS_SA_RUST" "$INPUT" "$WORK_DIR/rust_ext.sa" --ext-mem --threads "$THREADS"

echo
echo "(output files in $WORK_DIR are removed at exit)"
