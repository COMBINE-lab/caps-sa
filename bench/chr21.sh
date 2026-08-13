#!/usr/bin/env bash
# Reproduce the chr21 numbers quoted in bench/README.md.
#
#   bench/chr21.sh [work-dir] [threads]
#
# Builds two inputs from hg38 chr21, because they exercise different costs and
# conflating them is what made the original slowdown report hard to read:
#
#   chr21.0123   forward ++ reverse complement, one byte per base, codes 0..=3,
#                ambiguous bases dropped. ~80 MB, alphabet size 4, no long runs.
#                This is the input libsais is usually benchmarked on.
#   chr21.fa     the raw FASTA, headers and newlines included. ~45 MB, and it
#                still contains its ~6.6 Mb of `N`. Wrapped at 60 columns, so
#                the `N` blocks are a period-61 repeat rather than a plain run.
#
# The second is the realistic one and the one that used to be pathological: a
# comparison-based suffix sort scans the whole shared prefix on every tied
# comparison, so an `N` block costs megabytes per comparison.
set -euo pipefail

work_dir="${1:-bench/work}"
threads="${2:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$work_dir"
gz="$work_dir/chr21.fa.gz"
fa="$work_dir/chr21.fa"
bin="$work_dir/chr21.0123"

if [ ! -f "$gz" ]; then
  echo "== downloading hg38 chr21 ==" >&2
  curl -sSL -o "$gz" \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
fi
[ -f "$fa" ] || gzip -dc "$gz" > "$fa"

if [ ! -f "$bin" ]; then
  echo "== encoding forward ++ revcomp as codes 0..=3 ==" >&2
  python3 - "$gz" "$bin" <<'PY'
import gzip, sys
code = {"A": 0, "C": 1, "G": 2, "T": 3}
comp = {0: 3, 1: 2, 2: 1, 3: 0}
fwd = bytearray()
with gzip.open(sys.argv[1], "rt") as fh:
    for line in fh:
        if line.startswith(">"):
            continue
        for ch in line.strip().upper():
            c = code.get(ch)
            if c is not None:
                fwd.append(c)
rc = bytearray(comp[b] for b in reversed(fwd))
with open(sys.argv[2], "wb") as out:
    out.write(fwd)
    out.write(rc)
print(f"{sys.argv[2]}: {len(fwd) + len(rc)} bytes", file=sys.stderr)
PY
fi

echo "== building (fat LTO, target-cpu=native) ==" >&2
RUSTFLAGS="-C target-cpu=native" cargo build --release --example caps_sa --manifest-path "$root/Cargo.toml"
caps_sa="$root/target/release/examples/caps_sa"

run() {
  local label="$1" input="$2"
  shift 2
  # `--verify` is an O(n) independent check of the result; it is timed and
  # reported separately by the binary, so it never inflates the build time.
  printf '%-28s ' "$label"
  "$caps_sa" "$input" /dev/null --threads "$threads" --verify "$@" 2>&1 |
    awk '/^build:/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9.]+s$/) b = $i }
         /^verify:/ { v = $0 }
         END { printf "build %-9s %s\n", b, (v ~ /OK/ ? "verify OK" : "VERIFY FAILED") }'
}

echo
echo "threads: $threads"
echo
run "0123 (80 MB, no N)" "$bin"
run "FASTA (45 MB, 6.6 Mb N)" "$fa"
echo
echo "For wall+CPU together, wrap a single run:" >&2
echo "  /usr/bin/time -p $caps_sa $bin /dev/null --threads $threads" >&2
