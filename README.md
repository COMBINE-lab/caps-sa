# caps-sa

A pure-Rust implementation of **CaPS-SA** (Khan et al., WABI 2023), a
cache-friendly, parallel, sample-sort-based suffix array constructor.

The crate is generic over the symbol type (`u8`, `u16`, `u32`, `u64`,
`[u8; N]`, … — anything implementing the [`Symbol`] trait) and the
index type (`u32`, `u64`, `usize`), produces a standard lexicographic
suffix array, and scales to human-genome inputs (≈ 6 × 10⁹ symbols) on
commodity hardware via an external-memory sample-sort path that
streams the SA out as positions are emitted.

📖 **Documentation:** <https://combine-lab.github.io/caps-sa/>

## Status

Both the in-memory and external-memory paths are implemented, tested,
and benchmarked. 73 unit tests pass and the SA output is differentially
verified against a brute-force reference on small and random inputs, and
against [`verify_sa`](#verifying-a-suffix-array) at genome scale.

### In-memory fast path

`build_in_memory` on a byte text routes through a **radix-seeded prefix
doubling** algorithm rather than the merge kernel. The merge kernel is
still the general path and still backs everything else; the fast path is
taken only when the comparator is provably plain lexicographic (see
[Choosing a path](#choosing-a-path)).

The reason is that a comparison-based suffix sort pays twice on real
genomic input. It performs `n log n` merge steps, and every tied step
scans the shared prefix of two suffixes from the beginning. Genome FASTA
carries megabyte-scale runs of `N` — period-61 once 60-column line
wrapping is included — so a single comparison can scan millions of
bytes. Measured on chr21 that drives the cost per merge step from 13 ns
to 222 ns, a 16x penalty that is entirely scan time.

The fast path sorts by a packed fixed-depth key, then resolves what
remains by doubling on ranks. The packing picks the narrowest field
width that holds the alphabet, so DNA over `{0,1,2,3}` resolves 32
symbols per key rather than the 8 a raw byte key gives. After the seed,
no comparison reads the text again, so a megabyte-long run of `N` costs
exactly what random DNA costs.

Apple M4 Max (12 P-cores), 12 threads, suffix arrays byte-identical to
the merge kernel's and independently verified:

| input | before | after | CPU before | CPU after |
| ----- | ------ | ----- | ---------- | --------- |
| chr21 fwd ++ revcomp, ASCII `ACGT`, 80 MB | 6.08 s | **0.61 s** | 28.1 s | 5.0 s |
| same, pre-coded to `0..3`, 80 MB | 6.08 s | **0.57 s** | 28.1 s | 5.0 s |
| chr21 FASTA, 47.5 MB, 6.6 Mb of `N`   | 27.8 s | **1.04 s** | 283.5 s | 5.2 s |
| same, via `build_in_memory_sample_sort` | 3.41 s | **1.18 s** | 34.5 s | 7.2 s |

Peak RSS on the 80 MB input is 2.21 GB, down from 2.83 GB, because the
seed is an MSD counting sort that recomputes keys from the text rather
than materialising a key array.

The two DNA rows above land in the same place because the alphabet is
ranked to a dense code range before packing. Without that step the ASCII
row would use 8-bit fields — its largest byte is `'T'` (84) even though
it has four symbols — fitting 8 symbols per key instead of 32, and would
take 1.40 s rather than 0.61 s. Keys are then built by a SWAR gather
over the ranked text, so eight symbols cost three shift-or-mask pairs
instead of eight dependent shift-or-lookup steps.

Note the two inputs are different problems; benchmarking one
implementation on the first and another on the second is not a
comparison. `bench/chr21.sh` prepares both.

### External memory

On the human genome (GRCh38, 32 threads on AMD EPYC 9575F), caps-sa is
**7% faster than upstream CaPS-SA's ext-mem path** and uses **23% less
RAM**, while beating upstream's in-mem wall time by 3% at 1/10 of the
RAM. See [`bench/README.md`](bench/README.md) for the full methodology
and the optimisation ladder that got us there. Those paths still use the
merge kernel, so they retain the scan cost described above on
repeat-heavy input.

On the human genome (GRCh38, 32 threads on AMD EPYC 9575F), caps-sa is
**7% faster than upstream CaPS-SA's ext-mem path** and uses **23% less
RAM**, while beating upstream's in-mem wall time by 3% at 1/10 of the
RAM. See [`bench/README.md`](bench/README.md) for the full methodology
and the optimisation ladder that got us there.

The crate ships four entry points sharing one LCP-enhanced merge
kernel:

| Path                                                  | API                                  |
| ----------------------------------------------------- | ------------------------------------ |
| In-memory, parallel merge-sort                        | `build_in_memory`                    |
| In-memory, sample-sort (alternative for huge `n`)     | `build_in_memory_sample_sort`        |
| External-memory, disk-spilling sample-sort            | `build_ext_mem`                      |
| Any of the three above, restricted to a subset        | `*_for_positions`                    |

All four paths share the same SIMD LCP fast path (AVX-512BW hybrid →
AVX2 → NEON → scalar), selected once per build entry via
`LcpDispatch::detect()` and threaded into the inner loop as a function
pointer — no per-call feature-detect overhead. The same byte-level
SIMD function backs **every symbol width** via a byte-view dispatch in
`LcpDispatch::lcp<S: Symbol>`: a single AVX-512 byte-compare followed
by `byte_lcp / size_of::<S>()` recovers the symbol-LCP for `u16`,
`u32`, `[u8; 3]`, `u64`, and any other `Symbol`. Measured on a Zen 5
host this lifts the LCP function from ~200 ms scalar to 4–29 ms SIMD
across widths (7× on `u64` to 45× on `u8` for a 1 M-symbol long-LCP
microbench).

## Example

```rust
use caps_sa::build_in_memory;

let text = b"banana";
let sa: Vec<u32> = build_in_memory(text);
// `sa` is the standard lexicographic suffix array of `text`. The
// index type is generic — pick `u32`, `u64`, or `usize` for your input.
```

For large inputs, stream the SA from disk-spilling buckets so the
output is never fully materialised in RAM:

```rust
use caps_sa::{ExtMemOpts, build_ext_mem};

let opts = ExtMemOpts::default();
build_ext_mem(&text, &opts, |sa_pos| {
    // `sa_pos` is the next suffix position in lex order.
    // The caller streams these straight to disk / a packed array.
    Ok(())
})?;
```

For workflows that sort only a subset of positions (e.g. STAR-style
genome indexing where many positions are filtered out — N's, spacers),
hand only the positions you want sorted to `*_for_positions`. The
others never enter the sort:

```rust
use caps_sa::build_ext_mem_for_positions;

let positions: Vec<u64> =
    (0..text.len() as u64).filter(|&p| text[p as usize] < 4).collect();
build_ext_mem_for_positions(&text, positions, &opts, |sa_pos| {
    Ok(())
})?;
```

### Verifying a suffix array

`verify_sa` checks a candidate in `O(n)` without re-running any
construction algorithm and without depending on LCP length, so it stays
usable on the repetitive inputs that are hardest to trust:

```rust
use caps_sa::{build_in_memory, verify_sa};

let text = b"banana";
let sa: Vec<u32> = build_in_memory(text);
assert!(verify_sa(text, &sa).is_ok());
```

It inverts `sa` to get ranks, then checks that
`(text[p], rank[p + 1])` increases strictly along it, with `rank[n]`
treated as smaller than every real rank. A permutation of `0..n`
satisfies that condition exactly when it is the suffix array. The bench
CLI exposes it as `--verify`.

## Choosing a path

`build_in_memory` takes the radix-seeded doubling fast path only when
the requested comparator is provably the plain lexicographic one. All
three conditions are soundness requirements, and each defaults to
declining:

| Condition | Why |
| --------- | --- |
| `Opts::max_context` unbounded | A finite bound makes the merge comparator fall through to `LimitProvider::boundary_order`, which compares *lengths*, so it is not lexicographic. |
| `LimitProvider::plain_lex_len()` reports the full text | Rules out `SegmentedText`, whose scans stop at segment boundaries, and any custom `boundary_order`. |
| symbol type is exactly `u8` | Packing wider symbols into an order-preserving key is endianness-dependent: on a little-endian host `0x0100 > 0x0001` as `u16` values, but their byte views compare the other way. |

`plain_lex_len` is a new `LimitProvider` method that defaults to `None`.
An implementation that delegates `lim_at` to `PlainText` but overrides
`boundary_order` for a different convention — STAR's spacer-as-largest
ordering is the motivating example — inherits `None` and keeps today's
semantics without changing a line.

Given those, the fast path also covers two cases beyond a plain whole-text
build:

- **`*_for_positions` subsets.** Doubling cannot be restricted to a subset
  directly, since a round compares `rank[p + d]` and that successor is
  generally outside the subset, so ranks must exist for every text
  position. The full array is built and filtered in one `O(n)` pass
  instead. Below one eighth of the text this declines and the merge kernel
  runs, since building and discarding a whole array would cost more than
  sorting a small subset. That ratio is a performance heuristic, not a
  correctness condition. Duplicate or out-of-range positions also decline,
  because the output is a permutation of the input *multiset* and a
  membership filter cannot reproduce that.
- **`build_in_memory_sample_sort`.** This path exists to sort in RAM, so
  where doubling applies it is strictly better: same output, no bucket
  machinery, none of the scan cost.

`build_ext_mem` deliberately stays on the merge kernel: its purpose is to
bound peak memory, and prefix doubling needs a rank for every position in
the text, which would defeat exactly that. Segmented texts and symbols
wider than `u8` also stay on the merge kernel.

### Skipping long repeats

Those paths get the same pathology fixed in the comparator instead, which
costs no extra memory. If `text[s..e)` has period `q` and two suffixes
start at `a < b` inside it with `(b - a) % q == 0`, they agree until the
later one reaches `e`, so

```text
lcp(a, b) >= e - b
```

is known in `O(1)` from the run's bounds with nothing scanned. When the
phase does not match, the two suffixes differ within `q` symbols and the
ordinary scan is already short. Scans are additionally bounded so they
stop at a run's start rather than traversing it.

Detecting only single-symbol runs would miss the case that actually
occurs: in wrapped FASTA an `N` block is 60 `N`s followed by a newline,
which is period 61, not period 1. Periods up to 64 are considered.
Measured on synthetic periodic inputs, periods 1, 2, 61 and 64 sort
5.8-6.9x faster, while periods 65 and 171 are not detected and run at
parity, so alpha-satellite arrays (canonical monomer 171 bases) fall
outside the detector.

Detection is two-stage so texts without repeats pay almost nothing: a
sampling pass collects the periods that occur at all, and the full scan
runs only for those. On `N`-free DNA the table comes out empty and every
query short-circuits. The table is a few dozen entries, so the
external-memory path keeps its memory bound.

| ext-mem input | before | after | CPU before | CPU after | peak RSS |
| ------------- | ------ | ----- | ---------- | --------- | -------- |
| chr21 FASTA, 47.5 MB | 24.2 s | **1.47 s** | 268 s | 17.5 s | 147 → 190 MB |
| chr21 `N`-free, 80 MB | 3.49 s | **1.65 s** | 33.9 s | 15.1 s | 214 → 285 MB |

Seven changes get there, each measured separately:

| change | chr21.0123 | chr21 FASTA |
| ------ | ---------- | ----------- |
| baseline | 3.49 s | 24.19 s |
| skip periodic runs | 3.55 s | 2.48 s |
| prefetch the next candidates' text | 2.90 s | 1.99 s |
| subarray target 64Ki → 128Ki records | 2.54 s | 1.86 s |
| seed phase-1 subarrays with the packed key | 2.11 s | 1.65 s |
| merge cascade run pairs in parallel | 2.02 s | 1.55 s |
| pipeline the emit against the next merge | 1.85 s | 1.45 s |
| re-sort run-free partitions by key | **1.65 s** | **1.47 s** |

Phase 1 goes from 22.85 s to 0.42 s on the FASTA input, and from 1.20 s
to 0.32 s on the `N`-free one. Peak RSS rises by under 30 MB, so the
bounded-memory guarantee the path exists for is intact.

## Algorithm

The in-memory kernel is a parallel merge-sort whose two-way merge uses
an **LCP-enhanced comparison**: an LCP array travels alongside each
sorted run, so the merge decides the order of two candidates in `O(1)`
in two of three cases and only falls back to a symbol-by-symbol scan
when the carried LCP equals the current boundary. The three-case
analysis is in `src/sample_sort.rs::merge`.

The external-memory path wraps that kernel in a sample-sort:

1. **Sort + sample + spill.** Split positions into `p` subarrays, sort
   each with the in-memory kernel in parallel, sample `~c·ln n`
   suffixes uniformly, spill each sorted subarray to a disk-spilling
   bucket.
2. **Select pivots.** Sort the pooled samples and pick `p − 1` evenly-
   spaced pivots, defining `p` partition ranges over the global SA.
3. **Distribute.** Binary-search each sorted subarray against the
   pivots and route sub-subarrays into the corresponding partition's
   bucket.
4. **Per-partition merge.** Load each partition's bucket into RAM,
   cascade 2-way LCP-enhanced merges over its sub-subarrays, emit the
   resulting sorted positions via the caller-supplied closure.

Peak RAM is bounded at `~O(text + n/p)` per worker regardless of
input size; the SA is never fully materialised — partitions are
streamed out in lex order.

## Performance — short version

(See [`bench/README.md`](bench/README.md) for the full numbers.)

| Input                        | Threads | caps-sa ext-mem | upstream ext-mem |
| ---------------------------- | ------- | --------------- | ---------------- |
| Yeast (12 MB)                | 4       | **0.99 s**      | 3.94 s           |
| Random DNA 100 MB            | 4       | **11.39 s**     | 12.17 s          |
| Human genome GRCh38 (3.1 GB) | 32      | **10.47 min / 5.03 GB** | 10.93 min / 6.46 GB |

The in-memory sample-sort path (`build_in_memory_sample_sort`) is
available for hosts with enough RAM to skip disk entirely; on the
human genome it benches at 11.64 min / 55 GB — same wall, ~10× the
RAM, useful only when disk is the constraint.

## Reference

- Upstream reference C++ implementation:
  <https://github.com/jamshed/CaPS-SA>
- Paper: Khan et al., *CaPS-SA: A Practical Algorithm for Parallel
  Suffix Array Construction.* Workshop on Algorithms in Bioinformatics
  (WABI 2023). <https://doi.org/10.4230/LIPIcs.WABI.2023.16>

## License

MIT, matching upstream CaPS-SA. See [`LICENSE`](LICENSE).
