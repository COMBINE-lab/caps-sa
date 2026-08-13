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

Both the in-memory and external-memory paths are implemented, tested on
Linux, macOS, and Windows, and differentially verified against direct suffix
comparison on small, random, segmented, filtered, and finite-context inputs,
and against [`verify_sa`](#verifying-a-suffix-array) at genome scale.

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
| chr21 fwd ++ revcomp, `N`-free, 80 MB | 6.08 s | **0.89 s** | 28.1 s | 5.0 s |
| chr21 FASTA, 47.5 MB, 6.6 Mb of `N`   | 27.8 s | **1.04 s** | 283.5 s | 5.2 s |

Note the two inputs are different problems; benchmarking one
implementation on the first and another on the second is not a
comparison. `bench/chr21.sh` prepares both.

### External memory

The external-memory and sample-sort paths still use the merge kernel, so
they retain the scan cost described above on repeat-heavy input.

On the complete ruSTAR-shaped GENCODE Human v50 input (6.56 billion text
symbols, 6.18 billion retained suffixes, and 1.40 million segments), caps-sa
0.7.0 builds the generalized, filtered suffix array in **172.953 s** at 32
physical cores with **8.75 GiB peak RSS**. That is 35.4% faster and 12.8% less
memory than the pre-pass 0.7 baseline, with identical output. See the
[performance guide](https://combine-lab.github.io/caps-sa/reference/performance/)
for the production-shaped and upstream-comparison results.

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

Inputs with many repeated long contexts can opt into bounded geometric LCP
memoization during the final partition merges:

```rust
use caps_sa::LcpMemoizationPolicy;

let opts = ExtMemOpts::default()
    .lcp_memoization(LcpMemoizationPolicy::geometric());
```

The direct path remains the default: memoization pays only when the workload
contains enough repeated long contexts. See the
[user guide](https://combine-lab.github.io/caps-sa/concepts/geometric-memoization/)
for when to enable it and how to tune it, and
[`docs/geometric-memoization.md`](docs/geometric-memoization.md) for the full
design and measurement record.

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

Everything else (`*_for_positions`, `build_in_memory_sample_sort`,
`build_ext_mem`, segmented texts, wider symbols) runs on the CaPS-SA
merge kernel exactly as before.

## Algorithm

The in-memory kernel is a parallel merge-sort whose two-way merge uses
an **LCP-enhanced comparison**: an LCP array travels alongside each
sorted run, so the merge decides the order of two candidates in `O(1)`
in two of three cases and only falls back to a symbol-by-symbol scan
when the carried LCP equals the current boundary. The three-case
analysis is in `src/sample_sort.rs::merge`.

The external-memory path wraps that kernel in a sample-sort:

1. **Presample pivots.** Sort a small deterministic position sample and pick
   `p - 1` evenly-spaced pivots, defining the final partition ranges.
2. **Sort + distribute.** Split positions into `p` subarrays, sort each in an
   outer Rayon task, and write its pivot-delimited slices directly to their
   final disk-spilling partition buckets.
3. **Per-partition merge.** Load each partition's bucket into RAM,
   cascade 2-way LCP-enhanced merges over its sub-subarrays, emit the
   resulting sorted positions via the caller-supplied closure.

Peak RAM is bounded at `~O(text + n/p)` per worker regardless of
input size; the SA is never fully materialised — partitions are
streamed out in lex order.

## Performance — short version

(See the [performance guide](https://combine-lab.github.io/caps-sa/reference/performance/)
for definitions and the full measurement context.)

Current production-shaped measurement:

| Input | Threads | Wall | Peak RSS | Output |
| --- | ---:| ---:| ---:| ---:|
| ruSTAR-shaped GRCh38 + GENCODE v50, `u64`, segmented + ACGT-filtered | 32 physical | **172.953 s** | **8.75 GiB** | 6,176,694,310 suffixes |

Earlier standard, unsegmented suffix-array comparison against upstream C++:

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
