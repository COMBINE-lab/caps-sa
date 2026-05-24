# caps-sa

A pure-Rust implementation of **CaPS-SA** (Khan et al., WABI 2023), a
cache-friendly, parallel, sample-sort-based suffix array constructor.

The crate is generic over the symbol type (`u8`, `u16`, …; any `Ord +
Copy`) and the index type (`u32`, `u64`, `usize`), produces a standard
lexicographic suffix array, and scales to human-genome inputs (≈ 6 ×
10⁹ symbols) on commodity hardware via an external-memory sample-sort
path that streams the SA out as positions are emitted.

## Status

Both the in-memory and external-memory paths are implemented, tested,
and benchmarked. 35 unit tests pass and the SA output is differentially
verified against a brute-force reference on small and random inputs.

On the human genome (GRCh38, 32 threads on AMD EPYC 9575F), caps-sa is
**4% faster than upstream CaPS-SA's ext-mem path** and uses **22% less
RAM**, while matching upstream's in-mem wall time at 1/10 of the RAM.
See [`bench/README.md`](bench/README.md) for the full methodology and
the optimisation ladder that got us there.

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
pointer — no per-call feature-detect overhead.

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
