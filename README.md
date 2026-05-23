# caps-sa

A pure-Rust implementation of **CaPS-SA** (Khan et al., WABI 2023), a
cache-friendly, parallel, sample-sort-based suffix array constructor.

The crate is generic over the symbol type (`u8`, `u16`, …; any `Ord +
Copy`) and the index type (`u32`, `u64`, `usize`), produces a standard
lexicographic suffix array, and is intended to scale to very large
inputs — human-genome scale (≈ 6 × 10⁹ symbols) — via an external-
memory sample-sort path.

## Status

- **Phase 1 — in-memory:** ✅ implemented. Parallel merge-sort with an
  LCP-enhanced two-way merge (the inner sorting kernel of upstream
  CaPS-SA). Differentially tested against a brute-force suffix array on
  small and random inputs. Suitable for genomes up to roughly
  chromosome-22 scale (low-hundreds-of-millions of suffixes).
- **Phase 2 — external memory:** planned. Disk-spilling buckets
  (`Ext_Mem_Bucket`) and sample-sort partitioning around the in-memory
  kernel. This is the path that brings peak RAM well below standard
  SA-construction tools for large inputs.
- **Phase 3 — polish:** SIMD-unrolled LCP comparison; wider integer
  alphabets via a `u16` text path.

## Example

```rust
use caps_sa::build_in_memory;

let text = b"banana";
let sa: Vec<u32> = build_in_memory(text);
// sa is the standard lexicographic suffix array of `text`.
// `build_in_memory` is generic — pick `u32`, `u64`, or `usize` for the
// index type to match the size of your input.
```

## Algorithm

The in-memory kernel is a parallel merge-sort where the two-way merge
uses an **LCP-enhanced comparison**. Carrying an LCP array alongside
each sorted run, the merge decides the order of two candidates in O(1)
for most steps and only falls back to a symbol-by-symbol scan when the
carried LCP equals the current boundary. The three-case analysis is in
`src/sample_sort.rs::merge`.

## Reference

- Upstream reference C++ implementation:
  <https://github.com/jamshed/CaPS-SA>
- Paper: Khan et al., *CaPS-SA: A Practical Algorithm for Parallel
  Suffix Array Construction.* Workshop on Algorithms in Bioinformatics
  (WABI 2023). <https://doi.org/10.4230/LIPIcs.WABI.2023.16>

## License

MIT, matching upstream CaPS-SA. See [`LICENSE`](LICENSE).
