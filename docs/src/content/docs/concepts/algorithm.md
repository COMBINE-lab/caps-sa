---
title: The algorithm
description: How CaPS-SA builds a suffix array — LCP-enhanced merge, sample sort, and the external-memory pipeline.
---

caps-sa is a Rust port of **CaPS-SA** (Khan et al., WABI 2023). This page sketches the three ideas that make it fast: an LCP-enhanced merge, a sample-sort wrapper, and an external-memory pipeline that streams the result.

## 1. LCP-enhanced merge

The in-memory kernel is a **parallel merge-sort** over suffix positions. The trick is in the two-way merge: an **LCP array** travels alongside each sorted run, recording the longest common prefix between each suffix and its predecessor in that run.

When merging two runs, the carried LCPs let the merge decide the order of the two front candidates in **`O(1)` in two of three cases**, falling back to a symbol-by-symbol scan only when the carried LCP exactly equals the current boundary. This is what avoids the `O(n²)` blow-up of a naive suffix comparison sort on repetitive text. The three-case analysis lives in `src/sample_sort.rs::merge`.

When a fallback scan *is* needed, it runs through the **SIMD LCP fast path** (see below).

## 2. Sample sort

For inputs too large for a single merge-sort pass, caps-sa wraps the kernel in a **sample sort** — the same structure for both the in-memory (`build_in_memory_sample_sort`) and external-memory (`build_ext_mem`) paths. With `p` subproblems:

1. **Sort + sample + spill.** Split the positions into `p` subarrays, sort each with the in-memory kernel in parallel, sample `~c·ln n` suffixes uniformly from each, and spill the sorted subarray to a bucket.
2. **Select pivots.** Sort the pooled samples and pick `p − 1` evenly-spaced pivots. These define `p` partition ranges that together cover the whole SA.
3. **Distribute.** Binary-search each sorted subarray against the pivots and route its sub-slices into the matching partition's bucket.
4. **Per-partition merge.** Load each partition's bucket, cascade 2-way LCP-enhanced merges over its sub-slices, and emit the resulting sorted positions through the caller's closure.

Because the partitions are globally ordered, emitting them in turn yields the full SA in lexicographic order, and **peak RAM stays bounded at `~O(text + n/p)` per worker** regardless of input size.

## 3. External memory

The external-memory path (`build_ext_mem`) is the default for production-scale genomes. The buckets in steps 1 and 3 are **disk-spilling**: sorted runs are written to a pool of temporary files instead of held in RAM. Positions are read back partition-by-partition only when that partition is merged, then streamed straight out — the suffix array is **never fully materialised in memory**.

The bucket pool collapses the `2 × p` logical buckets onto a small set of physical temp files (one per worker by default) to keep kernel-level write contention bounded. Tuning knobs — subproblem count, working directory, physical file count — are on [`ExtMemOpts`](/caps-sa/reference/api/#extmemopts).

:::note[Positioned I/O]
The pooled bucket path uses positioned reads/writes (`pread`/`pwrite` on Unix, `seek_read`/`seek_write` on Windows) so many workers can share one file handle without a shared cursor. It is portable across Unix and Windows as of v0.6.1.
:::

## The SIMD LCP kernel

Every path shares one LCP routine, selected once per build via `LcpDispatch::detect()` and threaded into the inner loop as a function pointer — no per-call feature detection. The dispatch ladder is **AVX-512BW hybrid → AVX2 → NEON → scalar**.

A single byte-level SIMD compare backs **every symbol width**: an AVX-512 byte-compare followed by `byte_lcp / size_of::<S>()` recovers the symbol-LCP for `u16`, `u32`, `[u8; 3]`, `u64`, and any other `Symbol`. Measured on a Zen 5 host this lifts the LCP function from ~200 ms scalar to **4–29 ms** across widths (7× on `u64` up to 45× on `u8` for a 1 M-symbol long-LCP microbenchmark).

## Generalized suffix arrays

caps-sa builds a *standard* SA over a single text. To build a **generalized** SA over many sequences (e.g. all chromosomes of a genome, each terminated so suffixes don't run across boundaries), rewrite the text with distinct sentinels — the standard SA of the transformed text is the generalized SA you want.

For callers that can't afford the sentinel bytes, a [`LimitProvider`](/caps-sa/reference/api/#segmented-texts) lets the merge **stop LCP scans at segment boundaries** directly. `SegmentedText` carries the cumulative segment ends; `PlainText` (the default) imposes no boundaries and monomorphizes to the same assembly as the un-segmented path.

## Reference

- Upstream C++ implementation: <https://github.com/jamshed/CaPS-SA>
- Paper: Khan et al., *CaPS-SA: A Practical Algorithm for Parallel Suffix Array Construction.* WABI 2023. <https://doi.org/10.4230/LIPIcs.WABI.2023.16>
