---
title: Introduction
description: What caps-sa is, what a suffix array is, and when to reach for it.
---

**caps-sa** is a pure-Rust implementation of **CaPS-SA** (Khan et al., *Cache-friendly, Parallel Suffix array construction*, WABI 2023) — a parallel, cache-friendly suffix array constructor built on sample sort with LCP-enhanced comparison.

It produces a **standard lexicographic suffix array**, is generic over both the symbol type and the index width, and scales to human-genome inputs (≈ 6 × 10⁹ symbols) on commodity hardware through an external-memory path that streams the sorted positions out as they are produced — so the full array never has to live in RAM.

## What is a suffix array?

For a text `T` of length `n`, the **suffix array** `SA` is the permutation of `0..n` that lists the starting positions of every suffix of `T` in lexicographic order. It is the workhorse index behind full-text search, the Burrows–Wheeler transform, MUMs/MEMs and seed-and-extend aligners, LCP-based repeat finding, and more.

```
T = b a n a n a            sorted suffixes        SA
    0 1 2 3 4 5            ─────────────────       ──
                           5  a                     5
                           3  a n a                 3
                           1  a n a n a             1
                           0  b a n a n a           0
                           4  n a                    4
                           2  n a n a                2
```

Building the SA is the expensive step: a naive comparison sort is `O(n²)` in the worst case because suffix comparisons can scan long shared prefixes. caps-sa keeps comparisons cheap by carrying an **LCP** (longest-common-prefix) array alongside each sorted run, so the merge usually decides an order in `O(1)`.

## Two ways to use it

caps-sa is delivered as two things that share one core:

- **The `caps-sa` library** — published on [crates.io](https://crates.io/crates/caps-sa), embedded in larger tools. It is the suffix-array backend for the genome indexer in [`rustar-aligner`](https://github.com/scverse/rustar-aligner).
- **The `caps_sa` CLI** — a small example binary (`examples/caps_sa.rs`) that reads a byte file, builds its suffix array, and writes packed positions to disk. It is the harness used for head-to-head benchmarks against the upstream C++ implementation.

See [The library & the CLI](/caps-sa/concepts/crates/) for how they relate.

## Which build path?

| Situation | Entry point | Notes |
| --- | --- | --- |
| Text fits comfortably in RAM | `build_in_memory` | Parallel merge-sort. Returns a `Vec<I>`. |
| Huge text, RAM-rich host | `build_in_memory_sample_sort` | Sample-sort, RAM-only buckets; streams positions. |
| Huge text, bounded RAM | `build_ext_mem` | Disk-spilling sample-sort; peak RAM `~O(text + n/p)`. |
| Only a subset of positions | any `*_for_positions` | Sort just the positions you pass; the rest never enter the sort. |

All paths produce the same lexicographic SA with the same "shorter suffix sorts first" tie-break, and share the same SIMD LCP kernel.

## Status

Both the in-memory and external-memory paths are implemented, unit-tested, and differentially verified against a brute-force reference on small and random inputs. On GRCh38 (32 threads, AMD EPYC 9575F) caps-sa is ~7% faster than upstream CaPS-SA's external-memory path while using ~23% less RAM.

Next: [Installation](/caps-sa/getting-started/installation/) · [Quick start](/caps-sa/getting-started/quick-start/) · [The algorithm](/caps-sa/concepts/algorithm/)
