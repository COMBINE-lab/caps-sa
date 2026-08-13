---
title: Performance
description: Production-shaped caps-sa measurements and the standard upstream comparison.
---

Unless stated otherwise, numbers are suffix-array construction time and output
is streamed to a hash sink. The current production-shaped measurements use 32
pinned physical cores on an AMD EPYC 9555. Every candidate emitted the same
position count and output hash.

## caps-sa 0.7.0 on the ruSTAR workload

The current integration benchmark preserves the generalized, filtered suffix
array that ruSTAR actually requests:

- GENCODE Human v50 GRCh38 primary-assembly FASTA and comprehensive
  primary-assembly GTF;
- `sjdbOverhang=100` and all 698,597 deduplicated splice junctions;
- 6,557,611,930 symbols, 1,397,582 segments, and STAR boundary ordering;
- 6,176,694,310 retained ACGT-starting suffixes;
- external-memory `u64`, 8,192 partitions, 32 physical cores.

| Implementation | Wall | User CPU | Peak RSS | Phase 1 | Phase 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pre-pass 0.7 baseline | 267.592 s | 7,622.88 s | 10,512,408 KiB | 106.847 s | 155.794 s |
| caps-sa 0.7.0 | **172.953 s** | **4,731.05 s** | **9,169,892 KiB** | **49.029 s** | **118.743 s** |

The release is **35.4% faster** and uses **12.8% less peak RSS**. The retained
changes fuse external sorting and distribution, prefetch upcoming text
positions during merge, avoid transient record conversions, use task-local
phase-1 ping-pong storage at high outer parallelism, and add a bounded coarse
directory for large `SegmentedText` boundary sets.

On the focused chromosome-21 backbone plus every GENCODE-derived junction
flank, stable measurements improved from 11.97–12.00 seconds to 7.77–7.80
seconds. All 359,616,038 emitted positions matched the reference.

## Optional packed-prefix phase-1 seed

The opt-in packed-prefix seed was measured against the current 0.7.0 main on
the same complete fixture, with one warm-up followed by three interleaved
measured runs. Values below are medians:

| Configuration | Build | User CPU | Peak RSS | Phase 1 | Phase 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct LCP | 198.924 s | 5,153.25 s | 9,142,300 KiB | 49.038 s | 144.923 s |
| Geometric memo only | 171.205 s | 4,684.95 s | 9,161,688 KiB | 49.034 s | 117.021 s |
| Packed seed only | 162.650 s | 3,987.38 s | 9,527,240 KiB | 12.204 s | 145.187 s |
| Packed seed + geometric memo | **134.618 s** | **3,507.54 s** | 9,536,692 KiB | **12.179 s** | **117.265 s** |

The seed reduces phase 1 by 75.1% and improves the memoized ruSTAR
configuration by 21.4%. Memoization changes phase 4 by 19.25% without the seed
and 19.23% with it, confirming that the policies compose rather than competing
for the same work. Every run emitted 6,176,694,310 positions with ordered hash
`e81c8f9881e322148741a23c92ae2000`.

Peak RSS increased by 366–376 MiB (about 4.1%), matching the bounded
`(u64, u64)` key records held by the 32 active phase-1 tasks. The dense ruSTAR
alphabet required no ranked-text copy.

On chr21 without annotations, the direct build improved from a 1.291-second
median to 0.934 seconds (27.7%). On the chr21 backbone plus every annotation-
derived flank it improved from 7.709 to 6.025 seconds (21.8%), and the latter
matched all 359,616,038 reference positions exactly.

## Geometric memoization

The table above has geometric memoization enabled on both sides; do not add its
isolated percentage to the 35.4% release improvement. Its separate direct A/B
on the complete fixture measured:

| Policy | Wall | User CPU | Peak RSS |
| --- | ---: | ---: | ---: |
| Disabled | 291.710 s | 8,029.81 s | 10,672,468 KiB |
| Geometric defaults | **267.128 s** | **7,607.24 s** | **9,696,720 KiB** |

That is an **8.43% wall-time** and **5.26% user-CPU** improvement. Smaller or
less repetitive inputs were neutral or only slightly positive, so the feature
is opt-in. See [Geometric LCP memoization](/caps-sa/concepts/geometric-memoization/)
for selection and tuning guidance.

## Standard unsegmented comparison

The following older dataset compares caps-sa with the upstream C++ CaPS-SA
reference on standard, unsegmented suffix arrays. It is retained as a generic
implementation comparison; it is **not** the annotated ruSTAR workload above.
Full methodology and the optimization ladder are in
[`bench/README.md`](https://github.com/COMBINE-lab/caps-sa/blob/main/bench/README.md).

| Input | Threads | caps-sa | upstream C++ |
| --- | --- | --- | --- |
| Yeast · 12 MB | 4 | **0.99 s** | 3.94 s |
| Random DNA · 100 MB | 4 | **11.39 s** | 12.17 s |
| Human GRCh38 · 3.1 GB | 32 | **10.47 min** | 10.93 min |
| GRCh38 peak RAM | 32 | **5.03 GB** | 6.46 GB |

On that historical GRCh38 input (32 threads, AMD EPYC 9575F), caps-sa was
**~7% faster** than upstream's external-memory path and used **~23% less RAM**.

## In-memory sample-sort

`build_in_memory_sample_sort` skips disk entirely for hosts with the RAM to spare. On GRCh38 it benches at **11.64 min / 55 GB** — about the same wall time as the external-memory path at roughly 10× the RAM. Reach for it only when disk I/O is the constraint; otherwise the external-memory path is both faster and far lighter.

## The LCP kernel

Suffix comparison is the hot loop, and its cost is dominated by computing the longest common prefix of two suffixes. caps-sa selects one SIMD LCP backend per build — **AVX-512BW hybrid → AVX2 → NEON → scalar** — and threads it through the merge as a function pointer, so there is no per-comparison feature-detection overhead.

A single byte-level SIMD compare serves every symbol width through a byte-view dispatch. On a Zen 5 host, a 1 M-symbol long-LCP microbenchmark drops from **~200 ms scalar to 4–29 ms SIMD**:

| Symbol width | Speedup vs scalar |
| --- | --- |
| `u8` | ~45× |
| `u16` | ~30× |
| `u32` | ~15× |
| `u64` | ~7× |

## Tuning notes

- **`subproblem_count`** (`p`) auto-targets ~65,536 selected positions per subarray, bounded below by the worker count and above by 8,192. More subproblems mean smaller per-partition merges (lower peak residency) at the cost of more bucket bookkeeping.
- **`physical_file_count`** defaults to one temp file per worker. Raise it to reduce kernel write contention on fast local disks; lower it on networked filesystems with high metadata cost (or set `CAPS_SA_N_PHYS`).
- **`work_dir`** should point at the fastest local scratch available for the external-memory path.
- **`SegmentedText`** automatically adds a bounded coarse boundary directory
  for at least 256 segments. There is no tuning knob; the measured layout is
  capped at 8 MiB and avoids global boundary searches on generalized SAs with
  hundreds of thousands of short strings.
- **Geometric LCP memoization** is disabled by default. Enable
  `LcpMemoizationPolicy::geometric()` only after an A/B shows enough reused
  long contexts to repay table lookups; retain the measured thresholds unless
  workload-specific profiling supports a change.

See [`ExtMemOpts`](/caps-sa/reference/api/#extmemopts) for the full set of knobs.
