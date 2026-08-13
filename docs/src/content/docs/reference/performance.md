---
title: Performance
description: How caps-sa compares to the upstream C++ CaPS-SA reference.
---

All numbers below are **suffix-array build time only** (file I/O excluded), measured against the upstream C++ CaPS-SA reference. Full methodology and the optimisation ladder are in [`bench/README.md`](https://github.com/COMBINE-lab/caps-sa/blob/main/bench/README.md).

## External-memory path

| Input | Threads | caps-sa | upstream C++ |
| --- | --- | --- | --- |
| Yeast · 12 MB | 4 | **0.99 s** | 3.94 s |
| Random DNA · 100 MB | 4 | **11.39 s** | 12.17 s |
| Human GRCh38 · 3.1 GB | 32 | **10.47 min** | 10.93 min |
| GRCh38 peak RAM | 32 | **5.03 GB** | 6.46 GB |

On the human genome (GRCh38, 32 threads, AMD EPYC 9575F) caps-sa is **~7% faster** than upstream's external-memory path and uses **~23% less RAM**.

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

See [`ExtMemOpts`](/caps-sa/reference/api/#extmemopts) for the full set of knobs.
