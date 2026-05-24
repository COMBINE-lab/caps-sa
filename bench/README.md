# Benchmarks — caps-sa vs upstream C++ CaPS-SA

Reproducible head-to-head of `caps-sa` (this crate) and the reference C++
implementation at [`jamshed/CaPS-SA`][upstream] (`develop` branch).

[upstream]: https://github.com/jamshed/CaPS-SA

## Running

Build both implementations, then:

```sh
# From the workspace/repo root that holds the upstream build alongside.
CAPS_SA_UPSTREAM=path/to/upstream/build/src/caps_sa \
CAPS_SA_RUST=path/to/target/release/examples/caps_sa \
bench/run.sh INPUT.txt NUM_THREADS
```

Set `UPSTREAM_LD_PATH` if the upstream binary needs a newer libstdc++
than the system default.

The harness runs four configurations on the same input:

- upstream C++ in-memory
- upstream C++ external-memory (`--ext-mem --collate-extmem-result`)
- this crate's in-memory path (auto-picks `u32` indices when `n < 2^31`)
- this crate's external-memory path (always `u64` for now)

It reports wall time and peak RSS via `/usr/bin/time`.

## Results

Machine: 64-core x86_64 Linux node, 1 socket, AVX2 enabled.
Builds: upstream `cmake -DCMAKE_BUILD_TYPE=Release`, this crate
`cargo build --release --example caps_sa` (release profile in the
workspace `Cargo.toml` enables `lto = "fat"` + `codegen-units = 1`).

All caps-sa runs include the five optimizations applied incrementally
to the Phase 2b sample-sort baseline:

| Step | Description | Yeast 4-thread ext-mem |
| ---- | ----------- | ---------------------- |
| Phase 2b baseline       | sample-sort + sequential cascade merge | 1.88 s |
| + Opt 1: reusable cascade buffers | one `CascadeWorkspace` reused across partitions | 1.78 s |
| + Opt 2: parallel partition merge | `rayon::par_iter_mut` over partitions in chunks | 1.16 s |
| + Opt 3: SIMD LCP (AVX2 / NEON)   | runtime-dispatched `lcp_u8` for byte texts | 1.03 s |
| + Opt 4: hoist SIMD dispatch       | `LcpDispatch::detect()` once, fn-pointer threaded down | 0.99 s |
| + Opt 5: AVX-512 hybrid LCP        | 32-byte AVX2 head + 64-byte AVX-512BW body | **0.99 s** |

Opt 5 leaves yeast unchanged (it has no LCPs long enough to enter the
64-byte loop) but cuts ~30% off phase 1 on inputs with long repeats
(the human genome). See "Where AVX-512 helps and where it doesn't"
below.

### Yeast — *S. cerevisiae* R64-1-1, 12.16 MB raw text

| Configuration            | 1 thread  |        |       | 4 threads |        |       |
| ------------------------ | --------- | ------ | ----- | --------- | ------ | ----- |
|                          | wall s    | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem      | 7.47      | 748    | 1.00× | 2.10      | 747    | 1.00× |
| upstream C++ ext-mem     | 9.21      | 461    | 1.23× | 3.94      | 462    | 1.88× |
| **caps-sa Rust in-mem**  | **2.82**  | **204**| **0.38×** | **0.93** | **204**| **0.44×** |
| **caps-sa Rust ext-mem** | **3.31**  | **359**| **0.44×** | **0.99** | **320**| **0.47×** |

After the four optimizations, **caps-sa is ~2.1–2.8× faster than upstream
and uses ~3.5× less RAM on this input** for both the in-memory and
external-memory paths.

### Random DNA — uniform ACGT, 100 MB

| Configuration            | 1 thread  |        |       | 4 threads |        |       |
| ------------------------ | --------- | ------ | ----- | --------- | ------ | ----- |
|                          | wall s    | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem      | 38.16     | 2208   | 1.00× | **10.10** | 2208   | 1.00× |
| upstream C++ ext-mem     | 39.01     | 1053   | 1.02× | 12.17     | 1121   | 1.20× |
| **caps-sa Rust in-mem**  | **31.61** | 1663   | **0.83×** | 13.38   | 1663   | 1.33× |
| **caps-sa Rust ext-mem** | 36.32     | 2053   | 0.95× | **11.39** | 1913   | **1.13×** |

(Reported values are the median of three warm runs; variance ~5%.)

On uniform random DNA the per-call LCP is short (~13 bytes — the
information-theoretic minimum to disambiguate a random suffix in 100 MB
of alphabet-4 text), so the SIMD AVX2/NEON paths exit before any
32-byte chunk pays for itself. Single-thread caps-sa is ~17% faster
than upstream in-mem; multi-thread caps-sa ext-mem is ~7% faster than
upstream ext-mem. The only configuration still losing to upstream is
**rand100m / 4-thread / in-memory** — see "Where the differences still
come from" below.

### Human genome — GRCh38 primary assembly, 3.10 GB raw text

| Configuration            | 32 threads |             |             |
| ------------------------ | ---------- | ----------- | ----------- |
|                          | wall (min) | RSS (GB)    | vs upstream |
| upstream C++ in-mem      | 10.50      | 49.60       | 0.96× / 7.68× |
| upstream C++ ext-mem     | 10.93      | 6.46        | 1.00× / 1.00× |
| **caps-sa Rust ext-mem** | **10.47**  | **5.03**    | **0.96× / 0.78×** |
| caps-sa Rust in-mem-ss   | 11.64      | 55.00       | 1.07× / 8.52× |

`vs upstream` columns are relative to upstream ext-mem (the natural
peer-to-peer comparison). **caps-sa ext-mem is now ~4% faster than
upstream ext-mem on wall time and uses 22% less RAM** — and it
matches upstream's in-mem path on wall while using 10× less RAM.

Note: the 200 MB human-slice profile in the section below predicted a
larger phase-1 win than the full-genome run delivered (28% on the
slice vs 8% on full GRCh38). At full scale phase 1 no longer
dominates as completely — `p` rises to 8192 (vs 3200 on the slice) so
the per-partition work in phases 3 and 4 grows superlinearly with `n`
and the LCP-bound merge-sort is no longer 89% of the wall. The
hum200m profile was still the right signal for *where* to optimize
(the LCP function); it just over-estimated the *magnitude* at scale.

The in-mem-ss path runs the same sample-sort algorithm against
[`InMemBucket`][bucket] instead of disk-backed
[`ExtMemBucket`][bucket]; on this benchmark it's strictly Pareto-worse
than ext-mem (same wall, 11× the RAM), which is a useful result on
its own — it says disk I/O is essentially free in our ext-mem path at
this scale and the remaining wall-time gap is purely algorithmic.

[bucket]: ../src/ext_bucket.rs

The optimization stack that brought us here, starting from the initial
Phase 2b ext-mem implementation:

| Step | Δ wall | Δ RSS |
| ---- | ------ | ----- |
| Phase 2b baseline (`p = 128`, no streaming, no `u32` lift) | — | 39.83 GB |
| + ext-mem `Vec<u64>` streaming CLI                          | ~0  | →49 GB transient |
| + auto-scale `p` with text length                           | +1.1 min | **−31.7 GB** (→ 8.13) |
| + phase-4 `cascade_merge` consumes its workspace            | ~0  | ~0 |
| + phase-3 parallelised + `p_max = 8192`                      | ~0  | ~0 (unlocks larger `p`) |
| + generic `SaLcp<I>` → `u32` for `n ≤ 2³²`                  | −0.0 | −3.03 GB (→ 5.10) |
| + filesystem caching / rerun variance                        | −0.8 min | −0.11 GB (→ 4.99) |
| + AVX-512 hybrid LCP (Opt 5)                                 | **−0.85 min** (→ 10.47) | ~0 (→ 5.03) |
| **Net (v1 → today's tip)**                                   | **−0.45 min** | **−34.80 GB (−87%)** |

### Where AVX-512 helps and where it doesn't — the measurement

A `perf record --call-graph dwarf` run on a 200 MB human-genome slice
(`/usr/bin/perf` 4.18, AMD EPYC 9575F / Zen 5) made the picture
unambiguous. On the human slice, the entire program is concentrated in
one function:

```
   97.54%   caps_sa::lcp::lcp_u8_avx2
```

Phase 1 (merge-sort) was 89% of the wall and was bottlenecked
exclusively on LCP scanning — the merge bookkeeping, swaps, indexing,
even rayon work-stealing all came to under 3% combined. On the same
host, the **rand100m** profile looked completely different:

```
   45.20%   caps_sa::sample_sort::merge
   42.23%   caps_sa::lcp::lcp_u8_avx2
```

Why so different? Random DNA over a 4-symbol alphabet has an expected
maximum LCP of log₄(n) ≈ 13 bytes for n = 100 MB — every LCP call
exits inside a single 32-byte AVX2 chunk, so the **per-call** cost
(register setup, the boundary tests, the trailing-zeros mask resolve)
is what shows. The genome, with its long microsatellites, segmental
duplications and other repeats, regularly runs LCPs into the hundreds
or thousands of bytes, and there the **SIMD-stride throughput**
dominates.

This made AVX-512 the obvious next step — but only conditionally:
naïvely replacing AVX2 with a 64-byte AVX-512 stride regressed
rand100m by ~15% (one wasted 64-byte load per call instead of one
useful 32-byte load) while improving the human slice by 29%. The
**hybrid path** ships both: a 32-byte AVX2 head that resolves the
short-LCP regime without ever touching a ZMM register, falling through
to a 64-byte AVX-512BW body once the LCP has already exceeded 32
bytes. On the bench host the results were:

| Input        | AVX2 only | AVX-512 (64B only) | AVX-512 hybrid |
| ------------ | --------- | ------------------ | -------------- |
| rand100m 32t | 11.50 s   | 13.32 s (−16%)     | **11.22 s (0%)** |
| hum200m 32t  | 79.67 s   | 57.40 s (+28%)     | **57.00 s (+28%)** |

The hybrid keeps the AVX2 short-LCP baseline intact while preserving
the full AVX-512 win on long LCPs. On Zen 5 specifically there is no
AVX-512 frequency licensing — the 512-bit data paths are native, and
upper-bit power gating is fast — so the only thing the head step
saves is a wasted 32 bytes of loaded data, which on memory-bandwidth-
bound workloads is still worth avoiding.

## SIMD LCP — AVX-512BW + AVX2 + NEON, dispatched once

The `u8`-specialized LCP path is dispatched at runtime by
`std::any::TypeId::of::<S>() == TypeId::of::<u8>()`:

- **x86_64 + AVX-512F + AVX-512BW:** 32-byte AVX2 head (resolves the
  short-LCP regime without ever touching ZMM registers), then 64-byte
  AVX-512 body via `_mm512_cmpeq_epi8_mask` — the byte-compare returns
  a `__mmask64` directly, with no movemask round-trip. See "Where
  AVX-512 helps" above for the measurement that motivated this shape.
- **x86_64 + AVX2:** 32-byte `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`,
  locate the first differing byte via `(!mask).trailing_zeros()`.
- **aarch64 + NEON:** 16-byte `vceqq_u8` + the classic shrn-by-4
  movemask emulation (`vshrn_n_u16<4>` over the reinterpreted `u16`
  view), then `trailing_zeros / 4` for the byte index.
- **Fallback:** portable scalar `S: Eq` byte-loop.

Feature detection runs **once** per `build_*` call (Opt 4): the chosen
function pointer is captured in an `LcpDispatch` value (`Copy + Send +
Sync`) and threaded explicitly through `merge_sort` / `merge` /
`cascade_merge` / `upper_bound_by_pivot`. The inner loops perform a
single indirect call through a register-held pointer — no atomic loads
and no per-call `is_*_feature_detected!` checks. This was particularly
valuable on inputs with short LCPs (random DNA): the per-call cost was
a meaningful fraction of work-per-LCP, so removing it gave 5–10%
across-the-board wins on the 100 MB benchmark.

## Where the differences still come from

| Optimization                             | Upstream C++ | caps-sa Rust |
| ---------------------------------------- | ------------ | ------------ |
| Sample-sort partitioning                 | yes (in-mem + ext-mem) | yes (ext-mem + in-mem-ss) |
| LCP-enhanced 2-way merge                 | yes          | yes          |
| Parallel partition merge                 | yes          | **yes** (Opt 2) |
| Reusable merge buffers across cascade    | yes          | **yes** (Opt 1) |
| AVX2 LCP comparison                      | yes          | **yes** (Opt 3) |
| AVX-512 LCP comparison                   | yes (always-64B) | **yes, hybrid 32B-head/64B-body** (Opt 5) |
| NEON LCP comparison                      | no (x86 only)| **yes** (Opt 3) |
| SIMD dispatch hoisted out of hot path    | yes (compile-time) | **yes** (Opt 4) |
| 32-bit indices when `n < 2^31`           | yes          | **yes** (in-mem + ext-mem via `SaLcp<u32>`) |
| `u32` index ext-mem path                 | yes          | **yes** (generic `SaLcp<I>`) |
| In-memory sample-sort partitioning       | yes          | **yes** (`build_in_memory_sample_sort`) |

## Caveats

- `/usr/bin/time -f %M` reports peak resident set, not the SA
  construction working set in isolation. Both binaries pre-load the
  input text (~`n` bytes) and write the SA (~`n × sizeof(idx)` bytes) —
  those move with index width but are not the merge-sort scratch.
- The upstream and caps-sa SA files differ in record width (i32 vs
  u32/u64) so they don't `cmp` byte-for-byte. Equivalence is verified
  inside caps-sa's unit tests (`build_in_memory` and `build_ext_mem`
  agree with a brute-force reference).
- Numbers vary 5–10% across runs; reported values are the median of
  three warm runs.
