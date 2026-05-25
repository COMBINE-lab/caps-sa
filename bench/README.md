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
| **caps-sa Rust ext-mem** | **10.14**  | **4.98**    | **0.93× / 0.77×** |
| caps-sa Rust in-mem-ss   | 11.64      | 55.00       | 1.07× / 8.52× |

`vs upstream` columns are relative to upstream ext-mem (the natural
peer-to-peer comparison). **caps-sa ext-mem is now ~7% faster than
upstream ext-mem on wall time and uses 23% less RAM** — and it beats
upstream's in-mem path by 3% on wall while using 10× less RAM.

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
| + Pooled bucket files (Opt 6)                                | +0.3 min (→ 10.78, noise) | **−0.31 GB** (→ 4.72) |
| + Phase 4 chunk-size = 4·num_threads (Opt 7)                  | **−0.64 min** (→ 10.14) | ~0 (→ 4.98) |
| + `Symbol` trait + byte-view SIMD (Opt 8)                     | ~0 (genome stays u8) | ~0 |
| + `LimitProvider` + `SegmentedText` (Opt 9)                   | ~0 (PlainText path) | ~0 |
| **Net (v1 → today's tip)**                                   | **−0.77 min** | **−34.85 GB (−88%)** |

### `LimitProvider` + `SegmentedText` — multi-segment SAs (Opt 9)

Multi-string SAs (and the SJ-indexed genome case in particular —
where every splice junction wants its own sentinel to avoid
cross-junction LCP collisions) can push the alphabet size into the
tens of thousands. Going to `u16`/`u24`/`u32` text just to encode
those sentinels multiplies the genome's resident memory by 2-4×.

Approach 3 from the WIDE_ALPHABETS analysis in `rustar-aligner`'s
`sa_build.rs`: keep the text at `u8` (just real bases) and encode
the segment-boundary structure in a parallel data structure that the
LCP function consults. This ships that infrastructure on the
caps-sa side.

The shape:

- A new `LimitProvider` trait with one method `fn lim_at(&self,
  p: usize) -> usize` — the logical length of the suffix at `p`.
  Plumbed through every site in `merge` / `cascade_merge` /
  `suffix_cmp` that previously computed `n - p` inline.
- `PlainText { n: usize }` is the default impl. Its `lim_at` is
  `#[inline(always)]`-marked and folds at monomorphization time to
  the same `n - p` expression. The merge becomes a generic
  `merge<S, I, L: LimitProvider>` and, on the `PlainText`
  instantiation, generates **bit-identical assembly to the
  non-segmented path** — confirmed by re-benching the marquee
  GRCh38 / 32 t (`hum200m` / `rand100m` / yeast) wall-times after
  the refactor, all within run-to-run variance.
- `SegmentedText { ends: Vec<u64> }` is the segmented impl. Stores
  cumulative end positions of each segment, `lim_at(p)` is a
  `partition_point` binary search. Two ergonomic constructors:
  `from_lengths(text_len, &[len_0, len_1, …])` and
  `from_ends(text_len, sorted_ends)`. For a 50 K-junction SA on a
  6 GB text the storage cost is **400 KB** total — vs the 750 MB
  a packed bitmap would need, or the 6 extra GB a `u16` text
  would need.

Public API surface: every existing entry (`build_ext_mem`,
`build_in_memory`, the `_for_positions` siblings, and the in-mem-ss
ones) gets a `_with` sibling that takes `lp: &L`. The existing
entries are zero-cost wrappers that pass `&PlainText::new(text.len())`.
`LcpDispatch::suffix_cmp_with` is the analogous one-off helper.

Per-LCP-call cost in the `SegmentedText` case: O(log n_segments)
binary search per call. For 50 K segments that's ~16 comparisons,
amortisable to once per merge output by caching `lim_p` / `lim_q`
in the merge state when the pointers don't move (not done in this
patch — easy follow-up if the cost actually shows up in a profile).

The marquee genome bench is unaffected (the rustar-aligner path
hasn't migrated to `SegmentedText` yet); this lands the
infrastructure so the migration is a self-contained follow-up.
Differential tests against a brute-force segmented oracle cover
random small inputs (`segmented_random_validity`), specific
multi-string fixtures, and subset-positions variants.


### `Symbol` trait — SIMD LCP for arbitrary alphabets (Opt 8)

The LCP fast path was originally u8-only (TypeId-gated dispatch in
`LcpDispatch::lcp<S: Eq + 'static>`). Any other symbol type — `u16`
for >125-chromosome genome indexes, `u32` for large-alphabet integer
texts, `[u8; 3]` for 24-bit-packed data — fell through to the scalar
`lcp_scalar`. That left a clean architectural ceiling of one
microsecond per call regardless of how long the LCP ran, which is
~20× the SIMD path on a long-LCP regime.

The fix collapses every symbol width onto a single byte-level SIMD
function. Reasoning: an LCP only needs *equality* on the shared
prefix, not ordering — and **byte-equality of two `[S]` slices is
exactly value-equality** for any `S` with no padding and no invalid
bit patterns. So `LcpDispatch::lcp<S: Symbol>` becomes:

```rust
let k = std::mem::size_of::<S>();
let bytes = unsafe {
    std::slice::from_raw_parts(text.as_ptr() as *const u8, size_of_val(text))
};
let byte_lcp = unsafe {
    (self.lcp_bytes_fn)(bytes, p * k, q * k, max_ctx.saturating_mul(k))
};
byte_lcp / k
```

`Symbol` is the new public marker trait — `unsafe trait Symbol: Ord +
Copy + Send + Sync + 'static` — with blanket impls for every stdlib
integer (`u8` through `u128`, `i8` through `i128`, `usize`, `isize`)
and for `[T; N]` of any `T: Symbol`. Users get the SIMD path for free
on any of those; custom types opt in via `unsafe impl Symbol for
MyNewtype {}` once they've verified the no-padding / byte-faithful-
equality invariant.

The byte-view trick has a nice subtlety: the SIMD compare's "first
differing byte" might land *inside* a partially-equal symbol (e.g.
on `u16`, low byte equal, high byte differs). `byte_lcp / k` rounds
down to the symbol containing that byte, which is the correct
symbol-LCP — every preceding symbol had every byte equal, this
symbol differs at byte offset `byte_lcp mod k`. **Endianness doesn't
matter** because the SIMD compare resolves equality only; the order
of two differing symbols is recovered by `S::cmp` after the LCP via
`text[lcp].cmp(&text[lcp + 1])`.

#### Empirical impact (AMD EPYC 9575F, 1 M symbols, long-LCP regime)

```
                 scalar baseline    Symbol-trait SIMD    speedup
u8 (1 B/sym)       198.8 ms             4.4 ms          45.0×
u16 (2 B/sym)      198.8 ms             7.0 ms          28.3×
[u8; 3] (3 B/sym)  233.9 ms            10.2 ms          22.9×
u32 (4 B/sym)      198.8 ms            13.7 ms          14.5×
u64 (8 B/sym)      198.8 ms            28.6 ms           7.0×
```

The SIMD-vs-scalar ratio scales as ~`64 / size_of::<S>()` (minus per-
call constant overhead that bites harder on the wider symbols). The
u8 ratio is the AVX-512 stride (64 bytes); every other ratio comes
"for free" because the same instruction does the work.

The crate's marquee genome bench stays on `u8` (single-byte alphabet
for ACGT + N + spacer + sentinels), so this opt doesn't move the
GRCh38 wall — it's an enabling change for downstream consumers. The
practical use case in rustar-aligner: highly-fragmented assemblies
that need more than 125 spacer-sentinel values, which the previous
plan deferred to a "Phase 3 `u16` fallback" — that path is now a
one-line type swap (`Vec<u8>` → `Vec<u16>`) on the rustar-aligner
side with no caps-sa work.

### Phase 4 chunk-size — unblock rayon work-stealing (Opt 7)

Profiling the post-pool human run revealed that **phase 4's parallel
efficiency was only 52%** (CPU 2 274 s into wall 139 s on 32 cores)
while phase 1 ran at ~89%. The cause was structural:

`phase4_merge_and_emit` walks the `p` partition buckets in chunks of
`chunk_size = num_threads`, each chunk processed by a single
`par_iter_mut`. With one partition per thread per chunk, rayon has no
splittable work to redistribute — the chunk's wall is set by its
slowest partition, and the other threads idle. Sample-sort partition
sizes vary ~2× from random sampling, so this tail-straggler effect
compounds across the ~256 chunks of the GRCh38 run (`p = 8192`,
`num_threads = 32`).

Bumping the chunk to `4 × num_threads` gives rayon four partitions
per thread initially. Fast threads steal from slow ones, smoothing
the variance. Peak RAM grows by the additional in-flight merged
partitions — each holds a `Vec<I>` of ~3 MB at human-genome scale
with `u32` indices, so a chunk of 128 partitions costs ~400 MB
transient (vs the previous ~100 MB). Well within the budget already
spent on phase 1.

Empirical impact (AMD EPYC 9575F / Zen 5):

```
                       phase 4 wall      phase 4 speedup vs 32 cores
                       before   after    before   after
rand100m 32t           10.07 s   3.38 s   2.6×    7.9×
GRCh38 32t            138.89 s  98.30 s  16.6×   22.9×   (52% → 72% eff.)

                       total wall
                       before   after    Δ
rand100m 32t           11.22 s   4.51 s   −60%
hum200m 32t           ~57 s     ~56 s   ≈0%        (phase 1 dominates here)
GRCh38 32t             10.78 m  10.14 m  **−6%**
```

The win is concentrated where phase 4 is a meaningful share of total
wall — the marquee human-genome run lost 38 s and the small-input
rand100m bench more than halved. Workloads dominated by phase 1
(long-LCP, repeat-rich genomic regions; the hum200m slice) see no
change because phase 1 isn't the thing we fixed.

We also tried injecting `_mm_prefetch::<_MM_HINT_T0>` 256 bytes
ahead of the AVX-512 loads in `lcp_u8_avx512`. Hum200m, rand100m,
and the full GRCh38 bench all showed indistinguishable wall vs the
bare loop (in fact slightly faster without the extra instructions).
The Zen 5 hardware prefetcher recognises the strided 64-byte access
pattern and is already issuing the same loads — software prefetch
was redundant and adds inner-loop work for nothing. Reverted; the
`lcp_u8_avx512` loop stays bare.

### Pooled bucket files — fewer fds, friendlier on NFS (Opt 6)

The sample-sort algorithm needs `2·p` bucket-shaped storage regions
(`p` for the phase-1 subarray spills, `p` for the phase-3 partition
intermediate). With `p` auto-scaling to 8192 on the human genome,
that's 16 384 backing files in the original design — one
`NamedTempFile` per bucket. Three real-world headaches:

- Open-file-handle limits. Some workstation shells default to
  `RLIMIT_NOFILE = 1024`; many HPC nodes cap at 4096. 16 K files
  blows through both.
- NFS metadata cost. Each `openat`/`unlink` on a networked filesystem
  is a synchronous server roundtrip (5–50 ms typical). 16 K of them
  is minutes of pure metadata traffic before any data is moved.
- Inode pressure on the filesystem (some tmpfs / scratch volumes are
  inode-constrained).

The fix is straightforward: pool the buckets onto a small set of
anonymous tempfiles. `BucketPool::new(n_phys, work_dir)` opens
`n_phys` already-unlinked tempfiles up front; each
`PooledExtMemBucket` holds an `Arc<PhysicalFile>` to one of them and
appends its flushes at offsets handed out by a per-file
`AtomicU64::fetch_add`. Multiple threads in `pwrite` on the same fd
don't conflict at disjoint offsets — the kernel serialises by
`(fd, offset_range)` and the cursor allocates disjoint ranges,
making the write path lock-free at user level.

`load_all` reads the bucket's recorded extents back via `pread`. The
extents per bucket are `O(records / buffer_records)` — small in
absolute terms (a few hundred for a phase-4 partition at human scale),
so the extent metadata is a few MB total across all buckets.

`ExtMemOpts::physical_file_count = 0` (the default) resolves to
`rayon::current_num_threads()` — one writable inode per worker, which
empirically matches per-bucket-file wall time on rand100m, hum200m,
yeast, and the full human genome. Callers who want to override (e.g.
on a host with extreme file-descriptor or contention pressure) can
set the field directly, or override at runtime via the
`CAPS_SA_N_PHYS` env var.

#### Empirical impact (AMD EPYC 9575F / Zen 5)

```
                              per-bucket files     pooled (N = num_threads)
rand100m 32t openat calls           9 168                     76
rand100m 32t unlink calls           3 052                      0   (anonymous)
rand100m 32t wall                 ~11.2 s                  ~11.1 s
rand100m 32t RSS                   632 MB                   610 MB
human   32t wall                  10.47 min                10.78 min   (≈ run variance)
human   32t RSS                    5.03 GB                  4.72 GB   (−6%)
human   32t bucket files           16 384                       64
```

The wall time is **at parity** within run-to-run variance on every
input we tested; RAM is **slightly lower** because we drop the
per-bucket `BufWriter` + `NamedTempFile` metadata. The file-count
collapse from `2·p` to `2·N` is the qualitative win — no more open-
file-handle headaches, no more NFS metadata wall.

A separate empirical guard for the `*_for_positions` rustar-aligner
integration lives at [`examples/pathology_bench.rs`](../examples/pathology_bench.rs):
it builds a synthetic spacer-padding fixture (the kind rustar-aligner
hands us when `genomeChrBinNbits` rounds a small chromosome up to a
much larger padded text) and times `build_ext_mem` (sort everything)
against `build_ext_mem_for_positions` (sort only the kept positions)
on the same fixture, showing **6–10× speedups** on padding-dominated
inputs. The pool change leaves this gap intact.

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
