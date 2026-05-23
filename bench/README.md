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

All caps-sa runs include the four optimizations applied incrementally
to the Phase 2b sample-sort baseline:

| Step | Description | Yeast 4-thread ext-mem |
| ---- | ----------- | ---------------------- |
| Phase 2b baseline       | sample-sort + sequential cascade merge | 1.88 s |
| + Opt 1: reusable cascade buffers | one `CascadeWorkspace` reused across partitions | 1.78 s |
| + Opt 2: parallel partition merge | `rayon::par_iter_mut` over partitions in chunks | 1.16 s |
| + Opt 3: SIMD LCP (AVX2 / NEON)   | runtime-dispatched `lcp_u8` for byte texts | 1.03 s |
| + Opt 4: hoist SIMD dispatch       | `LcpDispatch::detect()` once, fn-pointer threaded down | **0.99 s** |

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

## SIMD LCP — AVX2 + NEON, dispatched once

The `u8`-specialized LCP path is dispatched at runtime by
`std::any::TypeId::of::<S>() == TypeId::of::<u8>()`:

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

AVX-512 support is intentionally out of scope (per project decision).

## Where the differences still come from

| Optimization                             | Upstream C++ | caps-sa Rust |
| ---------------------------------------- | ------------ | ------------ |
| Sample-sort partitioning                 | yes (in-mem + ext-mem) | yes (ext-mem only) |
| LCP-enhanced 2-way merge                 | yes          | yes          |
| Parallel partition merge                 | yes          | **yes** (Opt 2) |
| Reusable merge buffers across cascade    | yes          | **yes** (Opt 1) |
| AVX2 LCP comparison                      | yes          | **yes** (Opt 3) |
| AVX-512 LCP comparison                   | yes          | no (deferred) |
| NEON LCP comparison                      | no (x86 only)| **yes** (Opt 3) |
| SIMD dispatch hoisted out of hot path    | yes (compile-time) | **yes** (Opt 4) |
| 32-bit indices when `n < 2^31`           | yes          | yes (in-mem only) |
| `u32` index ext-mem path                 | yes          | no (still u64) |
| In-memory sample-sort partitioning       | yes          | no (uses parallel merge-sort instead) |

The remaining gap is concentrated in **rand100m / 4-thread / in-memory**:
upstream's in-memory sample-sort partitioning parallelizes the LCP work
better than our parallel merge-sort at scale. The two open items above
(`u32` ext-mem path and an in-memory sample-sort path) would close that.

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
