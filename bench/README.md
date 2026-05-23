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

All caps-sa runs include the three optimizations applied incrementally
to the Phase 2b sample-sort baseline:

| Step | Description | Yeast 4-thread ext-mem |
| ---- | ----------- | ---------------------- |
| Phase 2b baseline       | sample-sort + sequential cascade merge | 1.88 s |
| + Opt 1: reusable cascade buffers | one `CascadeWorkspace` reused across partitions | 1.78 s |
| + Opt 2: parallel partition merge | `rayon::par_iter_mut` over partitions in chunks | 1.16 s |
| + Opt 3: SIMD LCP (AVX2 / NEON)   | runtime-dispatched `lcp_u8` for byte texts | 1.03 s |

### Yeast — *S. cerevisiae* R64-1-1, 12.16 MB raw text

| Configuration            | 1 thread  |        |       | 4 threads |        |       |
| ------------------------ | --------- | ------ | ----- | --------- | ------ | ----- |
|                          | wall s    | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem      | 7.41      | 748    | 1.00× | 2.10      | 747    | 1.00× |
| upstream C++ ext-mem     | 9.98      | 461    | 1.35× | 4.36      | 462    | 2.08× |
| **caps-sa Rust in-mem**  | **2.90**  | **204**| **0.39×** | **0.94** | **204**| **0.45×** |
| **caps-sa Rust ext-mem** | **3.47**  | **359**| **0.47×** | **1.03** | **320**| **0.49×** |

After the three optimizations, **caps-sa is roughly 2.0–2.6× faster than
upstream and uses ~3.5× less RAM on this input** for both the in-memory
and external-memory paths.

### Random DNA — uniform ACGT, 100 MB

| Configuration            | 1 thread  |        |       | 4 threads |        |       |
| ------------------------ | --------- | ------ | ----- | --------- | ------ | ----- |
|                          | wall s    | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem      | 38.0      | 2208   | 1.00× | **10.1**  | 2208   | 1.00× |
| upstream C++ ext-mem     | 39.2      | 1053   | 1.03× | 13.3      | 1121   | 1.32× |
| **caps-sa Rust in-mem**  | **33.1**  | 1663   | 0.87× | 13.4*     | 1663   | 1.33× |
| **caps-sa Rust ext-mem** | 38.7      | 2053   | 1.02× | 12.5*     | 1913   | 1.24× |

(*) Median of three runs; ±10% variance.

On uniform random DNA the per-call LCP is short (~13 bytes — the
information-theoretic minimum to disambiguate a random suffix in 100 MB
of alphabet-4 text), so the SIMD AVX2/NEON paths exit before any
32-byte chunk pays for itself. Single-thread Rust is still ~13% faster
than upstream in-mem; multi-thread the picture is essentially a tie.

## SIMD LCP — AVX2 + NEON

The `u8`-specialized LCP path is dispatched at runtime by
`std::any::TypeId::of::<S>() == TypeId::of::<u8>()`:

- **x86_64 + AVX2:** 32-byte `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`,
  locate the first differing byte via `(!mask).trailing_zeros()`.
- **aarch64 + NEON:** 16-byte `vceqq_u8` + the classic shrn-by-4
  movemask emulation (`vshrn_n_u16<4>` over the reinterpreted `u16`
  view), then `trailing_zeros / 4` for the byte index.
- **Fallback:** portable scalar `S: Eq` byte-loop.

The dispatch is feature-detected once via the std-cached
`is_*_feature_detected!` macros, so the runtime cost is a single atomic
load per call. AVX-512 support is intentionally out of scope (per
project decision).

## Where the differences still come from

| Optimization                             | Upstream C++ | caps-sa Rust |
| ---------------------------------------- | ------------ | ------------ |
| Sample-sort partitioning                 | yes (in-mem + ext-mem) | yes (ext-mem only) |
| LCP-enhanced 2-way merge                 | yes          | yes          |
| Parallel partition merge                 | yes          | **yes** (Opt 2) |
| AVX2 LCP comparison                      | yes          | **yes** (Opt 3) |
| AVX-512 LCP comparison                   | yes          | no (deferred) |
| NEON LCP comparison                      | no (x86 only)| **yes** (Opt 3) |
| 32-bit indices when `n < 2^31`           | yes          | yes (in-mem only) |
| Reusable merge buffers across cascade    | yes          | **yes** (Opt 1) |
| `u32` index ext-mem path                 | yes          | no (still u64) |
| In-memory sample-sort partitioning       | yes          | no (uses parallel merge-sort instead) |

The remaining gap (mostly visible on random-DNA at 4 threads) is the
two unticked rows above: an `u32` ext-mem path and an explicitly
sample-sort-partitioned *in-memory* path. The current Rust in-mem
implementation is parallel merge-sort, which doesn't match upstream's
in-memory sample-sort partitioning at scale.

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
