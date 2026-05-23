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

Set `UPSTREAM_LD_PATH` if the upstream binary needs a newer libstdc++ than
the system default.

The harness runs four configurations on the same input:

- upstream C++ in-memory
- upstream C++ external-memory (`--ext-mem --collate-extmem-result`)
- this crate's in-memory path (auto-picks `u32` indices when `n < 2^31`)
- this crate's external-memory path (always `u64` for now)

It reports wall time and peak RSS via `/usr/bin/time`.

## Results

Machine: 64-core x86_64 Linux node, 1 socket, AVX2 enabled.
Builds: upstream `cmake -DCMAKE_BUILD_TYPE=Release`, this crate
`cargo build --release --example caps_sa`.

### Yeast — *S. cerevisiae* R64-1-1, 12.16 MB raw text

| Configuration             | 1 thread |        |       | 4 threads |        |       |
| ------------------------- | -------- | ------ | ----- | --------- | ------ | ----- |
|                           | wall s   | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem       | 7.45     | 747    | 1.00× | 2.09      | 747    | 1.00× |
| upstream C++ ext-mem      | 9.26     | 461    | 1.24× | 3.98      | 462    | 1.90× |
| **caps-sa Rust in-mem**   | **3.25** | **204**| **0.44×** | **1.06** | **204**| **0.51×** |
| caps-sa Rust ext-mem      | 3.78     | 341    | 0.51× | 1.88      | 346    | 0.90× |

`ratio` = wall time relative to upstream C++ in-mem. **The Rust in-memory
path is roughly 2× faster than upstream's in-mem and uses ~3.5× less RAM
on this input** (driven mostly by 32-bit indices + a less wasteful
merge-sort working set).

### Random DNA — uniform ACGT, 100 MB

| Configuration             | 1 thread  |        |       | 4 threads |        |       |
| ------------------------- | --------- | ------ | ----- | --------- | ------ | ----- |
|                           | wall s    | RSS MB | ratio | wall s    | RSS MB | ratio |
| upstream C++ in-mem       | 38.3      | 2208   | 1.00× | **10.1**  | 2208   | 1.00× |
| upstream C++ ext-mem      | 38.8      | 1053   | 1.01× | 12.0      | 1125   | 1.19× |
| **caps-sa Rust in-mem**   | **33.7**  | 1663   | 0.88× | 13.8      | 1663   | 1.37× |
| caps-sa Rust ext-mem      | 39.0      | 1858   | 1.02× | 21.4      | 1206   | 2.12× |

At 100 MB the picture flips multi-threaded — upstream's sample-sort
partitioning parallelizes the inner work better than our Phase 2b cascade
merge does. Single-threaded the Rust port is still slightly faster.

## Where the differences come from

| Optimization                             | Upstream C++ | caps-sa Rust |
| ---------------------------------------- | ------------ | ------------ |
| Sample-sort partitioning                 | yes (in-mem + ext-mem) | yes (ext-mem only) |
| LCP-enhanced 2-way merge                 | yes          | yes          |
| **Parallel partition merge**             | **yes**      | no (sequential cascade) |
| **AVX2 / AVX-512 LCP comparison**        | **yes**      | no (scalar)  |
| 32-bit indices when `n < 2^31`           | yes          | yes (in-mem only) |
| Reusable merge buffers across cascade    | yes          | no (new `Vec` per level) |

These are the four clearly addressable opportunities. The bench harness is
deliberately simple so each one can be evaluated in isolation.

## Caveats

- `/usr/bin/time -f %M` reports peak resident set, not the SA construction
  working set in isolation. Both binaries pre-load the input text (~`n`
  bytes) and write the SA (~`n × sizeof(idx)` bytes) — those move with
  index width but are not the merge-sort scratch.
- The upstream and caps-sa SA files differ in record width (i32 vs
  u32/u64) so they don't `cmp` byte-for-byte. Equivalence is verified
  inside caps-sa's unit tests (`build_in_memory` and `build_ext_mem`
  agree with a brute-force reference).
- Numbers vary 5-10% across runs; reported values are the median of three
  warm runs.
