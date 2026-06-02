---
title: The library & the CLI
description: The two artifacts this repository produces — the caps-sa library crate and the caps_sa CLI — and how they relate.
---

The repository produces **two artifacts that share one core**: the `caps-sa` library crate and the `caps_sa` command-line example. Both call into the same sample-sort kernel and SIMD LCP routine; they differ only in how you drive them.

## 1. The library — `caps-sa`

The primary deliverable, published on [crates.io](https://crates.io/crates/caps-sa). It is a `[lib]`-only crate (no required binary) with a small dependency surface — `rayon` for parallelism and `tempfile` for the external-memory buckets.

```toml
# Cargo.toml
[dependencies]
caps-sa = "0.6"
```

It exposes a family of `build_*` entry points returning either an in-memory `Vec<I>` or streaming positions through a caller closure. This is the form other tools embed — for example, it backs the genome-index suffix-array construction in [`rustar-aligner`](https://github.com/scverse/rustar-aligner), which uses the streaming `build_ext_mem_for_positions` path to pack the SA straight to disk without ever holding it in RAM.

See the [Library API](/caps-sa/reference/api/) for the full surface.

## 2. The CLI — `caps_sa`

A minimal command-line tool shipped as a Cargo **example** (`examples/caps_sa.rs`). It reads a file as raw bytes, builds the suffix array, and writes it to disk as a packed little-endian `u64[]` (or `u32[]` for small inputs), printing build timing to stderr.

```bash
cargo build --release --example caps_sa
./target/release/examples/caps_sa input.bin sa.bin --ext-mem --threads 16
```

Its original purpose is **reproducible head-to-head benchmarking** against the upstream C++ CaPS-SA reference (see `bench/`), but it doubles as a standalone suffix-array constructor for any byte file.

See the [CLI parameters](/caps-sa/reference/cli/) for every flag.

## How they relate

```
                  ┌─────────────────────────────┐
                  │   sample-sort kernel +       │
                  │   SIMD LCP merge (src/)      │
                  └──────────────┬──────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                       │
   ┌──────────▼───────────┐              ┌────────────▼───────────┐
   │  caps-sa  (library)  │              │  caps_sa  (CLI example)│
   │  crates.io · embedded│              │  file in → SA file out │
   │  by rustar-aligner   │              │  benchmark harness     │
   └──────────────────────┘              └────────────────────────┘
```

Pick the **library** to embed suffix-array construction in a larger program; pick the **CLI** to build a suffix array from a file on disk or to reproduce the benchmarks.
