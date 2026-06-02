---
title: Installation
description: Add the caps-sa library to your crate, or build the caps_sa CLI from source.
---

caps-sa requires **Rust 1.89 or newer** (the AVX-512 LCP fast path uses intrinsics stabilised in 1.89; the crate still compiles on 1.88 without that one fast path).

## The library

Add it from [crates.io](https://crates.io/crates/caps-sa):

```bash
cargo add caps-sa
```

Or pin it in `Cargo.toml` directly:

```toml
[dependencies]
caps-sa = "0.6"
```

The crate has a small dependency surface — `rayon` and `tempfile` — and no C/C++ build step.

## The CLI

The `caps_sa` command is shipped as a Cargo example. Build it from a clone:

```bash
git clone https://github.com/COMBINE-lab/caps-sa
cd caps-sa
cargo build --release --example caps_sa
# binary lands at: target/release/examples/caps_sa
```

Or install it onto your `$PATH` straight from the repository:

```bash
cargo install --git https://github.com/COMBINE-lab/caps-sa --example caps_sa
```

## Platform support

The in-memory and external-memory paths build and run on **Linux, macOS, and Windows** (x86-64 and aarch64). The SIMD LCP kernel auto-selects the best available backend at runtime — AVX-512BW → AVX2 → NEON → scalar — so a binary built on one host stays correct on another.

Next: [Quick start](/caps-sa/getting-started/quick-start/).
