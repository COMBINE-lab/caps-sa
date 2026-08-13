---
title: Quick start
description: Build your first suffix array with the caps-sa library and the caps_sa CLI.
---

## Library: in-memory

The simplest entry point. Pass a byte slice, get back its lexicographic suffix array. The index type is generic — pick `u32`, `u64`, or `usize`:

```rust
use caps_sa::build_in_memory;

let text = b"banana";
let sa: Vec<u32> = build_in_memory(text);

assert_eq!(sa, vec![5, 3, 1, 0, 4, 2]);
// suffixes in lex order: "a", "ana", "anana", "banana", "na", "nana"
```

The result is a *standard* lexicographic suffix array with the "shorter suffix sorts first" tie-break (as if `text` is followed by a symbol smaller than every other).

## Library: external-memory, streaming

For large inputs, stream the SA out of the disk-spilling sample-sort so it is never fully materialised in RAM. You receive each position, in lex order, through a closure:

```rust
use caps_sa::{ExtMemOpts, build_ext_mem};

let text = std::fs::read("genome.bin")?;
let opts = ExtMemOpts::default();

let mut out = std::io::BufWriter::new(std::fs::File::create("sa.bin")?);
build_ext_mem(&text, &opts, |sa_pos| {
    use std::io::Write;
    out.write_all(&sa_pos.to_le_bytes())   // u64, little-endian
})?;
```

### Optional: seed phase 1 from packed prefixes

Dense byte alphabets can opt into segment-aware fixed-depth keys for the
external-memory phase-1 sorts:

```rust
use caps_sa::{ExtMemOpts, PackedPrefixSeedPolicy};

let opts = ExtMemOpts::default()
    .packed_prefix_seed(PackedPrefixSeedPolicy::DenseAlphabetOnly);
```

The mode is disabled by default because it adds one key record per selected
suffix in each active phase-1 task. It requires unbounded comparisons and a
`LimitProvider` with a representable `boundary_rank()`; otherwise caps-sa
falls back automatically. `DenseAlphabetOnly` never creates a text-sized copy.
See the [library API](/caps-sa/reference/api/#packed-prefix-phase-1-seed) for
gapped alphabets and custom boundary conventions.

### Optional: reuse repeated long contexts

For inputs with many long repeated contexts, opt into geometric LCP
memoization for the final partition merges:

```rust
use caps_sa::{ExtMemOpts, LcpMemoizationPolicy, PackedPrefixSeedPolicy};

let opts = ExtMemOpts::default()
    .packed_prefix_seed(PackedPrefixSeedPolicy::DenseAlphabetOnly)
    .lcp_memoization(LcpMemoizationPolicy::geometric());
```

It is intentionally disabled by default. Ordinary short comparisons run
directly; each partition activates its bounded local table only after learning
enough exact long-LCP intervals. The complete ruSTAR-shaped GRCh38 + GENCODE
v50 workload improved by 8.4% in the isolated A/B, while smaller or less
repetitive inputs can be neutral or slightly slower. See
[Geometric LCP memoization](/caps-sa/concepts/geometric-memoization/) before
enabling it broadly.

The policies compose: packed prefixes reduce phase 1, while geometric
memoization reduces phase 4. On complete ruSTAR-shaped GRCh38 + GENCODE v50,
the memoized build improved from 171.205 to 134.618 seconds when the packed
seed was also enabled, with identical output.

## Library: sort only a subset

When many positions should be excluded from the sort (e.g. `N`s or inter-sequence spacers in a genome), hand only the positions you want sorted to a `*_for_positions` entry point. The others never enter the sort:

```rust
use caps_sa::build_ext_mem_for_positions;

let positions: Vec<u64> =
    (0..text.len() as u64).filter(|&p| text[p as usize] < 4).collect();

build_ext_mem_for_positions(&text, positions, &opts, |sa_pos| {
    // sa_pos ranges only over the positions you passed in
    Ok(())
})?;
```

## CLI

Build the example binary, then construct a suffix array from any byte file. The SA is written as packed little-endian integers; build timing prints to stderr:

```bash
cargo build --release --example caps_sa

# in-memory (default) — good for small/medium inputs
./target/release/examples/caps_sa input.bin sa.bin

# external-memory, 16 threads — for genome-scale inputs
./target/release/examples/caps_sa input.bin sa.bin --ext-mem --threads 16
```

See the [CLI parameters](/caps-sa/reference/cli/) and the [Library API](/caps-sa/reference/api/) for the complete surface.
