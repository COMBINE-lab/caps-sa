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
