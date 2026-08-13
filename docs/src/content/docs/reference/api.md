---
title: Library API
description: The main public surface of the caps-sa crate — build entry points, options, traits, and limit providers.
---

The crate's surface is a family of `build_*` functions over two traits — [`Symbol`](#symbol) (the element type) and [`Index`](#index) (the SA index width) — plus option structs and a [`LimitProvider`](#segmented-texts) hook for segmented texts. This page covers the entry points you will reach for first; see [docs.rs/caps-sa](https://docs.rs/caps-sa) for the exhaustive list.

## In-memory builders

Return the suffix array as a `Vec<I>`.

```rust
pub fn build_in_memory<S, I>(text: &[S]) -> Vec<I>
where S: Symbol, I: Index;

pub fn build_in_memory_with_opts<S, I>(text: &[S], opts: &Opts) -> Vec<I>
where S: Symbol, I: Index;

pub fn build_in_memory_for_positions<S, I>(text: &[S], positions: Vec<I>) -> Vec<I>
where S: Symbol, I: Index;
```

- **`build_in_memory`** — parallel merge-sort; the simplest entry point. Pick `I` (`u32`/`u64`/`usize`) large enough to address `text`.
- **`build_in_memory_with_opts`** — same, with an [`Opts`](#opts) for tuning.
- **`build_in_memory_for_positions`** — sort only the supplied `positions`; the returned vector is those positions in lexicographic order.

## Streaming builders

Emit each SA position (as `u64`, in lex order) through a closure instead of allocating the whole array. Both the external-memory and in-memory sample-sort paths share this shape.

```rust
pub fn build_ext_mem<S, F>(text: &[S], opts: &ExtMemOpts, emit: F) -> io::Result<()>
where S: Symbol, F: FnMut(u64) -> io::Result<()>;

pub fn build_ext_mem_for_positions<S, F>(
    text: &[S], positions: Vec<u64>, opts: &ExtMemOpts, emit: F,
) -> io::Result<()>
where S: Symbol, F: FnMut(u64) -> io::Result<()>;

pub fn build_in_memory_sample_sort<S, F>(text: &[S], opts: &ExtMemOpts, emit: F) -> io::Result<()>
where S: Symbol, F: FnMut(u64) -> io::Result<()>;
```

- **`build_ext_mem`** — disk-spilling sample-sort; peak RAM `~O(text + n/p)`. The default for genome-scale inputs.
- **`build_ext_mem_for_positions`** — as above, but only the supplied positions enter the sort (filter out `N`s, spacers, …).
- **`build_in_memory_sample_sort`** — identical streaming interface with RAM-only buckets.

Each has a `_with` variant taking a [`LimitProvider`](#segmented-texts), and a `try_*` variant whose closure returns your own error type `E` (yielding `Result<(), BuildError<E>>`) instead of `io::Error`. For example: `build_ext_mem_with`, `try_build_ext_mem`, `build_ext_mem_for_filter`.

```rust
use caps_sa::build_ext_mem;

build_ext_mem(text, &Default::default(), |sa_pos| {
    writer.write_all(&sa_pos.to_le_bytes())
})?;
```

## Options

### `Opts`

In-memory tuning.

```rust
pub struct Opts {
    /// Bound on extension comparisons inside the merge.
    /// `usize::MAX` (default) is unbounded.
    pub max_context: usize,
}
```

### `ExtMemOpts`

Sample-sort / external-memory tuning. `Default::default()` is tuned for genome-scale runs.

| Field | Default | Meaning |
| --- | --- | --- |
| `max_context` | `usize::MAX` | Suffix-comparison context cap. Equal prefixes that exhaust it use `boundary_order`; `MAX` gives full ordering. |
| `subproblem_count` | `0` → auto | Number of subarrays `p`. `0` targets ~65,536 selected positions per subarray, clamped to `[rayon workers, 8,192]` and never above `n`. |
| `work_dir` | `std::env::temp_dir()` | Directory for the temporary bucket files. |
| `physical_file_count` | `0` → auto | Physical temp files in the bucket pool. `0` picks one per worker; the `p` logical partition buckets collapse onto this pool. |
| `ordered_phase4_emit` | `false` | Opt into the bounded ordered emitter (lower transient residency on skewed inputs, slightly slower on balanced ones). |
| `lcp_memoization` | `Disabled` | Optional exact long-LCP reuse. `LcpMemoizationPolicy::geometric()` selects the GRCh38-tuned defaults. |

`ExtMemOpts` is non-exhaustive. Construct it with `default()` or `from_env()`
and use its builder methods rather than an external struct literal.

The `CAPS_SA_N_PHYS` environment variable overrides `physical_file_count` for one-off runs.

## Traits

### `Symbol`

The text element type. Implemented for the primitive integers and fixed byte arrays out of the box.

```rust
pub unsafe trait Symbol: Ord + Copy + Send + Sync + 'static {}
// implemented for u8, u16, u32, u64, [u8; N], …
```

caps-sa is generic over the symbol width; the same byte-level SIMD LCP routine serves every `Symbol` via a byte-view dispatch.

### `Index`

The SA index width. Pick the narrowest type that can address your text.

```rust
pub trait Index: Copy + Eq + Ord + Send + Sync + /* … */ {
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;
    fn zero() -> Self;
}
// implemented for u32, u64, usize
```

## Segmented texts

By default the SA is built over one contiguous text ([`PlainText`]). To build a **generalized** SA whose LCP scans stop at sequence boundaries — without inserting sentinel bytes — pass a `SegmentedText` to a `_with` builder.

```rust
use caps_sa::{SegmentedText, build_ext_mem_with};

// boundaries from per-sequence lengths …
let seg = SegmentedText::from_lengths(text.len(), &[100, 50, 220]);
// … or from cumulative ends (e.g. STAR's chr_start table)
let seg = SegmentedText::from_ends(text.len(), vec![100, 150, 370]);

build_ext_mem_with(text, &seg, &opts, |sa_pos| { Ok(()) })?;
```

For large segment collections, `SegmentedText` automatically builds a bounded
coarse directory so each lookup searches only a local range of cumulative
ends. `PlainText` (the default `LimitProvider`) imposes no boundaries and
monomorphizes to the same code as the un-segmented path, so there is no cost
when you don't need segments.

## Errors

The plain builders return `std::io::Result<()>`. The `try_*` builders return `Result<(), BuildError<E>>`, where `E` is the error type your `emit` closure produces — letting you abort a build with your own error rather than forcing it through `io::Error`.
