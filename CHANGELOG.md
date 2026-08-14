# caps-sa Changelog

Release notes for the [`caps-sa`](https://crates.io/crates/caps-sa) crate.

## Unreleased

### Added

- Opt-in `PackedPrefixSeedPolicy` for seeding external-memory phase-1 sorts
  with segment-aware fixed-depth `u64` prefix keys. The default is
  `Disabled`; `DenseAlphabetOnly` never allocates a second text-sized buffer,
  while `remap(max_extra_bytes)` explicitly bounds an order-preserving ranked
  copy for gapped byte alphabets.
- `LimitProvider::boundary_rank()` and `BoundaryRank` let a provider declare
  whether segment ends sort below or above real symbols. This semantic
  capability is separate from the `ExtMemOpts` activation policy and defaults
  to `None` for custom providers.

### Changed

- Eligible packed-prefix builds resolve most phase-1 comparisons from one
  segment-bounded key, use exact key-derived LCPs between runs, and invoke the
  full comparator only inside equal-key groups. On the complete 6.56-billion-
  symbol ruSTAR GRCh38 + GENCODE v50 fixture, the seed reduced phase 1 from
  49.038 to 12.204 seconds and the memoized build from 171.205 to 134.618
  seconds (21.4%) at 32 physical cores, with identical output.
- Packed-prefix eligibility is checked before alphabet scanning. Non-`u8`
  symbols, finite contexts, providers without a representable boundary order,
  and over-budget remaps fall back to the existing comparison sort.

### Memory

- The packed seed holds one `(u64, I)` record per selected suffix in each
  active phase-1 task. On the annotated GRCh38 `u64` run this added 366-376
  MiB (about 4.1%) peak RSS. Callers must opt in so this bounded worker scratch
  and any explicitly budgeted ranked-text copy are never imposed silently.

## [v0.7.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.7.0) — 2026-08-13

### Added

- Opt-in geometric LCP memoization for the external-memory phase-4 merge.
  `LcpMemoizationPolicy::Geometric` reuses exact long-LCP intervals through
  bounded, partition-local tables; short comparisons run directly and tables
  activate lazily. The opaque, non-exhaustive
  `GeometricMemoizationConfig` exposes the probe, admission, activation, and
  per-partition capacity thresholds through getters and builder methods.
  Memoization is disabled by default.
- `LcpMemoizationPolicy::geometric()` selects the measured defaults directly;
  `GeometricMemoizationConfig` can also be passed to
  `ExtMemOpts::lcp_memoization` through its `From` conversion.
- Builder and environment controls for selecting memoization, plus unstable
  environment-only profiling diagnostics, including
  `CAPS_SA_GEOMETRIC_MEMO` and the `CAPS_SA_MEMO_*` tunables.
- Cross-platform CI for debug/release tests, documentation, Clippy, formatting,
  and Rust 1.89 MSRV coverage.

### Changed

- Phase 1 of the external-memory path now fuses subarray sorting and partition
  distribution, reducing intermediate work and temporary storage.
- External buckets now decode directly into the position/LCP arrays consumed
  by phase 4, and phase 1 routes slices from those same separate arrays. This
  removes transient array-of-struct conversions without changing the disk
  format.
- When phase 1 already has at least one outer task per worker, each subarray
  uses a task-local ping-pong merge sort. This avoids nested Rayon scheduling,
  prevents stacked per-task scratch during work stealing, and removes
  per-level copy-back passes. Explicit low-subproblem builds retain recursive
  parallelism.
- `SegmentedText` now builds a bounded coarse boundary directory for large
  segment collections. On the 6.56-billion-symbol ruSTAR GRCh38 plus GENCODE
  fixture (1.40 million segments), this reduced the complete memoized build
  from 260.338 s to 172.953 s with 33 MiB additional peak RSS and identical
  output.
- Merge kernels prefetch upcoming text positions on supported targets.
- Debug and test builds validate complete LCP arrays at construction boundaries.
- Finite-`max_context` merges now stop at the same comparison cap as pivot
  selection and the public suffix comparator, rather than reading one extra
  symbol before applying `boundary_order`.

### Compatibility

`ExtMemOpts` is now non-exhaustive; construct it with `default()` or
`from_env()` and its builder methods. This one-time 0.x API break prevents
future option additions from repeatedly breaking struct literals. Version
bumps `0.6.1 → 0.7.0`.

Detailed memoization counters remain available through the environment-backed
profiling path but are no longer a public `ExtMemOpts` field; their internal
shape can evolve without becoming a stable API commitment.

## [v0.6.1](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.6.1) — 2026-06-01

### Added

- Windows support for the pooled external-memory bucket path.
  `PooledExtMemBucket` previously required the Unix `pread`/`pwrite`
  file-extension API (`FileExt::read_at`/`write_at`) and emitted a
  `compile_error!` on non-Unix targets. Positioned reads/writes are now
  routed through `cfg`-gated `pread_one`/`pwrite_one` helpers that use
  `FileExt::seek_read`/`seek_write` on Windows. Every call carries its
  offset explicitly and never relies on the handle's implicit cursor,
  so concurrent positioned I/O from multiple threads to disjoint
  offsets stays correct on both platforms.

### Fixed

- Two clippy lints in `ext_mem.rs` (`unusual_byte_groupings` on a hex
  seed literal, `doc_lazy_continuation` on a `+`-prefixed doc line).

---

## [v0.6.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.6.0) — 2026-05-27

### Added

- Generic-error `try_*` external-memory and in-memory sample-sort
  builders returning `BuildError<E>`, so callers can abort emit
  callbacks with their own error type instead of forcing
  `std::io::Error`.
- `ExtMemOpts::from_env()` and builder setters for external-memory
  tuning, including `CAPS_SA_WORK_DIR` / `CAPS_SA_TMPDIR`.
- Opt-in bounded ordered phase-4 emission via
  `ExtMemOpts::ordered_phase4_emit(true)` or
  `CAPS_SA_ORDERED_PHASE4=1`. The default remains the faster
  collect-then-emit phase-4 path.

### Changed

- Phase 1 writes `(position, lcp)` records directly from the existing
  position and LCP arrays, avoiding a transient `Vec<SaLcp<_>>`.
- Phase 3 appends partition sub-slices while resetting the first LCP
  inside the bucket layer, avoiding one allocation/copy per non-empty
  sub-subarray.
- The merge loop caches current stream-head `LimitProvider::lim_at`
  results, reducing repeated segment-boundary lookups for segmented
  callers.
- `Index::from_usize` docs now state the existing unchecked-cast
  behavior explicitly.

### Compatibility

This release adds a public field to `ExtMemOpts`, which can break
struct-literal construction without `..Default::default()`. Version
bumps `0.5.0 → 0.6.0`.

---

## [v0.5.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.5.0) — 2026-05-25

### Added

- **`build_ext_mem_for_filter` / `build_ext_mem_for_filter_with`** —
  new public entry points that take a `Fn(u64) -> bool` predicate over
  text positions instead of a pre-materialised `Vec<u64>` of kept
  positions. caps-sa walks the predicate **once** at build start to
  materialise a 1-bit-per-position bitmap + a tiny per-block popcount
  prefix sum. Phase 1's subarray fill is then driven by `count_ones` +
  `trailing_zeros` on `u64` words; the predicate is **never invoked
  again** after the initial pass.

  Memory: `(n + 7) / 8` bytes for the bitmap + ~`8 × ⌈n / 65 536⌉`
  bytes for the prefix sum. On the human genome (n ≈ 6.2 × 10⁹) that
  is **~770 MB total**, vs the ~50 GB `Vec<u64>` the `_for_positions`
  path needs for the equivalent STAR-style ACGT-only sampling.

  When to use:
  - **`_for_filter`** when kept positions are described by a cheap
    per-position predicate and the caller has the text in RAM.
    STAR's `text[p] < 4` ACGT-only filter is the motivating case.
  - **`_for_positions`** when the caller already has an explicit
    `Vec<u64>` or the kept set is sparse enough that the bitmap
    representation is wasteful.

  ```rust
  use caps_sa::{build_ext_mem_for_filter, ExtMemOpts};
  let opts = ExtMemOpts::default();
  build_ext_mem_for_filter(text, |p| text[p as usize] < 4, &opts, |sa_pos| {
      // sa_pos is in lex order; the predicate filters at build time.
      write_one(sa_pos)?;
      Ok(())
  })?;
  ```

### Changed

- `build_ext_mem_inner` and `build_in_memory_ss_inner` now
  `drop(source)` immediately after `phase1_sort_sample_spill` returns.
  Phases 2-4 never touch the source. For `_for_positions` callers this
  frees the caller-supplied `Vec<u64>` ~5 minutes earlier on
  genome-scale runs (e.g. the human genome's 47 GB kept-positions
  `Vec`); for the new `_for_filter` callers it frees the bitmap.

### Tests

Four new tests cover the filter API:
- `|_| true` matches the identity build.
- ACGT-filter matches the `_for_positions` Vec path on small
  randomised inputs.
- Cross-`FILTERED_WORDS_PER_BLOCK`-boundary correctness on a 200 K-
  position text — verifies the prefix sum's block layout.
- Sparse predicate (~5 % acceptance) — exercises the skip-loop's
  whole-word-eat path, which fixed an off-by-one in the initial
  bring-up where the `skip >= 64` outer test under-counted the
  per-word popcount.

61/61 tests pass.

### Notes

A truly *O(1)* select-1 structure (darray / Elias-Fano) on top of the
bitmap would shrink the `fill_chunk` skip phase further. With
`chunk_size` ≈ 750 K and only `p` ≈ 8192 `fill_chunk` calls per build,
the current `O(log n_blocks + ≤ one block)` cost is already a rounding
error vs the SA build's other phases. See `src/ext_mem.rs` for the
follow-up note.

### Compatibility

Strictly additive — no public API was removed or changed. Crate
version bumps `0.4.1 → 0.5.0` per the project's precedent of minor
bumps for new public APIs.

---

## [v0.4.1](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.4.1) — 2026-05-25 (earlier)

- `LimitProvider::boundary_order` — caller-controlled tie-break convention.

## [v0.4.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.4.0) — earlier

- `LimitProvider` + `SegmentedText` for multi-segment SAs.

## [v0.3.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.3.0) — earlier

- `Symbol` trait + byte-view SIMD dispatch for any alphabet width.

## [v0.2.1](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.2.1) — earlier

- Phase 4 `chunk_size = 4 × num_threads` — unblocks work-stealing.

## [v0.2.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.2.0) — earlier

- Pool ext-mem buckets onto `num_threads` anonymous tempfiles.

## [v0.1.0](https://github.com/COMBINE-lab/caps-sa/releases/tag/v0.1.0) — earlier

- Initial release.
