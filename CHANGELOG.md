# caps-sa Changelog

Release notes for the [`caps-sa`](https://crates.io/crates/caps-sa) crate.

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
