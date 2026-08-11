//! Per-suffix length providers for segmented suffix-array construction.
//!
//! In the standard SA construction the "natural length" of the suffix
//! starting at position `p` is `text.len() - p`. For *segmented* texts
//! (multi-string SAs, splice-junction indexes, etc.) we want LCP
//! comparisons to stop at the next segment boundary instead — the
//! suffix logically ends there, and the merge resolves cross-segment
//! ordering by "shorter-suffix-is-smaller" (the standard generalised-SA
//! convention).
//!
//! The [`LimitProvider`] trait abstracts the per-suffix length lookup
//! and is plumbed through every site in `merge` / `cascade_merge` /
//! `suffix_cmp` that previously computed `n - p` inline.
//! [`PlainText`] is the zero-cost default — its `lim_at` is
//! `#[inline(always)]` and folds to the same `n - p` expression the
//! current code emits, so the non-segmented path generates **bit-
//! identical assembly** to today's after monomorphization.
//! [`SegmentedText`] holds a sorted cumulative-ends `Vec<u64>` and
//! does a `partition_point` per lookup; the merge can cache the
//! result across LCP calls so the cost amortises to ~one binary
//! search per output record.
//!
//! See `bench/README.md` "Approach 3 — segmented LCP" for the design
//! rationale and the comparison against the `[u8; 3]` (24-bit-text)
//! alternative.

/// Per-suffix length provider. The merge and cascade-merge code use
/// `lp.lim_at(p)` instead of `text.len() - p`; the LCP function itself
/// is unchanged (the merge passes the appropriately-capped
/// `max_ctx` to the existing SIMD path).
///
/// Implementations must be `Sync` so the rayon-parallel sort can
/// share one provider across worker threads.
pub trait LimitProvider: Sync {
    /// Logical length of the suffix starting at position `p` in
    /// symbols — i.e. the number of comparable symbols before the
    /// next segment boundary or end-of-text. Must be at most
    /// `text.len() - p`.
    fn lim_at(&self, p: usize) -> usize;

    /// Order to resolve when one or both suffixes hit their boundary
    /// before any byte of their shared prefix differs. The default
    /// is `lim_a.cmp(&lim_b)` — "shorter-suffix-is-smaller", the
    /// standard generalised-SA / multi-string-SA convention, what a
    /// `Vec<&str>` sort with `&str` ordering produces.
    ///
    /// Custom impls can override for different boundary conventions.
    /// The motivating example is STAR's `spacer-as-largest` ordering:
    /// the suffix that hits a spacer first is *larger*, equivalently
    /// the longer-`lim` one is smaller, with an ascending-position
    /// tie-break when both `lim`s coincide:
    ///
    /// ```ignore
    /// fn boundary_order(&self, p_a: usize, lim_a: usize,
    ///                   p_b: usize, lim_b: usize) -> Ordering {
    ///     lim_b.cmp(&lim_a).then(p_a.cmp(&p_b))
    /// }
    /// ```
    ///
    /// `p_a` / `p_b` are the suffix start positions in the same
    /// coordinate space the merge sees (the spacer-free text's
    /// coordinates when invoked through `*_with` entries on a
    /// rustar-aligner-style spacer-free text). The default impl
    /// ignores them; impls that want a position tie-break use them.
    #[inline]
    fn boundary_order(
        &self,
        p_a: usize,
        lim_a: usize,
        p_b: usize,
        lim_b: usize,
    ) -> std::cmp::Ordering {
        let _ = (p_a, p_b);
        lim_a.cmp(&lim_b)
    }

    /// `Some(n)` iff this provider describes an unsegmented text of `n`
    /// symbols under the *standard* comparator: `lim_at(p) == n - p` for
    /// every `p`, and `boundary_order` left at its shorter-is-smaller
    /// default.
    ///
    /// Returning `Some` lets the crate substitute a specialised suffix-array
    /// algorithm that assumes plain lexicographic order. It is therefore a
    /// promise about the *comparator*, not merely about the lengths.
    ///
    /// The default is `None`, which keeps every existing and third-party
    /// implementation on the general merge kernel at today's semantics. In
    /// particular, an implementation that delegates `lim_at` to [`PlainText`]
    /// but overrides [`boundary_order`][LimitProvider::boundary_order] to get
    /// a different convention (STAR's spacer-as-largest ordering is the
    /// motivating example) inherits `None` and is safe without doing
    /// anything.
    ///
    /// Override this only if you have *not* overridden `boundary_order`.
    #[inline]
    fn plain_lex_len(&self) -> Option<usize> {
        None
    }
}

/// Default provider for non-segmented texts: `lim_at(p) = n - p`.
/// Stored as a single `usize`; the `#[inline(always)]` `lim_at`
/// folds at monomorphization time into the same `n - p` the merge
/// used before this abstraction existed, so non-segmented callers
/// pay zero overhead.
#[derive(Copy, Clone, Debug)]
pub struct PlainText {
    /// Total text length in symbols.
    pub n: usize,
}

impl PlainText {
    /// New `PlainText` for a text of `n` symbols.
    #[inline]
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl LimitProvider for PlainText {
    #[inline(always)]
    fn lim_at(&self, p: usize) -> usize {
        self.n - p
    }

    #[inline]
    fn plain_lex_len(&self) -> Option<usize> {
        Some(self.n)
    }
}

/// Provider for texts partitioned into segments at known cumulative
/// end positions. `lim_at(p)` returns the distance from `p` to the next
/// boundary. Large collections automatically add a compact coarse directory,
/// reducing the binary search to the boundaries inside one position block.
///
/// Base storage is `8 × n_segments` bytes for the cumulative ends. For at
/// least 256 segments, the optional `u32` directory adds at most 8 MiB and at
/// most another `8 × n_segments` bytes. This remains much smaller than a
/// per-symbol boundary bitmap or a widened text alphabet at genome scale.
///
/// Lookup is `O(log n_segments)` in the fallback and `O(log b)` with the
/// directory, where `b` is the number of boundaries in one coarse block. The
/// merge also caches `lim_p`/`lim_q` while a suffix remains at a run front.
///
/// Two constructors:
/// - [`from_lengths`][Self::from_lengths] takes per-segment lengths
///   and builds the cumulative-ends list internally. Most ergonomic
///   when the caller has `[chr_len_0, chr_len_1, …]` already.
/// - [`from_ends`][Self::from_ends] takes the sorted cumulative
///   ends directly. Useful when the caller already has them — e.g.
///   STAR's `chr_start[]` table.
///
/// Both constructors require the segments to cover the whole text
/// (`sum(lengths) == text_len`, or `ends.last() == Some(text_len)`).
#[derive(Clone, Debug)]
pub struct SegmentedText {
    n: usize,
    /// Sorted, strictly-increasing cumulative end positions. After
    /// segment 0 of length 100 ends at index 100, `ends[0] = 100`.
    /// After segment 1 of length 50 (positions 100..150),
    /// `ends[1] = 150`. The last entry equals the total text length.
    ends: Vec<u64>,
    /// Coarse position-to-boundary index for large segment collections.
    directory: Option<BoundaryDirectory>,
}

#[derive(Clone, Debug)]
struct BoundaryDirectory {
    block_shift: u32,
    /// Number of segment ends at or before each power-of-two block start.
    first_after_block_start: Vec<u32>,
}

impl BoundaryDirectory {
    /// Small segment collections already fit comfortably in cache and do not
    /// repay an extra directory lookup. Large collections get at most two
    /// million coarse blocks and roughly two blocks per end when text length
    /// permits it. The directory therefore occupies at most 8 MiB and at most
    /// eight additional bytes per segment.
    const MIN_ENDS: usize = 256;
    const MAX_BLOCKS: usize = 2_000_000;

    fn build(n: usize, ends: &[u64]) -> Option<Self> {
        if ends.len() < Self::MIN_ENDS || ends.len() > u32::MAX as usize || n == 0 {
            return None;
        }

        let target_blocks = ends.len().saturating_mul(2).clamp(1, Self::MAX_BLOCKS);
        let min_block_size = n.div_ceil(target_blocks);
        let block_size = min_block_size
            .checked_next_power_of_two()
            .unwrap_or(1usize << (usize::BITS - 1));
        let block_shift = block_size.trailing_zeros();
        let n_blocks = n.div_ceil(block_size);
        let mut first_after_block_start = Vec::with_capacity(n_blocks + 1);
        let mut end_index = 0usize;
        for block in 0..=n_blocks {
            let block_start = block.saturating_mul(block_size).min(n) as u64;
            while end_index < ends.len() && ends[end_index] <= block_start {
                end_index += 1;
            }
            first_after_block_start.push(end_index as u32);
        }
        Some(Self {
            block_shift,
            first_after_block_start,
        })
    }
}

impl SegmentedText {
    /// Build from per-segment lengths. The sum must equal `text_len`.
    pub fn from_lengths(text_len: usize, lengths: &[usize]) -> Self {
        let mut ends = Vec::with_capacity(lengths.len());
        let mut cum: u64 = 0;
        for &len in lengths {
            cum += len as u64;
            ends.push(cum);
        }
        assert_eq!(
            cum as usize, text_len,
            "SegmentedText::from_lengths: per-segment lengths sum to {cum} but text_len is {text_len}",
        );
        let directory = BoundaryDirectory::build(text_len, &ends);
        Self {
            n: text_len,
            ends,
            directory,
        }
    }

    /// Build from sorted, strictly-increasing cumulative end positions.
    /// `ends.last()` must equal `text_len`.
    pub fn from_ends(text_len: usize, ends: Vec<u64>) -> Self {
        assert!(
            ends.windows(2).all(|w| w[0] < w[1]),
            "SegmentedText::from_ends: ends must be strictly increasing",
        );
        match ends.last() {
            Some(&last) => assert_eq!(
                last as usize, text_len,
                "SegmentedText::from_ends: last end ({last}) != text_len ({text_len})",
            ),
            None => assert_eq!(
                text_len, 0,
                "SegmentedText::from_ends: empty ends but text_len ({text_len}) != 0",
            ),
        }
        let directory = BoundaryDirectory::build(text_len, &ends);
        Self {
            n: text_len,
            ends,
            directory,
        }
    }

    /// Total text length in symbols.
    #[inline]
    pub fn text_len(&self) -> usize {
        self.n
    }

    /// Number of segments.
    #[inline]
    pub fn n_segments(&self) -> usize {
        self.ends.len()
    }

    /// Cumulative end positions, sorted, strictly increasing.
    /// `ends()[i]` is the position one past the last symbol of
    /// segment `i`.
    #[inline]
    pub fn ends(&self) -> &[u64] {
        &self.ends
    }
}

impl LimitProvider for SegmentedText {
    #[inline]
    fn lim_at(&self, p: usize) -> usize {
        if let Some(directory) = &self.directory
            && p < self.n
        {
            let block = p >> directory.block_shift;
            let lo = directory.first_after_block_start[block] as usize;
            let mut hi = directory.first_after_block_start[block + 1] as usize;
            // If the block has no boundary, include the first boundary from a
            // later block so the local search still contains its answer.
            hi = hi.max(lo + 1).min(self.ends.len());
            let i = lo + self.ends[lo..hi].partition_point(|&b| b <= p as u64);
            return self.ends[i] as usize - p;
        }
        // First boundary strictly greater than p.
        let i = self.ends.partition_point(|&b| b <= p as u64);
        if i < self.ends.len() {
            self.ends[i] as usize - p
        } else {
            // p past the last boundary: just text-end.
            self.n - p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_lim_at_matches_n_minus_p() {
        let lp = PlainText::new(100);
        assert_eq!(lp.lim_at(0), 100);
        assert_eq!(lp.lim_at(50), 50);
        assert_eq!(lp.lim_at(99), 1);
        assert_eq!(lp.lim_at(100), 0);
    }

    #[test]
    fn segmented_from_lengths_cumulates_ends() {
        let lp = SegmentedText::from_lengths(15, &[3, 5, 7]);
        assert_eq!(lp.n_segments(), 3);
        assert_eq!(lp.ends(), &[3, 8, 15]);
    }

    #[test]
    #[should_panic(expected = "sum to")]
    fn segmented_from_lengths_rejects_undercoverage() {
        let _ = SegmentedText::from_lengths(20, &[3, 5, 7]);
    }

    #[test]
    fn segmented_lim_at_caps_at_next_boundary() {
        let lp = SegmentedText::from_lengths(15, &[3, 5, 7]);
        // Segment 0 = [0, 3): boundary at 3.
        assert_eq!(lp.lim_at(0), 3);
        assert_eq!(lp.lim_at(1), 2);
        assert_eq!(lp.lim_at(2), 1);
        // Segment 1 = [3, 8): boundary at 8.
        assert_eq!(lp.lim_at(3), 5);
        assert_eq!(lp.lim_at(5), 3);
        assert_eq!(lp.lim_at(7), 1);
        // Segment 2 = [8, 15): boundary at 15.
        assert_eq!(lp.lim_at(8), 7);
        assert_eq!(lp.lim_at(14), 1);
        assert_eq!(lp.lim_at(15), 0);
    }

    #[test]
    fn segmented_handles_single_segment_text() {
        let lp = SegmentedText::from_lengths(10, &[10]);
        assert_eq!(lp.lim_at(0), 10);
        assert_eq!(lp.lim_at(5), 5);
        assert_eq!(lp.lim_at(10), 0);
    }

    #[test]
    fn segmented_directory_matches_binary_search() {
        let lengths: Vec<usize> = (0..2_000).map(|i| 1 + i % 97).collect();
        let n = lengths.iter().sum();
        let indexed = SegmentedText::from_lengths(n, &lengths);
        assert!(indexed.directory.is_some());

        for p in 0..=n {
            let i = indexed.ends.partition_point(|&b| b <= p as u64);
            let want = if i < indexed.ends.len() {
                indexed.ends[i] as usize - p
            } else {
                n - p
            };
            assert_eq!(indexed.lim_at(p), want, "p={p}");
        }
    }

    #[test]
    fn segmented_handles_empty_text() {
        let lp = SegmentedText::from_lengths(0, &[]);
        assert_eq!(lp.n_segments(), 0);
        // No suffixes to query, but the constructor accepts it.
    }

    #[test]
    fn segmented_from_ends_matches_from_lengths() {
        let a = SegmentedText::from_lengths(15, &[3, 5, 7]);
        let b = SegmentedText::from_ends(15, vec![3, 8, 15]);
        assert_eq!(a.ends(), b.ends());
        for p in 0..=15 {
            assert_eq!(a.lim_at(p), b.lim_at(p), "p={p}");
        }
    }
}
