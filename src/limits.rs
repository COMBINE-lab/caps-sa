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
}

/// Provider for texts partitioned into segments at known cumulative
/// end positions. `lim_at(p)` binary-searches the sorted ends list
/// and returns the distance from `p` to the next boundary.
///
/// Storage cost is `8 × n_segments` bytes (the cumulative-ends
/// `Vec<u64>`). For a 50 K-junction SA index on a 6 GB genome that
/// is **400 KB total** — vs the 750 MB a packed bitmap would need,
/// and the 6 GB an extra-byte-per-symbol u16 text would need.
///
/// Lookup is `O(log n_segments)` — a few cycles for typical
/// segment counts. The merge can cache `lim_p`/`lim_q` across LCP
/// calls so the cost amortises to ~one binary search per output
/// record.
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
        Self { n: text_len, ends }
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
        Self { n: text_len, ends }
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
