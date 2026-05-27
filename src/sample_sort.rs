//! In-memory CaPS-SA-style suffix array construction.
//!
//! Phase 1 of the port: a parallel merge-sort with LCP-enhanced two-way merge,
//! exactly the inner sorting kernel of upstream CaPS-SA's `Suffix_Array::merge`
//! and `Suffix_Array::merge_sort` (see `include/Suffix_Array.hpp` and
//! `src/Suffix_Array.cpp`). The sample-sort partitioning around this kernel
//! (`select_pivots` → `distribute_sub_subarrays` → `merge_sub_subarrays`) is
//! Phase 2 / 3 work; the kernel here already produces a correct LCP-annotated
//! suffix array, and `rayon::join` gives parallel divide for free.
//!
//! The LCP-enhanced merge maintains:
//!
//! * `m` — the LCP between the last-output element and the current top of the
//!   *other* stream.
//! * `l_a` = `lcp_a[i_a]` — the LCP between the current top of the
//!   last-output stream and its immediate predecessor (which is the
//!   last-output element).
//!
//! Three cases per step:
//!
//! * `l_a > m`: the next candidate from the last-output stream agrees with the
//!   last-output element past where the other stream diverged — it lies on the
//!   same side of the other stream's top as the last-output element did, so it
//!   wins. No symbol comparison needed.
//! * `l_a < m`: the next candidate diverges from the last-output element
//!   *inside* the prefix shared with the other stream's top. Since the stream
//!   is sorted, the new candidate is larger than its predecessor; at the
//!   divergence offset it therefore exceeds the other stream's top — the
//!   other stream wins. No symbol comparison needed.
//! * `l_a == m`: undetermined; extend the LCP from offset `m` by an actual
//!   symbol scan and compare.

use crate::Index;
use crate::lcp::{LcpDispatch, Symbol};
use crate::limits::{LimitProvider, PlainText};
use rayon::join;

/// Tunable options for SA construction.
#[derive(Clone, Debug)]
pub struct Opts {
    /// Bound on extension comparisons inside the merge. `usize::MAX` (default)
    /// is unbounded — required for full lexicographic correctness when the
    /// caller's text doesn't guarantee comparisons terminate via sentinels
    /// within a known window.
    pub max_context: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            max_context: usize::MAX,
        }
    }
}

/// Build the suffix array of `text` in memory and return it.
///
/// Generic over the symbol type `S` (`Ord + Copy`, e.g. `u8`, `u16`, `u32`)
/// and the index type `I` (`u32`, `u64`, `usize`). Pick the narrowest `I`
/// that can hold `text.len()`.
///
/// Produces a *standard lexicographic* suffix array. The "shorter suffix is
/// smaller when one runs off the end of `text`" tie-break is applied — i.e.
/// the algorithm behaves as if `text` is followed by an implicit symbol
/// smaller than all of `S`.
pub fn build_in_memory<S, I>(text: &[S]) -> Vec<I>
where
    S: Symbol,
    I: Index,
{
    build_in_memory_with_opts(text, &Opts::default())
}

/// Variant of [`build_in_memory`] that accepts tuning options.
pub fn build_in_memory_with_opts<S, I>(text: &[S], opts: &Opts) -> Vec<I>
where
    S: Symbol,
    I: Index,
{
    build_in_memory_with(text, &PlainText::new(text.len()), opts)
}

/// Variant of [`build_in_memory`] that accepts a [`LimitProvider`].
/// With [`PlainText`] this is identical to [`build_in_memory`]; with
/// [`SegmentedText`][crate::limits::SegmentedText] the LCP scans stop
/// at segment boundaries.
pub fn build_in_memory_with<S, I, L>(text: &[S], lp: &L, opts: &Opts) -> Vec<I>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
{
    let n = text.len();
    let positions: Vec<I> = (0..n).map(I::from_usize).collect();
    build_in_memory_for_positions_with(text, positions, lp, opts)
}

/// Sort the caller-supplied `positions` by the lexicographic order of
/// their suffixes in `text`. Returns the positions reordered so that
/// `text[output[i]..]` is the i-th smallest suffix among the input set.
///
/// Equivalent to [`build_in_memory`] for the special case
/// `positions = (0..text.len()).collect()`; the explicit-positions form
/// lets callers skip suffixes they don't want included in the sort —
/// e.g. STAR-style genome indexing where only ACGT-starting positions
/// participate in the SA, avoiding the O(n) work of sorting and then
/// discarding the spacer-starting positions inside bin-padding.
///
/// The suffix at each position is still the slice `text[position..]`;
/// no positions are dropped from the input. To filter, the caller
/// constructs `positions` with only the indices they want.
pub fn build_in_memory_for_positions<S, I>(text: &[S], positions: Vec<I>) -> Vec<I>
where
    S: Symbol,
    I: Index,
{
    build_in_memory_for_positions_with_opts(text, positions, &Opts::default())
}

/// Variant of [`build_in_memory_for_positions`] that accepts tuning options.
pub fn build_in_memory_for_positions_with_opts<S, I>(
    text: &[S],
    positions: Vec<I>,
    opts: &Opts,
) -> Vec<I>
where
    S: Symbol,
    I: Index,
{
    build_in_memory_for_positions_with(text, positions, &PlainText::new(text.len()), opts)
}

/// Variant of [`build_in_memory_for_positions`] that accepts both a
/// [`LimitProvider`] (for segmented LCP truncation) and tuning options.
/// With [`PlainText`] this is identical to
/// [`build_in_memory_for_positions_with_opts`].
pub fn build_in_memory_for_positions_with<S, I, L>(
    text: &[S],
    positions: Vec<I>,
    lp: &L,
    opts: &Opts,
) -> Vec<I>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
{
    let n = positions.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sa: Vec<I> = positions;
    let mut sa_w: Vec<I> = vec![I::zero(); n];
    let mut lcp_arr: Vec<I> = vec![I::zero(); n];
    let mut lcp_w: Vec<I> = vec![I::zero(); n];

    // Choose the LCP implementation once for the whole build; the captured
    // function pointer travels through the recursion in a register, so the
    // inner merge loop pays no atomic load or feature-detection branch.
    let dispatch = LcpDispatch::detect();

    merge_sort(
        text,
        lp,
        &mut sa,
        &mut sa_w,
        &mut lcp_arr,
        &mut lcp_w,
        opts.max_context,
        dispatch,
    );

    sa
}

/// Recursive merge-sort with LCP maintenance.
///
/// Pre: `sa.len() == sa_w.len() == lcp_arr.len() == lcp_w.len()`. The contents
/// of `sa` are the suffix positions to sort (typically an identity
/// permutation at the top level). All other buffers are scratch / output.
///
/// Post: `sa` is sorted in ascending lexicographic order on
/// `text[sa[i]..]`; `lcp_arr[0] = 0` and `lcp_arr[i] = lcp(text[sa[i-1]..],
/// text[sa[i]..])` for `i >= 1`.
///
/// Visible to the rest of the crate so the external-memory path can sort
/// individual subarrays of positions using the same kernel.
#[allow(clippy::too_many_arguments)] // 4 buffers + text + lp + ctx + dispatch
pub(crate) fn merge_sort<S, I, L>(
    text: &[S],
    lp: &L,
    sa: &mut [I],
    sa_w: &mut [I],
    lcp_arr: &mut [I],
    lcp_w: &mut [I],
    max_ctx: usize,
    dispatch: LcpDispatch,
) where
    S: Symbol,
    I: Index,
    L: LimitProvider,
{
    let n = sa.len();
    debug_assert_eq!(sa_w.len(), n);
    debug_assert_eq!(lcp_arr.len(), n);
    debug_assert_eq!(lcp_w.len(), n);

    if n <= 1 {
        if n == 1 {
            lcp_arr[0] = I::zero();
        }
        return;
    }

    let mid = n / 2;
    let (sa_l, sa_r) = sa.split_at_mut(mid);
    let (sa_w_l, sa_w_r) = sa_w.split_at_mut(mid);
    let (lcp_l, lcp_r) = lcp_arr.split_at_mut(mid);
    let (lcp_w_l, lcp_w_r) = lcp_w.split_at_mut(mid);

    join(
        || merge_sort(text, lp, sa_l, sa_w_l, lcp_l, lcp_w_l, max_ctx, dispatch),
        || merge_sort(text, lp, sa_r, sa_w_r, lcp_r, lcp_w_r, max_ctx, dispatch),
    );

    // Merge the two sorted halves (still living in `sa`) into the workspace,
    // then copy the workspace back into the destination so the caller's
    // postcondition holds on `sa` / `lcp_arr`.
    merge(
        text, lp, sa_l, sa_r, lcp_l, lcp_r, sa_w, lcp_w, max_ctx, dispatch,
    );
    sa.copy_from_slice(sa_w);
    lcp_arr.copy_from_slice(lcp_w);
}

/// LCP-enhanced two-way merge of two sorted suffix arrays.
///
/// `x` / `lcp_x` and `y` / `lcp_y` must each be sorted with `lcp_*[0] == 0`
/// and `lcp_*[i] = lcp(arr[i-1], arr[i])` for `i >= 1`. The result is written
/// into `z` / `lcp_z` (length `x.len() + y.len()`).
///
/// Visible to the rest of the crate so the external-memory path can cascade
/// 2-way merges across each partition's sub-subarrays during Phase 4.
#[allow(clippy::too_many_arguments)] // CaPS-SA's merge takes 5 buffers + text + lp + ctx + dispatch
pub(crate) fn merge<S, I, L>(
    text: &[S],
    lp: &L,
    x: &[I],
    y: &[I],
    lcp_x: &[I],
    lcp_y: &[I],
    z: &mut [I],
    lcp_z: &mut [I],
    max_ctx: usize,
    dispatch: LcpDispatch,
) where
    S: Symbol,
    I: Index,
    L: LimitProvider,
{
    let len_x = x.len();
    let len_y = y.len();
    debug_assert_eq!(z.len(), len_x + len_y);
    debug_assert_eq!(lcp_z.len(), len_x + len_y);

    if len_x == 0 {
        z.copy_from_slice(y);
        lcp_z.copy_from_slice(lcp_y);
        return;
    }
    if len_y == 0 {
        z.copy_from_slice(x);
        lcp_z.copy_from_slice(lcp_x);
        return;
    }

    // The "swap-on-output-from-B" trick from upstream CaPS-SA: we always
    // label the stream we last output from as `A`, and the other as `B`. On
    // entry no output has been produced yet, but the convention is consistent
    // because both LCP arrays satisfy `lcp_*[0] == 0` and `m` starts at 0 —
    // the first iteration falls into the `l_a == m` branch and computes the
    // first comparison from scratch.
    let mut arr_a: &[I] = x;
    let mut arr_b: &[I] = y;
    let mut lcp_a: &[I] = lcp_x;
    let mut lcp_b: &[I] = lcp_y;
    let mut len_a = len_x;
    let mut len_b = len_y;
    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    let mut m: usize = 0;
    let mut k: usize = 0;
    let mut lim_a_cache: Option<(usize, usize)> = None;
    let mut lim_b_cache: Option<(usize, usize)> = None;

    while i_a < len_a && i_b < len_b {
        let l_a = lcp_a[i_a].to_usize();

        // (output_a, lcp_for_output, new_m)
        let (output_a, lcp_for_output, new_m) = if l_a > m {
            (true, l_a, m)
        } else if l_a < m {
            (false, m, l_a)
        } else {
            // Tied — extend by an actual symbol scan from offset m.
            let p_a = arr_a[i_a].to_usize();
            let p_b = arr_b[i_b].to_usize();
            // `lim_a` / `lim_b` are the per-suffix logical lengths from
            // the `LimitProvider`. With `PlainText` these fold to
            // `n_text - p` (the same expression the pre-LimitProvider
            // code computed inline); with `SegmentedText` they cap at
            // the next segment boundary so the LCP scan stops there.
            let lim_a = match lim_a_cache {
                Some((idx, lim)) if idx == i_a => lim,
                _ => {
                    let lim = lp.lim_at(p_a);
                    lim_a_cache = Some((i_a, lim));
                    lim
                }
            };
            let lim_b = match lim_b_cache {
                Some((idx, lim)) if idx == i_b => lim,
                _ => {
                    let lim = lp.lim_at(p_b);
                    lim_b_cache = Some((i_b, lim));
                    lim
                }
            };
            // Pass an already-segmentation-aware cap to the SIMD LCP so
            // it doesn't have to scan past the boundary. For PlainText
            // this is equivalent to the LCP function's own `n - p`
            // intersection — no extra work.
            let cap = lim_a.min(lim_b).min(max_ctx);
            let remaining_ctx = cap.saturating_sub(m);
            let ext = dispatch.lcp(text, p_a + m, p_b + m, remaining_ctx);
            let total = m + ext;
            let a_smaller = if total < lim_a && total < lim_b {
                text[p_a + total] < text[p_b + total]
            } else {
                // One or both suffixes ran off the end of their
                // segment (or hit `max_ctx`). Defer to the
                // [`LimitProvider`]'s boundary-ordering convention —
                // for [`PlainText`] / standard `SegmentedText` this
                // is `lim_a.cmp(&lim_b)` (shorter-is-smaller, the
                // generalised-SA convention); custom impls can flip
                // it (e.g. STAR's `spacer-as-largest`).
                lp.boundary_order(p_a, lim_a, p_b, lim_b).is_lt()
            };
            (a_smaller, m, total)
        };

        if output_a {
            z[k] = arr_a[i_a];
            lcp_z[k] = I::from_usize(lcp_for_output);
            i_a += 1;
            lim_a_cache = None;
        } else {
            z[k] = arr_b[i_b];
            lcp_z[k] = I::from_usize(lcp_for_output);
            i_b += 1;
            lim_b_cache = None;
            // Outputting from B: swap labels so the next iteration's
            // "last-output stream" is what was B.
            std::mem::swap(&mut arr_a, &mut arr_b);
            std::mem::swap(&mut lcp_a, &mut lcp_b);
            std::mem::swap(&mut len_a, &mut len_b);
            std::mem::swap(&mut i_a, &mut i_b);
            std::mem::swap(&mut lim_a_cache, &mut lim_b_cache);
        }
        m = new_m;
        k += 1;
    }

    // Drain the surviving stream. The first drained element carries the
    // boundary LCP (`m`) connecting it to the last cross-stream output;
    // subsequent drained elements use the source LCP array unchanged.
    drain(arr_a, lcp_a, i_a, len_a, z, lcp_z, &mut k, m);
    drain(arr_b, lcp_b, i_b, len_b, z, lcp_z, &mut k, m);
}

#[inline]
#[allow(clippy::too_many_arguments)] // drain handles both source streams via labelled args
fn drain<I: Index>(
    arr: &[I],
    lcp_src: &[I],
    mut i: usize,
    len: usize,
    z: &mut [I],
    lcp_z: &mut [I],
    k: &mut usize,
    boundary_m: usize,
) {
    let mut first = true;
    while i < len {
        z[*k] = arr[i];
        lcp_z[*k] = if first {
            I::from_usize(boundary_m)
        } else {
            lcp_src[i]
        };
        first = false;
        i += 1;
        *k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference suffix array via `sort_by` over byte slices.
    fn brute_force_sa(text: &[u8]) -> Vec<u32> {
        let mut sa: Vec<u32> = (0..text.len() as u32).collect();
        sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        sa
    }

    fn assert_matches_brute(text: &[u8]) {
        let got: Vec<u32> = build_in_memory(text);
        let want = brute_force_sa(text);
        assert_eq!(got, want, "mismatch on text {text:?}");
    }

    #[test]
    fn empty_text() {
        let sa: Vec<u32> = build_in_memory::<u8, u32>(&[]);
        assert!(sa.is_empty());
    }

    #[test]
    fn single_symbol() {
        let sa: Vec<u32> = build_in_memory(&[7u8]);
        assert_eq!(sa, vec![0]);
    }

    #[test]
    fn banana() {
        assert_matches_brute(b"banana");
    }

    #[test]
    fn mississippi() {
        assert_matches_brute(b"mississippi");
    }

    #[test]
    fn small_distinct_sentinel() {
        // Alphabet 0..=5 with a unique terminator. Models the
        // sentinel-transformed STAR text on a tiny example.
        let text: Vec<u8> = vec![0, 1, 2, 0, 1, 5, 0, 2, 1, 6];
        let got: Vec<u32> = build_in_memory(&text);
        let want = brute_force_sa(&text);
        assert_eq!(got, want);
    }

    #[test]
    fn random_byte_texts() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        for &n in &[1usize, 2, 3, 7, 33, 200, 1000] {
            let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
            let got: Vec<u32> = build_in_memory(&text);
            let want = brute_force_sa(&text);
            assert_eq!(got, want, "mismatch on random text len={n}");
        }
    }

    #[test]
    fn for_positions_full_set_matches_build_in_memory() {
        // Same output as build_in_memory when positions is the identity.
        let text = b"banana";
        let want: Vec<u32> = build_in_memory(text);
        let positions: Vec<u32> = (0..text.len() as u32).collect();
        let got = build_in_memory_for_positions(text, positions);
        assert_eq!(got, want);
    }

    #[test]
    fn for_positions_subset_matches_brute_force() {
        // Sort only the even positions of "mississippi" by their
        // suffixes; verify against brute force.
        let text = b"mississippi";
        let positions: Vec<u32> = (0..text.len() as u32).step_by(2).collect();
        let mut want = positions.clone();
        want.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        let got = build_in_memory_for_positions(text, positions);
        assert_eq!(got, want);
    }

    #[test]
    fn for_positions_random_subsets() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFEED);
        for &n in &[33usize, 200, 1000] {
            let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
            // Random subset of positions.
            let mut positions: Vec<u32> = (0..n as u32).collect();
            // Drop a random ~30%.
            positions.retain(|_| rng.random_range(0..10) < 7);
            let mut want = positions.clone();
            want.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
            let got = build_in_memory_for_positions(&text, positions);
            assert_eq!(got, want, "subset sort mismatch n={n}");
        }
    }

    #[test]
    fn random_with_unique_terminator() {
        // Distinct large terminator at the end — mimics the transform we'll
        // apply for STAR.
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
        for &n in &[1usize, 50, 500] {
            let mut text: Vec<u8> = (0..n).map(|_| rng.random_range(0..5u8)).collect();
            text.push(250); // unique max
            let got: Vec<u32> = build_in_memory(&text);
            let want = brute_force_sa(&text);
            assert_eq!(got, want);
        }
    }

    // ---- segmented SA tests ----

    use crate::limits::SegmentedText;

    /// Compare two suffixes under the segmented comparator (LCP
    /// truncated at the boundary, "shorter-is-smaller" tie-break).
    fn segmented_cmp(text: &[u8], lp: &SegmentedText, a: usize, b: usize) -> std::cmp::Ordering {
        use crate::limits::LimitProvider;
        let lim_a = lp.lim_at(a);
        let lim_b = lp.lim_at(b);
        let lim = lim_a.min(lim_b);
        for i in 0..lim {
            if text[a + i] != text[b + i] {
                return text[a + i].cmp(&text[b + i]);
            }
        }
        lim_a.cmp(&lim_b)
    }

    /// Assert that `sa` is a valid segmented SA over `text` partitioned
    /// by `lengths`:
    ///
    /// 1. it's a permutation of the positions in `positions`, and
    /// 2. every adjacent pair is in non-decreasing comparator order.
    ///
    /// Comparator-equivalent suffixes can appear in any relative order —
    /// caps-sa's merge isn't a stable sort, so we don't pin a canonical
    /// permutation.
    fn assert_segmented_sa_valid(text: &[u8], lengths: &[usize], positions: &[u32], sa: &[u32]) {
        let lp = SegmentedText::from_lengths(text.len(), lengths);
        let mut expected = positions.to_vec();
        expected.sort();
        let mut got_sorted = sa.to_vec();
        got_sorted.sort();
        assert_eq!(got_sorted, expected, "sa is not a permutation of positions");
        for w in sa.windows(2) {
            let a = w[0] as usize;
            let b = w[1] as usize;
            let ord = segmented_cmp(text, &lp, a, b);
            assert_ne!(
                ord,
                std::cmp::Ordering::Greater,
                "out of order: pos {a} > pos {b} under segmented comparator",
            );
        }
    }

    #[test]
    fn segmented_in_memory_matches_brute_force_small() {
        // 4 segments: "hello" | "world" | "banana" | "mississippi"
        let text: Vec<u8> = b"helloworldbananamississippi".to_vec();
        let lengths = &[5usize, 5, 6, 11];
        let lp = SegmentedText::from_lengths(text.len(), lengths);
        let sa: Vec<u32> = build_in_memory_with(&text, &lp, &Opts::default());
        let all_positions: Vec<u32> = (0..text.len() as u32).collect();
        assert_segmented_sa_valid(&text, lengths, &all_positions, &sa);
    }

    #[test]
    fn segmented_single_segment_equals_unsegmented() {
        // A single segment covering the whole text is the same as the
        // non-segmented SA — confirms the LimitProvider path doesn't
        // perturb the standard order when there's nothing to truncate.
        let text = b"mississippi";
        let lp = SegmentedText::from_lengths(text.len(), &[text.len()]);
        let got_segmented: Vec<u32> = build_in_memory_with(text, &lp, &Opts::default());
        let got_plain: Vec<u32> = build_in_memory(text);
        assert_eq!(got_segmented, got_plain);
    }

    #[test]
    fn segmented_random_validity() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5E6);
        for _ in 0..20 {
            let n_segments = rng.random_range(1..10usize);
            let lengths: Vec<usize> = (0..n_segments)
                .map(|_| rng.random_range(5..50usize))
                .collect();
            let n: usize = lengths.iter().sum();
            // Small alphabet so the LCP-truncation case actually fires.
            let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..3u8)).collect();
            let lp = SegmentedText::from_lengths(n, &lengths);
            let sa: Vec<u32> = build_in_memory_with(&text, &lp, &Opts::default());
            let all_positions: Vec<u32> = (0..n as u32).collect();
            assert_segmented_sa_valid(&text, &lengths, &all_positions, &sa);
        }
    }

    #[test]
    fn segmented_for_positions_subset_validity() {
        // Filter to even positions only, sort with segmentation.
        let text: Vec<u8> = b"helloworldbananamississippi".to_vec();
        let lengths = &[5usize, 5, 6, 11];
        let positions: Vec<u32> = (0..text.len() as u32).step_by(2).collect();
        let lp = SegmentedText::from_lengths(text.len(), lengths);
        let sa =
            build_in_memory_for_positions_with(&text, positions.clone(), &lp, &Opts::default());
        assert_segmented_sa_valid(&text, lengths, &positions, &sa);
    }

    // ---- STAR-convention boundary_order tests ----

    /// A `LimitProvider` wrapping [`SegmentedText`] with STAR's
    /// `spacer-as-largest` boundary semantics: the suffix that hits
    /// its limit first is *larger*, equivalently the longer-`lim`
    /// suffix is smaller, with an ascending-position tie-break when
    /// `lim_a == lim_b`. Used by rustar-aligner's `sa_build` to keep
    /// byte-for-byte STAR compatibility on the segmented arm.
    struct StarConvention {
        inner: SegmentedText,
    }

    impl crate::limits::LimitProvider for StarConvention {
        fn lim_at(&self, p: usize) -> usize {
            self.inner.lim_at(p)
        }
        fn boundary_order(
            &self,
            p_a: usize,
            lim_a: usize,
            p_b: usize,
            lim_b: usize,
        ) -> std::cmp::Ordering {
            lim_b.cmp(&lim_a).then(p_a.cmp(&p_b))
        }
    }

    /// Brute-force SA under STAR's convention (longer-lim is smaller,
    /// position tie-break). Used as the oracle for the differential
    /// test. With a position tie-break the SA is uniquely determined,
    /// so this can be compared with `assert_eq!`.
    fn star_brute_force_sa(text: &[u8], lengths: &[usize]) -> Vec<u32> {
        use crate::limits::LimitProvider;
        let lp = SegmentedText::from_lengths(text.len(), lengths);
        let mut sa: Vec<u32> = (0..text.len() as u32).collect();
        sa.sort_by(|&a, &b| {
            let pa = a as usize;
            let pb = b as usize;
            let lim_a = lp.lim_at(pa);
            let lim_b = lp.lim_at(pb);
            let lim = lim_a.min(lim_b);
            for i in 0..lim {
                if text[pa + i] != text[pb + i] {
                    return text[pa + i].cmp(&text[pb + i]);
                }
            }
            // STAR convention: longer-lim is smaller, then position.
            lim_b.cmp(&lim_a).then(pa.cmp(&pb))
        });
        sa
    }

    #[test]
    fn star_convention_matches_brute_force_small() {
        // 4 segments: "hello" | "world" | "banana" | "mississippi"
        let text: Vec<u8> = b"helloworldbananamississippi".to_vec();
        let lengths = &[5usize, 5, 6, 11];
        let lp = StarConvention {
            inner: SegmentedText::from_lengths(text.len(), lengths),
        };
        let got: Vec<u32> = build_in_memory_with(&text, &lp, &Opts::default());
        let want = star_brute_force_sa(&text, lengths);
        assert_eq!(got, want, "STAR-convention SA mismatch");
    }

    /// Exercises the STAR-specific within-segment longer-is-smaller
    /// case: in "AAAA" with one segment, STAR orders the longest
    /// suffix first (`AAAA < AAA < AA < A`) — opposite of the
    /// standard SA's `A < AA < AAA < AAAA`.
    #[test]
    fn star_convention_within_segment_longer_first() {
        let text = b"AAAA";
        let lp = StarConvention {
            inner: SegmentedText::from_lengths(text.len(), &[text.len()]),
        };
        let got: Vec<u32> = build_in_memory_with(text, &lp, &Opts::default());
        // Position 0 = "AAAA" (lim 4), 1 = "AAA" (lim 3), 2 = "AA"
        // (lim 2), 3 = "A" (lim 1). Longer-lim is smaller, so 0 < 1
        // < 2 < 3.
        assert_eq!(got, vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn star_convention_random_matches_brute_force() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE);
        for _ in 0..20 {
            let n_segments = rng.random_range(1..10usize);
            let lengths: Vec<usize> = (0..n_segments)
                .map(|_| rng.random_range(5..50usize))
                .collect();
            let n: usize = lengths.iter().sum();
            // Small alphabet so the boundary-tie-break case fires.
            let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..3u8)).collect();
            let lp = StarConvention {
                inner: SegmentedText::from_lengths(n, &lengths),
            };
            let got: Vec<u32> = build_in_memory_with(&text, &lp, &Opts::default());
            let want = star_brute_force_sa(&text, &lengths);
            assert_eq!(
                got, want,
                "STAR-convention SA mismatch (lengths={lengths:?}, text={text:?})",
            );
        }
    }
}
