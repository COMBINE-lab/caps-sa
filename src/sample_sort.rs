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
use crate::lcp::lcp;
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
    S: Ord + Copy + Sync,
    I: Index,
{
    build_in_memory_with_opts(text, &Opts::default())
}

/// Variant of [`build_in_memory`] that accepts tuning options.
pub fn build_in_memory_with_opts<S, I>(text: &[S], opts: &Opts) -> Vec<I>
where
    S: Ord + Copy + Sync,
    I: Index,
{
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }

    // Identity permutation; merge-sort will reorder in place (via a scratch
    // buffer of equal size).
    let mut sa: Vec<I> = (0..n).map(I::from_usize).collect();
    let mut sa_w: Vec<I> = vec![I::zero(); n];
    let mut lcp_arr: Vec<I> = vec![I::zero(); n];
    let mut lcp_w: Vec<I> = vec![I::zero(); n];

    merge_sort(
        text,
        &mut sa,
        &mut sa_w,
        &mut lcp_arr,
        &mut lcp_w,
        opts.max_context,
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
pub(crate) fn merge_sort<S, I>(
    text: &[S],
    sa: &mut [I],
    sa_w: &mut [I],
    lcp_arr: &mut [I],
    lcp_w: &mut [I],
    max_ctx: usize,
) where
    S: Ord + Copy + Sync,
    I: Index,
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
        || merge_sort(text, sa_l, sa_w_l, lcp_l, lcp_w_l, max_ctx),
        || merge_sort(text, sa_r, sa_w_r, lcp_r, lcp_w_r, max_ctx),
    );

    // Merge the two sorted halves (still living in `sa`) into the workspace,
    // then copy the workspace back into the destination so the caller's
    // postcondition holds on `sa` / `lcp_arr`.
    merge(text, sa_l, sa_r, lcp_l, lcp_r, sa_w, lcp_w, max_ctx);
    sa.copy_from_slice(sa_w);
    lcp_arr.copy_from_slice(lcp_w);
}

/// LCP-enhanced two-way merge of two sorted suffix arrays.
///
/// `x` / `lcp_x` and `y` / `lcp_y` must each be sorted with `lcp_*[0] == 0`
/// and `lcp_*[i] = lcp(arr[i-1], arr[i])` for `i >= 1`. The result is written
/// into `z` / `lcp_z` (length `x.len() + y.len()`).
#[allow(clippy::too_many_arguments)] // CaPS-SA's merge takes 5 buffers + text + ctx
fn merge<S, I>(
    text: &[S],
    x: &[I],
    y: &[I],
    lcp_x: &[I],
    lcp_y: &[I],
    z: &mut [I],
    lcp_z: &mut [I],
    max_ctx: usize,
) where
    S: Ord + Copy,
    I: Index,
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
    let n_text = text.len();

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
            let lim_a = n_text - p_a;
            let lim_b = n_text - p_b;
            let remaining_ctx = max_ctx.saturating_sub(m);
            let ext = lcp(text, p_a + m, p_b + m, remaining_ctx);
            let total = m + ext;
            let a_smaller = if total < lim_a && total < lim_b {
                text[p_a + total] < text[p_b + total]
            } else {
                // One or both suffixes ran off the end (or hit `max_ctx`).
                // Shorter remaining suffix is smaller — matches the implicit
                // end-sentinel convention.
                lim_a < lim_b
            };
            (a_smaller, m, total)
        };

        if output_a {
            z[k] = arr_a[i_a];
            lcp_z[k] = I::from_usize(lcp_for_output);
            i_a += 1;
        } else {
            z[k] = arr_b[i_b];
            lcp_z[k] = I::from_usize(lcp_for_output);
            i_b += 1;
            // Outputting from B: swap labels so the next iteration's
            // "last-output stream" is what was B.
            std::mem::swap(&mut arr_a, &mut arr_b);
            std::mem::swap(&mut lcp_a, &mut lcp_b);
            std::mem::swap(&mut len_a, &mut len_b);
            std::mem::swap(&mut i_a, &mut i_b);
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
}
