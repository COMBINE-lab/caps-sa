//! Packed fixed-depth keys, and the phase-1 subarray seed built on them.
//!
//! Phase 1 sorts each subarray from singletons, which is the case the merge
//! kernel handles worst: every leaf merge starts at `m = 0` and orders two
//! suffixes by scanning the text at two random addresses. Sorting by a packed
//! key first resolves the leading `k` symbols with no text access at all (16
//! symbols for a 6-letter genome alphabet at 4 bits each), and yields the LCP
//! between adjacent runs for free from `(key_a ^ key_b).leading_zeros()`. Only
//! suffixes agreeing through the whole key reach the merge kernel, on the
//! short slice they occupy.
//!
//! This is the bounded-memory counterpart to prefix doubling. Doubling itself
//! is not available here: it needs a rank for every position in the text,
//! which is exactly the memory the external-memory path refuses to spend. A
//! fixed-depth key needs one `u64` per record of the subarray in flight.
//!
//! The key is segment-aware. It packs `min(k, lim_at(p))` symbols, so it never
//! reads into the next segment, and pads with a reserved sentinel placed on
//! the side the provider's [`BoundaryRank`] demands. That is what lets a
//! splice-junction index, where the segment boundary rather than the text end
//! terminates most suffixes, take the path at all.

use crate::Index;
use crate::lcp::{LcpDispatch, Symbol};
use crate::limits::{BoundaryRank, LimitProvider};
use crate::sample_sort;
use rayon::prelude::*;

/// An order-preserving remap of the bytes that actually occur in a text onto
/// a dense code range, plus the resulting key geometry.
///
/// The field width is driven by how many *distinct* symbols a text uses, not
/// by the largest byte value in it, and the difference is not academic. A raw
/// FASTA uses six symbols, but the largest is `'T'` (84), so packing raw bytes
/// forces 8-bit fields and fits only 8 symbols per key. Ranking those six
/// bytes to `0..6` gives 4-bit fields and 16 symbols per key. A rustar-shaped
/// text is already dense (`0..=5`), so it pays nothing for the map.
///
/// The map is monotone by construction, since codes are assigned in ascending
/// byte order. That is what keeps a packed key order-preserving: `key_a <
/// key_b` still implies `suffix_a < suffix_b`.
pub(crate) struct Packer {
    /// The text with every byte replaced by its code, when the identity map
    /// does not already do that. Materializing it once removes a dependent
    /// table load from the packing loop. `None` when the text is already
    /// dense, so the common pre-coded genomic input pays no extra memory.
    ranked: Option<Vec<u8>>,
    /// Bits per packed field.
    bits: u32,
    /// Symbols per `u64` key.
    k: usize,
    /// Number of distinct codes in use. Codes are `0..alphabet`; `alphabet`
    /// itself is the boundary sentinel when it fits the field.
    alphabet: u32,
}

impl Packer {
    /// Build the map for `text`. The field is sized to hold `alphabet`, not
    /// `alphabet - 1`, because every key this module builds reserves one code
    /// for the boundary sentinel.
    fn new(text: &[u8]) -> Self {
        // Which bytes occur? One parallel pass, folded into a 256-entry set.
        let present = text
            .par_chunks(1 << 16)
            .map(|c| {
                let mut seen = [false; 256];
                for &b in c {
                    seen[b as usize] = true;
                }
                seen
            })
            .reduce(
                || [false; 256],
                |mut a, b| {
                    for i in 0..256 {
                        a[i] |= b[i];
                    }
                    a
                },
            );

        let mut code = [0u8; 256];
        let mut next = 0u16;
        let mut identity = true;
        for (b, &seen) in present.iter().enumerate() {
            if seen {
                code[b] = next as u8;
                identity &= next as usize == b;
                next += 1;
            }
        }
        let bits: u32 = match next {
            0..=1 => 1,
            2..=3 => 2,
            4..=15 => 4,
            _ => 8,
        };
        let ranked = if identity {
            None
        } else {
            let mut out = vec![0u8; text.len()];
            out.par_chunks_mut(1 << 16)
                .zip(text.par_chunks(1 << 16))
                .for_each(|(dst, src)| {
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = code[s as usize];
                    }
                });
            Some(out)
        };
        Self {
            ranked,
            bits,
            k: 64 / bits as usize,
            alphabet: next as u32,
        }
    }

    /// Whether a boundary sentinel fits alongside the alphabet in one field.
    ///
    /// With 8-bit fields and 256 distinct symbols there is no spare code, so
    /// keys are unavailable and the caller must fall back.
    #[inline]
    fn has_sentinel(&self) -> bool {
        (self.alphabet as u64) < (1u64 << self.bits)
    }

    /// Pack the `min(k, lim)` symbols at `text[p..]`, padding the rest with a
    /// boundary sentinel placed according to `rank`.
    ///
    /// Never reads past `p + lim`, so a key cannot see into the next segment.
    /// Under `ShorterFirst` the sentinel is code `0` and every real code is
    /// shifted up by one, so a padded field is strictly below any real symbol.
    /// Under `LongerFirst` the sentinel is `alphabet`, strictly above every
    /// real code, and no shift is needed.
    #[inline]
    fn key_at_bounded(&self, text: &[u8], p: usize, lim: usize, rank: BoundaryRank) -> u64 {
        debug_assert!(self.has_sentinel());
        let src = self.ranked.as_deref().unwrap_or(text);
        let take = self.k.min(lim).min(src.len() - p);
        let (bias, pad) = match rank {
            BoundaryRank::ShorterFirst => (1u64, 0u64),
            BoundaryRank::LongerFirst => (0u64, self.alphabet as u64),
        };
        let mut key = 0u64;
        for &c in &src[p..p + take] {
            key = (key << self.bits) | (c as u64 + bias);
        }
        for _ in take..self.k {
            key = (key << self.bits) | pad;
        }
        key
    }

    /// Symbols the two keys share, from their first differing field.
    #[inline]
    fn shared_fields(&self, key_a: u64, key_b: u64) -> usize {
        ((key_a ^ key_b).leading_zeros() / self.bits) as usize
    }
}

/// The alphabet map for `text`, or `None` when a packed key cannot represent
/// this text's order.
///
/// Computed once per build and handed to [`seed_subarray`], which would
/// otherwise re-scan the whole text for every subarray.
pub(crate) fn seed_params<S: Symbol>(text: &[S]) -> Option<Packer> {
    // Exactly `u8`, not merely one byte wide. `Symbol` is implemented for
    // `i8` too, and a packed key orders its fields as unsigned: `-1` has byte
    // `0xFF` and would sort above `1`, inverting the text's real order.
    if std::any::TypeId::of::<S>() != std::any::TypeId::of::<u8>() {
        return None;
    }
    // SAFETY: `S` is `u8` (just checked by `TypeId`, and `Symbol: 'static` so
    // the comparison is exact), so a byte view over the same memory is valid
    // for reads of the same length.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };
    let packer = Packer::new(bytes);
    packer.has_sentinel().then_some(packer)
}

/// Sort `sa` into suffix order and fill `lcp`, using a packed fixed-depth key
/// so that most of the ordering costs no text access at all.
///
/// Returns `false` without touching anything when the key cannot represent
/// this comparator, so the caller falls back to a plain `merge_sort`.
///
/// `sa_w` and `lcp_w` are the caller's existing merge scratch buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn seed_subarray<S: Symbol, I: Index, L: LimitProvider>(
    text: &[S],
    lp: &L,
    packer: Option<&Packer>,
    sa: &mut [I],
    lcp: &mut [I],
    sa_w: &mut [I],
    lcp_w: &mut [I],
    max_ctx: usize,
    dispatch: LcpDispatch,
    task_local: bool,
) -> bool {
    let Some(packer) = packer else {
        return false;
    };
    // A finite `max_context` truncates comparisons at a depth the key knows
    // nothing about, so the key's verdict and the merge's could disagree.
    if max_ctx != usize::MAX {
        return false;
    }
    let Some(rank) = lp.boundary_rank() else {
        return false;
    };
    let len = sa.len();
    if len < 2 {
        if len == 1 {
            lcp[0] = I::zero();
        }
        return true;
    }
    // SAFETY: `packer` is `Some` only when `S` is exactly `u8`.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };

    let mut keyed: Vec<(u64, I)> = sa
        .iter()
        .map(|&p| {
            let pu = p.to_usize();
            (packer.key_at_bounded(bytes, pu, lp.lim_at(pu), rank), p)
        })
        .collect();
    keyed.sort_unstable_by_key(|e| e.0);
    for (slot, e) in sa.iter_mut().zip(keyed.iter()) {
        *slot = e.1;
    }

    // Walk equal-key runs. Between runs the LCP falls straight out of the key
    // difference; inside one it needs the merge kernel.
    let mut i = 0usize;
    while i < len {
        let mut j = i + 1;
        while j < len && keyed[j].0 == keyed[i].0 {
            j += 1;
        }
        if j - i > 1 {
            // Mirror the caller's nesting choice. Phase 1 runs one task per
            // subarray once `p` reaches the worker count, and a rayon-splitting
            // merge inside such a task only adds scheduling on top of
            // parallelism that already saturates.
            let sort = if task_local {
                sample_sort::merge_sort_task_local
            } else {
                sample_sort::merge_sort
            };
            sort(
                text,
                lp,
                &mut sa[i..j],
                &mut sa_w[i..j],
                &mut lcp[i..j],
                &mut lcp_w[i..j],
                max_ctx,
                dispatch,
            );
        } else {
            lcp[i] = I::zero();
        }
        // Boundary entry: LCP against the last element of the previous run.
        if i > 0 {
            let a = sa[i - 1].to_usize();
            let b = sa[i].to_usize();
            let shared = packer.shared_fields(keyed[i - 1].0, keyed[i].0);
            // Cap by both suffixes' limits: a sentinel field can agree with a
            // real symbol's field past the end of the shorter suffix, so the
            // raw count can overstate the true LCP, and a wrong LCP would
            // silently corrupt the order at the next merge level.
            lcp[i] = I::from_usize(shared.min(lp.lim_at(a)).min(lp.lim_at(b)));
        }
        i = j;
    }
    lcp[0] = I::zero();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::{PlainText, SegmentedText};

    /// STAR's convention: the suffix that hits its boundary first is larger.
    struct StarSegmented {
        inner: SegmentedText,
    }

    impl LimitProvider for StarSegmented {
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

        fn boundary_rank(&self) -> Option<BoundaryRank> {
            Some(BoundaryRank::LongerFirst)
        }
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    /// Sort `positions` with the seed, then check the result is a permutation
    /// in non-decreasing suffix order under the provider's own comparator.
    ///
    /// The assertion is the *property*, not equality with a canonical answer:
    /// `SegmentedText`'s default `boundary_order` returns `Equal` for suffixes
    /// that end together with equal content, so their relative order is
    /// genuinely free and a stable-sort oracle is not a valid reference.
    fn check_sorted<L: LimitProvider>(text: &[u8], lp: &L, positions: Vec<u32>) {
        let packer = seed_params(text);
        assert!(packer.is_some(), "packer should be available for this text");
        let len = positions.len();
        let mut sa = positions.clone();
        let mut lcp = vec![0u32; len];
        let mut sa_w = vec![0u32; len];
        let mut lcp_w = vec![0u32; len];
        let dispatch = LcpDispatch::detect();
        let took = seed_subarray(
            text,
            lp,
            packer.as_ref(),
            &mut sa,
            &mut lcp,
            &mut sa_w,
            &mut lcp_w,
            usize::MAX,
            dispatch,
            true,
        );
        assert!(took, "the seed should have taken this input");

        let mut seen = sa.clone();
        seen.sort_unstable();
        let mut want = positions;
        want.sort_unstable();
        assert_eq!(seen, want, "seed must permute its input");

        for w in sa.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            assert!(
                dispatch.suffix_cmp_with(text, lp, a, b, usize::MAX).is_le(),
                "adjacent pair out of order: {a} then {b}",
            );
        }
    }

    #[test]
    fn seed_orders_a_plain_text() {
        let mut state = 12345u64;
        let text: Vec<u8> = (0..4096).map(|_| (lcg(&mut state) % 4) as u8).collect();
        let lp = PlainText::new(text.len());
        check_sorted(&text, &lp, (0..text.len() as u32).collect());
    }

    #[test]
    fn seed_orders_a_segmented_text_shorter_first() {
        let mut state = 999u64;
        // rustar's alphabet: bases 0..=3, N = 4, spacer = 5.
        let text: Vec<u8> = (0..8192)
            .map(|i| {
                if i % 617 < 9 {
                    5
                } else {
                    (lcg(&mut state) % 5) as u8
                }
            })
            .collect();
        let ends = spacer_ends(&text);
        let lp = SegmentedText::from_ends(text.len(), ends);
        let acgt: Vec<u32> = (0..text.len() as u32)
            .filter(|&p| text[p as usize] < 4)
            .collect();
        check_sorted(&text, &lp, acgt);
    }

    #[test]
    fn seed_orders_a_segmented_text_star_order() {
        let mut state = 4242u64;
        let text: Vec<u8> = (0..8192)
            .map(|i| {
                if i % 411 < 7 {
                    5
                } else {
                    (lcg(&mut state) % 5) as u8
                }
            })
            .collect();
        let ends = spacer_ends(&text);
        let lp = StarSegmented {
            inner: SegmentedText::from_ends(text.len(), ends),
        };
        let acgt: Vec<u32> = (0..text.len() as u32)
            .filter(|&p| text[p as usize] < 4)
            .collect();
        check_sorted(&text, &lp, acgt);
    }

    #[test]
    fn seed_declines_a_provider_without_a_rank() {
        struct NoRank(usize);
        impl LimitProvider for NoRank {
            fn lim_at(&self, p: usize) -> usize {
                self.0 - p
            }
        }
        let text: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packer = seed_params(&text);
        let mut sa: Vec<u32> = (0..8).collect();
        let mut lcp = vec![0u32; 8];
        let mut sa_w = vec![0u32; 8];
        let mut lcp_w = vec![0u32; 8];
        assert!(!seed_subarray(
            &text,
            &NoRank(text.len()),
            packer.as_ref(),
            &mut sa,
            &mut lcp,
            &mut sa_w,
            &mut lcp_w,
            usize::MAX,
            LcpDispatch::detect(),
            true,
        ));
    }

    /// Segment ends for a spacer-separated text: one past each maximal
    /// non-spacer run, closing at the text length.
    fn spacer_ends(text: &[u8]) -> Vec<u64> {
        let mut ends = Vec::new();
        let mut in_run = false;
        for (i, &b) in text.iter().enumerate() {
            if b == 5 {
                if in_run {
                    ends.push(i as u64);
                    in_run = false;
                }
            } else {
                in_run = true;
            }
        }
        if ends.last() != Some(&(text.len() as u64)) {
            ends.push(text.len() as u64);
        }
        ends
    }
}
