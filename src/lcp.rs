//! Suffix comparison primitives over a generic text.
//!
//! Mirrors CaPS-SA's `Suffix_Array::LCP` family (see `include/Suffix_Array.hpp`
//! and `include/Genomic_Text.hpp`). The implementation here is the scalar /
//! portable variant — SIMD-unrolled paths can be added later without changing
//! the call sites.
//!
//! The `lcp_*` functions return the length of the common prefix of two
//! suffixes, bounded by `max_ctx`. They do **not** decide ordering; callers
//! resolve ordering from the symbol immediately past the common prefix
//! (or, if both suffixes are exhausted within `max_ctx`, from their positions —
//! the standard "shorter suffix is smaller" convention only triggers when the
//! actual text length cuts the comparison off).

use std::cmp::Ordering;

/// Longest common prefix of the suffixes `text[p..]` and `text[q..]`,
/// bounded by `max_ctx` symbols.
///
/// Compares symbol-wise; returns the first index `i < max_ctx` such that
/// either `text[p + i]` or `text[q + i]` is past the end of `text`, or the
/// two symbols differ. If everything matches up to `max_ctx`, returns `max_ctx`.
#[inline]
pub fn lcp<S: Eq>(text: &[S], p: usize, q: usize, max_ctx: usize) -> usize {
    let n = text.len();
    let lim_p = n.saturating_sub(p).min(max_ctx);
    let lim_q = n.saturating_sub(q).min(max_ctx);
    let lim = lim_p.min(lim_q);

    // Scalar byte-at-a-time. A future SIMD variant (AVX2/AVX-512) can replace
    // this for `u8` texts — the CaPS-SA C++ uses 32- or 64-byte strides.
    let mut i = 0;
    while i < lim {
        // SAFETY: `i < lim`, and `lim` is bounded by `n - p` and `n - q`.
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// Total order on two suffixes of `text`, using `lcp` to find the first
/// differing symbol. If one suffix is a prefix of the other within `text`'s
/// bounds (one runs off the end first), the *shorter* one is considered
/// smaller, matching the standard "$ is smallest" convention used by SAIS and
/// CaPS-SA. With distinct end-sentinels in `text`, this branch is unreachable
/// for distinct positions.
#[inline]
pub fn suffix_cmp<S: Ord>(text: &[S], p: usize, q: usize, max_ctx: usize) -> Ordering {
    let n = text.len();
    let lim_p = n - p;
    let lim_q = n - q;
    let lim = lim_p.min(lim_q).min(max_ctx);

    let mut i = 0;
    while i < lim {
        match text[p + i].cmp(&text[q + i]) {
            Ordering::Equal => i += 1,
            other => return other,
        }
    }
    // Exhausted within `max_ctx` — the shorter remaining tail is smaller.
    lim_p.cmp(&lim_q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_matches_to_first_difference() {
        let text = b"banana";
        // suffix at 0: "banana", at 1: "anana". LCP = 0 (b vs a).
        assert_eq!(lcp(text, 0, 1, usize::MAX), 0);
        // suffix at 1: "anana", at 3: "ana". LCP = 3 ("ana"), then diff (n vs end).
        assert_eq!(lcp(text, 1, 3, usize::MAX), 3);
    }

    #[test]
    fn lcp_respects_max_ctx() {
        let text = b"aaaaaa";
        assert_eq!(lcp(text, 0, 1, 3), 3);
    }

    #[test]
    fn lcp_stops_at_text_end() {
        let text = b"abc";
        // suffix at 0: "abc", at 2: "c". LCP=0.
        assert_eq!(lcp(text, 0, 2, usize::MAX), 0);
        // suffix at 1: "bc", at 1: "bc". LCP=2.
        assert_eq!(lcp(text, 1, 1, usize::MAX), 2);
    }

    #[test]
    fn cmp_lex_order() {
        let text = b"banana";
        // "anana" < "banana"
        assert_eq!(suffix_cmp(text, 1, 0, usize::MAX), Ordering::Less);
        // "ana" < "anana" (prefix)
        assert_eq!(suffix_cmp(text, 3, 1, usize::MAX), Ordering::Less);
        // self-equal
        assert_eq!(suffix_cmp(text, 1, 1, usize::MAX), Ordering::Equal);
    }
}
