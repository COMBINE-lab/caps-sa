//! Suffix comparison primitives over a generic text.
//!
//! Mirrors CaPS-SA's `Suffix_Array::LCP` family (see `include/Suffix_Array.hpp`
//! and `include/Genomic_Text.hpp`). The generic `lcp` and `suffix_cmp`
//! entry points dispatch at runtime to a [`u8`]-specialized SIMD path
//! (AVX2 on x86_64, NEON on aarch64) when the symbol type is `u8`; for
//! any other type, they fall back to a portable scalar byte-loop.
//!
//! The `lcp_*` functions return the length of the common prefix of two
//! suffixes, bounded by `max_ctx`. They do **not** decide ordering;
//! callers resolve ordering from the symbol immediately past the common
//! prefix (or, if both suffixes are exhausted within `max_ctx`, from
//! their positions — the standard "shorter suffix is smaller"
//! convention only triggers when the actual text length cuts the
//! comparison off).

use std::any::TypeId;
use std::cmp::Ordering;

/// Longest common prefix of the suffixes `text[p..]` and `text[q..]`,
/// bounded by `max_ctx` symbols.
///
/// When `S` is `u8`, dispatches to AVX2 (x86_64) or NEON (aarch64) at
/// runtime via [`is_x86_feature_detected!`] / [`is_aarch64_feature_detected!`];
/// otherwise uses a scalar byte-loop. The `'static` bound is required
/// for the `TypeId`-based runtime check.
#[inline]
pub fn lcp<S>(text: &[S], p: usize, q: usize, max_ctx: usize) -> usize
where
    S: Eq + 'static,
{
    if TypeId::of::<S>() == TypeId::of::<u8>() {
        // SAFETY: the `TypeId` check guarantees `S == u8`, so `&[S]` and
        // `&[u8]` are the same type at runtime. We reinterpret the slice
        // header without changing its length/alignment.
        let text_u8: &[u8] =
            unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };
        return lcp_u8(text_u8, p, q, max_ctx);
    }
    lcp_scalar(text, p, q, max_ctx)
}

/// Generic scalar LCP fallback. Public so callers that already know
/// their symbol type isn't `u8` can skip the dispatch.
#[inline]
pub fn lcp_scalar<S: Eq>(text: &[S], p: usize, q: usize, max_ctx: usize) -> usize {
    let n = text.len();
    let lim_p = n.saturating_sub(p).min(max_ctx);
    let lim_q = n.saturating_sub(q).min(max_ctx);
    let lim = lim_p.min(lim_q);
    let mut i = 0;
    while i < lim {
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// `u8`-specialized LCP. Runtime-dispatches to AVX2 or NEON when the
/// CPU supports it; otherwise scalar.
#[inline]
pub fn lcp_u8(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature check is positive.
            return unsafe { lcp_u8_avx2(text, p, q, max_ctx) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature check is positive.
            return unsafe { lcp_u8_neon(text, p, q, max_ctx) };
        }
    }
    lcp_scalar(text, p, q, max_ctx)
}

/// AVX2 path: 32-byte vector compares, locate the first differing byte
/// via `_mm256_movemask_epi8` + `trailing_zeros`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lcp_u8_avx2(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
    };
    let n = text.len();
    let lim_p = n.saturating_sub(p).min(max_ctx);
    let lim_q = n.saturating_sub(q).min(max_ctx);
    let lim = lim_p.min(lim_q);
    let ptr = text.as_ptr();

    let mut i = 0usize;
    while i + 32 <= lim {
        // SAFETY: bounds ensured by the loop condition; unaligned loads.
        let va = unsafe { _mm256_loadu_si256(ptr.add(p + i) as *const __m256i) };
        let vb = unsafe { _mm256_loadu_si256(ptr.add(q + i) as *const __m256i) };
        let eq = _mm256_cmpeq_epi8(va, vb);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != u32::MAX {
            return i + (!mask).trailing_zeros() as usize;
        }
        i += 32;
    }
    // Tail: scalar.
    while i < lim {
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// NEON path: 16-byte compares, locate the first differing byte via the
/// "shrn by 4" movemask emulation — pack each `vceqq_u8` byte (`0xFF` or
/// `0x00`) into 4 mask bits of a single 64-bit lane, then
/// `trailing_zeros / 4` gives the byte index.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lcp_u8_neon(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    use std::arch::aarch64::{
        vceqq_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8, vshrn_n_u16,
    };
    let n = text.len();
    let lim_p = n.saturating_sub(p).min(max_ctx);
    let lim_q = n.saturating_sub(q).min(max_ctx);
    let lim = lim_p.min(lim_q);
    let ptr = text.as_ptr();

    let mut i = 0usize;
    while i + 16 <= lim {
        // SAFETY: bounds ensured by the loop condition; unaligned loads.
        let va = unsafe { vld1q_u8(ptr.add(p + i)) };
        let vb = unsafe { vld1q_u8(ptr.add(q + i)) };
        let eq = unsafe { vceqq_u8(va, vb) };
        let narrow = unsafe { vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq)) };
        let mask = unsafe { vget_lane_u64::<0>(vreinterpret_u64_u8(narrow)) };
        if mask != u64::MAX {
            // First differing nibble = first differing byte; 4 mask bits
            // cover one byte of the original compare.
            return i + ((!mask).trailing_zeros() as usize / 4);
        }
        i += 16;
    }
    while i < lim {
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// Total order on two suffixes of `text`. Uses [`lcp`] to find the first
/// differing symbol (so `u8` texts benefit from the SIMD path); when both
/// suffixes are exhausted within `max_ctx`, the shorter remaining tail
/// is smaller, matching the "$ is smallest" convention used by SAIS and
/// CaPS-SA. With distinct end-sentinels in `text`, that branch is
/// unreachable for distinct positions.
#[inline]
pub fn suffix_cmp<S>(text: &[S], p: usize, q: usize, max_ctx: usize) -> Ordering
where
    S: Ord + 'static,
{
    let n = text.len();
    let lim_p = n - p;
    let lim_q = n - q;
    let lim = lim_p.min(lim_q).min(max_ctx);
    let common = lcp(text, p, q, max_ctx);
    if common < lim {
        text[p + common].cmp(&text[q + common])
    } else {
        lim_p.cmp(&lim_q)
    }
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

    /// SIMD vs scalar agreement across pathological positions: long
    /// runs of identical bytes (exercises full-vector equal branches),
    /// 32-byte and 16-byte boundary differences (covers AVX2 and NEON
    /// chunk sizes), and unaligned tail bytes.
    #[test]
    fn simd_matches_scalar_on_u8() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA5A5);

        // (1) Long aaaa run with one mismatched byte at varying offsets.
        for diff_at in [0usize, 1, 31, 32, 33, 63, 64, 65, 100] {
            let mut text: Vec<u8> = vec![b'A'; 200];
            text[diff_at] = b'C';
            // Compare suffix(0) and suffix(0) shifted by 0 — they're
            // equal. So compare suffix(0) with a copy where the byte
            // differs at diff_at:
            // we build a 400-byte text where the first 200 are AAA... C ...AAA
            // and the next 200 are pure A's; compare suffix(200) (all-A)
            // with suffix(0) (has C at diff_at).
            let mut combined = vec![b'A'; 400];
            combined[diff_at] = b'C';
            let got = lcp(&combined, 0, 200, usize::MAX);
            assert_eq!(got, diff_at, "wrong LCP at diff_at={diff_at}");
        }

        // (2) Random texts cross-checked against scalar.
        for &n in &[1usize, 32, 33, 200, 1000] {
            let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..4u8)).collect();
            for _ in 0..20 {
                let p = rng.random_range(0..n);
                let q = rng.random_range(0..n);
                let want = lcp_scalar(&text, p, q, usize::MAX);
                let got = lcp(&text, p, q, usize::MAX);
                assert_eq!(got, want, "p={p} q={q} text={text:?}");
            }
        }
    }
}
