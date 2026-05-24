//! Suffix comparison primitives over a generic text.
//!
//! Mirrors CaPS-SA's `Suffix_Array::LCP` family (see `include/Suffix_Array.hpp`
//! and `include/Genomic_Text.hpp`). Performance-critical callers (the
//! merge-sort and cascade-merge inner loops) construct a [`LcpDispatch`]
//! **once** at the top of the SA build and pass it through. The dispatch
//! holds a function pointer chosen by [`is_x86_feature_detected!`] /
//! [`is_aarch64_feature_detected!`] at construction time, so the hot path
//! is a single indirect call through a register — no per-call atomic
//! loads, no per-call feature-detection branches, no per-call `TypeId`
//! checks once monomorphization has resolved `S`.
//!
//! The free-standing [`lcp`] / [`suffix_cmp`] / [`lcp_u8`] helpers remain
//! for one-off callers (and for the tests in this file). They construct a
//! [`LcpDispatch`] on every call and are correspondingly slower; algorithm
//! kernels should prefer the methods on [`LcpDispatch`].

use std::any::TypeId;
use std::cmp::Ordering;

/// A function-pointer dispatch for byte-text LCP. The architecture-specific
/// pointer is selected once at construction by feature detection; later
/// calls reduce to a register-resident indirect call.
///
/// `LcpDispatch` is `Copy`, `Send`, and `Sync` (a function pointer is all
/// three), so it threads freely through `rayon` boundaries.
#[derive(Copy, Clone)]
pub struct LcpDispatch {
    lcp_u8_fn: LcpU8Fn,
}

/// Internal function-pointer type for the `u8`-specialized LCP path.
/// `unsafe fn` because the AVX2 / NEON variants are `#[target_feature]`
/// gated; the dispatch's owner has already verified CPU support.
type LcpU8Fn = unsafe fn(&[u8], usize, usize, usize) -> usize;

impl LcpDispatch {
    /// Detect the best LCP implementation for this CPU. Cheap (a couple
    /// of `is_*_feature_detected!` checks) but does still touch the
    /// feature-detection cache, so call it **once** per top-level build.
    pub fn detect() -> Self {
        Self {
            lcp_u8_fn: pick_lcp_u8_impl(),
        }
    }

    /// Forced scalar dispatch — useful for tests and for clients that
    /// want a deterministic baseline.
    pub fn scalar() -> Self {
        Self {
            lcp_u8_fn: lcp_u8_scalar,
        }
    }

    /// Longest common prefix of `text[p..]` and `text[q..]`, bounded by
    /// `max_ctx`. Dispatches to the captured byte-text fast path when
    /// `S` is `u8`; falls back to portable scalar for any other symbol
    /// type.
    #[inline]
    pub fn lcp<S>(&self, text: &[S], p: usize, q: usize, max_ctx: usize) -> usize
    where
        S: Eq + 'static,
    {
        if TypeId::of::<S>() == TypeId::of::<u8>() {
            // SAFETY: the `TypeId` check guarantees `S == u8`, so `&[S]`
            // and `&[u8]` are the same type at runtime. We reinterpret
            // the slice header without changing its length/alignment.
            // After monomorphization on a concrete `S` and with LTO this
            // whole branch folds to either the byte path or the scalar
            // path with no runtime test.
            let text_u8: &[u8] =
                unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };
            return unsafe { (self.lcp_u8_fn)(text_u8, p, q, max_ctx) };
        }
        lcp_scalar(text, p, q, max_ctx)
    }

    /// Total order on two suffixes of `text`. Uses [`Self::lcp`] for the
    /// shared prefix, then resolves the first differing symbol or — if
    /// both suffixes are exhausted within `max_ctx` — orders by remaining
    /// length (shorter is smaller, the convention SAIS and CaPS-SA use).
    #[inline]
    pub fn suffix_cmp<S>(&self, text: &[S], p: usize, q: usize, max_ctx: usize) -> Ordering
    where
        S: Ord + 'static,
    {
        let n = text.len();
        let lim_p = n - p;
        let lim_q = n - q;
        let lim = lim_p.min(lim_q).min(max_ctx);
        let common = self.lcp(text, p, q, max_ctx);
        if common < lim {
            text[p + common].cmp(&text[q + common])
        } else {
            lim_p.cmp(&lim_q)
        }
    }
}

/// One-off LCP. Constructs a fresh [`LcpDispatch`] on every call — fine
/// for tests / introspection, slower than reusing one [`LcpDispatch`]
/// across an algorithm's inner loop.
#[inline]
pub fn lcp<S>(text: &[S], p: usize, q: usize, max_ctx: usize) -> usize
where
    S: Eq + 'static,
{
    LcpDispatch::detect().lcp(text, p, q, max_ctx)
}

/// One-off suffix comparison; see [`lcp`] for the cost note.
#[inline]
pub fn suffix_cmp<S>(text: &[S], p: usize, q: usize, max_ctx: usize) -> Ordering
where
    S: Ord + 'static,
{
    LcpDispatch::detect().suffix_cmp(text, p, q, max_ctx)
}

/// One-off `u8`-typed LCP that auto-selects AVX2 / NEON / scalar. Skips
/// the generic `TypeId` branch; otherwise equivalent in cost to
/// [`lcp`] for byte texts.
#[inline]
pub fn lcp_u8(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    let f = pick_lcp_u8_impl();
    unsafe { f(text, p, q, max_ctx) }
}

/// Generic scalar LCP. Public so callers that already know their symbol
/// type isn't `u8` can skip both dispatch and `TypeId` check.
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

/// Inspect this CPU's features and return the best [`LcpU8Fn`].
fn pick_lcp_u8_impl() -> LcpU8Fn {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512BW gives us a 64-byte byte-compare returning a 64-bit
        // mask register directly — no movemask intrinsic, no extract.
        // Both `f` (foundation) and `bw` (byte/word ops, for the
        // `_mm512_cmpeq_epi8_mask` we use) are required.
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return lcp_u8_avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return lcp_u8_avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return lcp_u8_neon;
        }
    }
    lcp_u8_scalar
}

/// `unsafe fn`-typed scalar — wrapper around [`lcp_scalar`] so all three
/// dispatch targets share the [`LcpU8Fn`] signature.
unsafe fn lcp_u8_scalar(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    lcp_scalar(text, p, q, max_ctx)
}

/// AVX-512BW path: 64-byte vector compares; `_mm512_cmpeq_epi8_mask`
/// returns the per-byte equality mask straight in a 64-bit `__mmask64`
/// register (no movemask round-trip), and `(!mask).trailing_zeros()`
/// gives the first differing byte.
///
/// The function leads with a single 32-byte AVX2 step. This keeps the
/// short-LCP regime (random DNA, where every call typically resolves in
/// the first ≤16 bytes) at AVX2's per-call cost — a 64-byte load + ZMM
/// register usage on a call that exits inside the first 32 bytes is
/// wasted work. Once we've established the LCP exceeds 32 bytes we
/// switch to the 64-byte stride for the rest of the comparison, which
/// is the regime the upstream genome bench (and `lcp_u8_avx512`'s
/// reason for existing) actually hits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lcp_u8_avx512(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    use std::arch::x86_64::{
        __m256i, __m512i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm512_cmpeq_epi8_mask, _mm512_loadu_si512,
    };
    let n = text.len();
    let lim_p = n.saturating_sub(p).min(max_ctx);
    let lim_q = n.saturating_sub(q).min(max_ctx);
    let lim = lim_p.min(lim_q);
    let ptr = text.as_ptr();

    let mut i = 0usize;
    // 32-byte head: AVX2 compare. If it resolves the LCP we never touch
    // a ZMM register.
    if i + 32 <= lim {
        // SAFETY: AVX2 is implied by AVX-512F; bounds checked above.
        let va = unsafe { _mm256_loadu_si256(ptr.add(p + i) as *const __m256i) };
        let vb = unsafe { _mm256_loadu_si256(ptr.add(q + i) as *const __m256i) };
        let eq = _mm256_cmpeq_epi8(va, vb);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != u32::MAX {
            return i + (!mask).trailing_zeros() as usize;
        }
        i += 32;
    }
    // 64-byte body: LCP exceeds 32 bytes, run at AVX-512 stride.
    //
    // We tried software prefetching (`_mm_prefetch::<_MM_HINT_T0>(ptr+256)`)
    // inside this loop on the theory that overlapping the next stride's
    // memory fetch with current-iteration execution would shave the
    // dominant phase 1 wall. It did not: hum200m, rand100m, and the
    // full GRCh38 32t bench all showed indistinguishable wall vs the
    // bare loop. The Zen 5 hardware prefetcher recognises the strided
    // pattern and is already issuing the same loads, so the explicit
    // hint just adds inner-loop instructions for no benefit. Left bare
    // for clarity.
    while i + 64 <= lim {
        // SAFETY: bounds ensured by the loop condition; unaligned loads.
        let va = unsafe { _mm512_loadu_si512(ptr.add(p + i) as *const __m512i) };
        let vb = unsafe { _mm512_loadu_si512(ptr.add(q + i) as *const __m512i) };
        let mask = _mm512_cmpeq_epi8_mask(va, vb);
        if mask != u64::MAX {
            return i + (!mask).trailing_zeros() as usize;
        }
        i += 64;
    }
    // 32-byte tail: covers the case where the residue past the 64-byte
    // loop is between 32 and 63 bytes.
    if i + 32 <= lim {
        // SAFETY: bounds checked above.
        let va = unsafe { _mm256_loadu_si256(ptr.add(p + i) as *const __m256i) };
        let vb = unsafe { _mm256_loadu_si256(ptr.add(q + i) as *const __m256i) };
        let eq = _mm256_cmpeq_epi8(va, vb);
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != u32::MAX {
            return i + (!mask).trailing_zeros() as usize;
        }
        i += 32;
    }
    while i < lim {
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// AVX2 path: 32-byte vector compares, locate the first differing byte
/// via `_mm256_movemask_epi8` + `trailing_zeros`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lcp_u8_avx2(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
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
    while i < lim {
        if text[p + i] != text[q + i] {
            return i;
        }
        i += 1;
    }
    i
}

/// NEON path: 16-byte compares, locate the first differing byte via the
/// "shrn by 4" movemask emulation — pack each `vceqq_u8` byte
/// (`0xFF` or `0x00`) into 4 mask bits of a single 64-bit lane, then
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
        let eq = vceqq_u8(va, vb);
        let narrow = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
        let mask = vget_lane_u64::<0>(vreinterpret_u64_u8(narrow));
        if mask != u64::MAX {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_matches_to_first_difference() {
        let text = b"banana";
        // suffix at 0: "banana", at 1: "anana". LCP = 0 (b vs a).
        assert_eq!(lcp(text, 0, 1, usize::MAX), 0);
        // suffix at 1: "anana", at 3: "ana". LCP = 3 ("ana"), then diff
        // (n vs end).
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

    /// SIMD vs scalar agreement across pathological positions: long runs
    /// of identical bytes (exercises full-vector equal branches),
    /// 32-byte and 16-byte boundary differences (covers AVX2 and NEON
    /// chunk sizes), and unaligned tail bytes.
    #[test]
    fn simd_matches_scalar_on_u8() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA5A5);

        for diff_at in [0usize, 1, 31, 32, 33, 63, 64, 65, 100] {
            // Build a 400-byte text: first half is "AAA…CAAAA…" with C at
            // `diff_at`, second half is all A's. The LCP between suffix
            // 200 (all-A) and suffix 0 (has C at diff_at) is exactly
            // diff_at.
            let mut combined = vec![b'A'; 400];
            combined[diff_at] = b'C';
            let got = lcp(&combined, 0, 200, usize::MAX);
            assert_eq!(got, diff_at, "wrong LCP at diff_at={diff_at}");
        }

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

    /// Verify the cached-dispatch path matches both scalar and the
    /// one-off helper on a tricky case spanning AVX2/NEON vector
    /// boundaries.
    #[test]
    fn dispatch_struct_matches_oneoff_and_scalar() {
        let scalar = LcpDispatch::scalar();
        let detected = LcpDispatch::detect();
        let mut text: Vec<u8> = vec![b'A'; 200];
        text[64] = b'T'; // diff right at the second AVX2 boundary
        assert_eq!(scalar.lcp(&text, 0, 100, usize::MAX), 64);
        assert_eq!(detected.lcp(&text, 0, 100, usize::MAX), 64);
    }

    /// Exercise the AVX-512 64-byte stride and the 32-byte tail: place
    /// the differing byte at offsets that straddle each boundary
    /// (0/63/64/65/95/96/97/127/128) and confirm the dispatched path
    /// agrees with scalar. On a non-AVX-512 host this devolves to the
    /// other SIMD paths but still verifies correctness.
    #[test]
    fn avx512_boundary_agreement() {
        let detected = LcpDispatch::detect();
        for diff_at in [0usize, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 200] {
            let mut text = vec![b'A'; 512];
            text[diff_at] = b'G';
            let got = detected.lcp(&text, 0, 256, usize::MAX);
            assert_eq!(got, diff_at, "diff_at={diff_at}");
        }
    }
}
