//! Suffix comparison primitives over a generic text.
//!
//! Mirrors CaPS-SA's `Suffix_Array::LCP` family (see `include/Suffix_Array.hpp`
//! and `include/Genomic_Text.hpp`). Performance-critical callers (the
//! merge-sort and cascade-merge inner loops) construct a [`LcpDispatch`]
//! **once** at the top of the SA build and pass it through. The dispatch
//! holds a function pointer chosen by [`is_x86_feature_detected!`] /
//! [`is_aarch64_feature_detected!`] at construction time, so the hot path
//! is a single indirect call through a register — no per-call atomic
//! loads, no per-call feature-detection branches.
//!
//! The free-standing [`lcp`] / [`suffix_cmp`] / [`lcp_u8`] helpers remain
//! for one-off callers (and for the tests in this file). They construct a
//! [`LcpDispatch`] on every call and are correspondingly slower; algorithm
//! kernels should prefer the methods on [`LcpDispatch`].
//!
//! ## Symbol types
//!
//! The whole crate is generic over any [`Symbol`] type. `Symbol` is an
//! `unsafe` marker for types whose in-memory bytes encode equality
//! (`a == b` iff their byte representations are equal) — i.e. no padding,
//! no invalid bit patterns. Blanket impls are provided for every stdlib
//! integer type and for fixed-size arrays of `Symbol`s, so `u8`, `u16`,
//! `u32`, `u64`, `[u8; 3]` (24-bit), … all work out of the box. The LCP
//! function casts `&[S]` to a byte view, runs a single byte-level SIMD
//! compare (AVX-512BW hybrid → AVX2 → NEON → scalar), then divides the
//! byte-LCP by `size_of::<S>()` to get the symbol-LCP. Endianness is
//! irrelevant because the byte-compare resolves equality only; symbol
//! ordering is recovered by the caller's `text[lcp].cmp(&text[lcp + 1])`
//! using `S`'s native `Ord`.

use std::cmp::Ordering;

use crate::limits::{LimitProvider, PlainText};

/// A symbol type for suffix-array construction. All stdlib unsigned
/// and signed integer types satisfy this, as does `[T; N]` for any
/// `T: Symbol` (arrays have no padding and inherit `Ord`
/// lexicographically from `T`). To use a custom type as a symbol,
/// implement this trait yourself; if the type contains padding, mark
/// it `#[repr(C, packed)]` first.
///
/// The trait bundles every other bound the algorithm needs from a
/// symbol type ([`Ord`] + [`Copy`] + [`Send`] + [`Sync`] + `'static`),
/// so the public API surface only ever needs `S: Symbol`.
///
/// # Safety
///
/// Implementations must guarantee that bit-equality of the in-memory
/// representation implies value-equality — i.e. **no padding bytes**,
/// **no invalid bit patterns** that two distinct values could share.
/// The byte-view SIMD LCP path in [`LcpDispatch::lcp`] casts `&[S]`
/// to a `&[u8]` view and compares bytes; if two distinct `S` values
/// could have identical bytes, or one `S` value could have two
/// different byte representations (e.g. via uninitialised padding),
/// the LCP function would return wrong answers and corrupt the
/// resulting suffix array.
pub unsafe trait Symbol: Ord + Copy + Send + Sync + 'static {}

macro_rules! impl_symbol_for_primitives {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: stdlib primitive integers have no padding and every
            // bit pattern is a valid value, so byte-equality is exactly
            // value-equality.
            unsafe impl Symbol for $t {}
        )*
    };
}
impl_symbol_for_primitives!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,
);

// SAFETY: arrays of `Symbol`s have no padding (Rust arrays are tightly
// packed) and inherit equality element-wise, so byte-equality of the
// whole array is exactly value-equality. This covers patterns like
// `[u8; 3]` for 24-bit alphabets and `[u32; 2]` for 64-bit-on-32-bit.
unsafe impl<T: Symbol, const N: usize> Symbol for [T; N] {}

/// A function-pointer dispatch for byte-level LCP. The architecture-
/// specific pointer is selected once at construction by feature
/// detection; later calls reduce to a register-resident indirect call.
///
/// `LcpDispatch` is `Copy`, `Send`, and `Sync` (a function pointer is
/// all three), so it threads freely through `rayon` boundaries.
///
/// The same byte-level function backs every symbol width: the
/// [`Self::lcp`] method casts `&[S]` to a byte view, calls the function
/// with byte-scale offsets, and divides the result by `size_of::<S>()`
/// to recover the symbol-level LCP.
#[derive(Copy, Clone)]
pub struct LcpDispatch {
    lcp_bytes_fn: LcpBytesFn,
}

/// Internal function-pointer type for the byte-level LCP path.
/// `unsafe fn` because the AVX2 / AVX-512 / NEON variants are
/// `#[target_feature]` gated; the dispatch's owner has already
/// verified CPU support.
type LcpBytesFn = unsafe fn(&[u8], usize, usize, usize) -> usize;

impl LcpDispatch {
    /// Detect the best LCP implementation for this CPU. Cheap (a couple
    /// of `is_*_feature_detected!` checks) but does still touch the
    /// feature-detection cache, so call it **once** per top-level build.
    pub fn detect() -> Self {
        Self {
            lcp_bytes_fn: pick_lcp_bytes_impl(),
        }
    }

    /// Forced scalar dispatch — useful for tests and for clients that
    /// want a deterministic baseline.
    pub fn scalar() -> Self {
        Self {
            lcp_bytes_fn: lcp_bytes_scalar,
        }
    }

    /// Longest common prefix of `text[p..]` and `text[q..]` in symbols,
    /// bounded by `max_ctx`. For any `S: Symbol` of non-zero size, this
    /// dispatches to the byte-level SIMD path with byte-scaled offsets
    /// and returns `byte_lcp / size_of::<S>()`.
    #[inline]
    pub fn lcp<S: Symbol>(&self, text: &[S], p: usize, q: usize, max_ctx: usize) -> usize {
        let k = std::mem::size_of::<S>();
        if k == 0 {
            // ZSTs: `Symbol` permits ZSTs (e.g. a unit `struct Foo;`),
            // but the byte-view dispatch can't divide by zero. Such a
            // text has zero bytes regardless of length, so every suffix
            // is identical and the LCP is just the length-bounded `lim`.
            let lim_p = text.len().saturating_sub(p).min(max_ctx);
            let lim_q = text.len().saturating_sub(q).min(max_ctx);
            return lim_p.min(lim_q);
        }
        // SAFETY: `Symbol`'s `unsafe` contract is exactly "bit-equality
        // is value-equality" — `&[S]` has the same byte representation
        // as a `&[u8]` view over the same bytes, with no padding to
        // worry about. `size_of_val(text)` gives the slice's exact
        // byte length, which Rust's slice invariant already guarantees
        // fits in `isize`.
        let bytes =
            unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, size_of_val(text)) };
        let byte_lcp = unsafe {
            (self.lcp_bytes_fn)(
                bytes,
                p.saturating_mul(k),
                q.saturating_mul(k),
                max_ctx.saturating_mul(k),
            )
        };
        byte_lcp / k
    }

    /// Total order on two suffixes of `text`. Uses [`Self::lcp`] for the
    /// shared prefix, then resolves the first differing symbol or — if
    /// both suffixes are exhausted within `max_ctx` — orders by remaining
    /// length (shorter is smaller, the convention SAIS and CaPS-SA use).
    ///
    /// Zero-cost wrapper around [`Self::suffix_cmp_with`] for the
    /// non-segmented case.
    #[inline]
    pub fn suffix_cmp<S: Symbol>(
        &self,
        text: &[S],
        p: usize,
        q: usize,
        max_ctx: usize,
    ) -> Ordering {
        self.suffix_cmp_with(text, &PlainText::new(text.len()), p, q, max_ctx)
    }

    /// Like [`Self::suffix_cmp`] but takes a [`LimitProvider`] so the
    /// suffix lengths used for the LCP-cap and the "shorter-is-smaller"
    /// tie-break come from a segmented view of the text. With
    /// [`PlainText`] this matches [`Self::suffix_cmp`] exactly.
    #[inline]
    pub fn suffix_cmp_with<S: Symbol, L: LimitProvider>(
        &self,
        text: &[S],
        lp: &L,
        p: usize,
        q: usize,
        max_ctx: usize,
    ) -> Ordering {
        let lim_p = lp.lim_at(p);
        let lim_q = lp.lim_at(q);
        let lim = lim_p.min(lim_q).min(max_ctx);
        let common = self.lcp(text, p, q, lim);
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
pub fn lcp<S: Symbol>(text: &[S], p: usize, q: usize, max_ctx: usize) -> usize {
    LcpDispatch::detect().lcp(text, p, q, max_ctx)
}

/// One-off suffix comparison; see [`lcp`] for the cost note.
#[inline]
pub fn suffix_cmp<S: Symbol>(text: &[S], p: usize, q: usize, max_ctx: usize) -> Ordering {
    LcpDispatch::detect().suffix_cmp(text, p, q, max_ctx)
}

/// One-off `u8`-typed LCP that auto-selects AVX-512 / AVX2 / NEON /
/// scalar. Convenience entry point for byte texts; equivalent in cost
/// to `LcpDispatch::detect().lcp::<u8>(...)` but skips the generic
/// indirection.
#[inline]
pub fn lcp_u8(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    let f = pick_lcp_bytes_impl();
    unsafe { f(text, p, q, max_ctx) }
}

/// Generic scalar LCP. Public so callers that already know they can
/// skip the SIMD dispatch (e.g. non-`Symbol` symbol types like
/// arbitrary `Eq` newtypes) can call this directly. Still
/// symbol-granularity; just doesn't go through the byte view.
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

/// Inspect this CPU's features and return the best [`LcpBytesFn`].
fn pick_lcp_bytes_impl() -> LcpBytesFn {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512BW gives us a 64-byte byte-compare returning a 64-bit
        // mask register directly — no movemask intrinsic, no extract.
        // Both `f` (foundation) and `bw` (byte/word ops, for the
        // `_mm512_cmpeq_epi8_mask` we use) are required.
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return lcp_bytes_avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return lcp_bytes_avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return lcp_bytes_neon;
        }
    }
    lcp_bytes_scalar
}

/// `unsafe fn`-typed scalar — wrapper around [`lcp_scalar`] for the
/// `u8` instantiation so all dispatch targets share the [`LcpBytesFn`]
/// signature.
unsafe fn lcp_bytes_scalar(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
    lcp_scalar(text, p, q, max_ctx)
}

/// AVX-512BW path: 64-byte vector compares; `_mm512_cmpeq_epi8_mask`
/// returns the per-byte equality mask straight in a 64-bit `__mmask64`
/// register (no movemask round-trip), and `(!mask).trailing_zeros()`
/// gives the first differing byte.
///
/// The function leads with a single 32-byte AVX2 step. This keeps the
/// short-LCP regime (random DNA, where every call typically resolves
/// in the first ≤16 bytes) at AVX2's per-call cost — a 64-byte load +
/// ZMM register usage on a call that exits inside the first 32 bytes
/// is wasted work. Once we've established the LCP exceeds 32 bytes we
/// switch to the 64-byte stride for the rest of the comparison, which
/// is the regime the upstream genome bench (and this function's
/// reason for existing) actually hits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn lcp_bytes_avx512(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
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
unsafe fn lcp_bytes_avx2(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
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
unsafe fn lcp_bytes_neon(text: &[u8], p: usize, q: usize, max_ctx: usize) -> usize {
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

    /// SIMD-vs-scalar agreement for `u16` text. The byte-view dispatch
    /// must return symbol-LCPs that match the scalar walk over `&[u16]`.
    /// Covers the case where the first differing byte lands inside a
    /// symbol whose previous bytes were equal (e.g. low byte of u16
    /// equal, high byte differs).
    #[test]
    fn simd_matches_scalar_on_u16() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1357);

        // Place a difference at every interesting byte boundary within
        // and across symbols; the symbol-LCP should equal byte_diff/2.
        let mut text = vec![0u16; 256];
        for byte_diff_at in [0usize, 1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 200] {
            text.iter_mut().for_each(|x| *x = 0xAAAA);
            // Flip one bit inside `text[byte_diff_at / 2]`, in either
            // the low or high byte depending on parity.
            let sym = byte_diff_at / 2;
            let mask = if byte_diff_at % 2 == 0 {
                0x00FF
            } else {
                0xFF00
            };
            text[sym] ^= mask & 0xAAAA; // toggle the bit pattern
            let got = lcp(&text, 0, 128, usize::MAX);
            // The first differing symbol is `byte_diff_at / 2`.
            assert_eq!(
                got, sym,
                "byte_diff_at={byte_diff_at}, expected symbol {sym}"
            );
        }

        // Random u16 texts: dispatched path must equal scalar.
        for &n in &[1usize, 16, 17, 100, 500] {
            let text: Vec<u16> = (0..n).map(|_| rng.random_range(0..16u16)).collect();
            for _ in 0..20 {
                let p = rng.random_range(0..n);
                let q = rng.random_range(0..n);
                let want = lcp_scalar(&text, p, q, usize::MAX);
                let got = lcp(&text, p, q, usize::MAX);
                assert_eq!(got, want, "u16 p={p} q={q} n={n}");
            }
        }
    }

    /// Same agreement check at `u32` granularity (4-byte symbols).
    #[test]
    fn simd_matches_scalar_on_u32() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x2468);

        for &n in &[1usize, 8, 9, 100, 500] {
            let text: Vec<u32> = (0..n).map(|_| rng.random_range(0..32u32)).collect();
            for _ in 0..20 {
                let p = rng.random_range(0..n);
                let q = rng.random_range(0..n);
                let want = lcp_scalar(&text, p, q, usize::MAX);
                let got = lcp(&text, p, q, usize::MAX);
                assert_eq!(got, want, "u32 p={p} q={q} n={n}");
            }
        }
    }

    /// 24-bit alphabet via `[u8; 3]`: every symbol-LCP must equal the
    /// scalar walk's. Exercises a symbol width that doesn't divide any
    /// SIMD chunk evenly.
    #[test]
    fn simd_matches_scalar_on_u8_3() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xACAC);

        for &n in &[1usize, 8, 22, 100, 333] {
            let text: Vec<[u8; 3]> = (0..n)
                .map(|_| {
                    [
                        rng.random_range(0..4u8),
                        rng.random_range(0..4u8),
                        rng.random_range(0..4u8),
                    ]
                })
                .collect();
            for _ in 0..20 {
                let p = rng.random_range(0..n);
                let q = rng.random_range(0..n);
                let want = lcp_scalar(&text, p, q, usize::MAX);
                let got = lcp(&text, p, q, usize::MAX);
                assert_eq!(got, want, "[u8;3] p={p} q={q} n={n}");
            }
        }
    }

    /// `u64` agreement — 8-byte symbols.
    #[test]
    fn simd_matches_scalar_on_u64() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFEED);

        for &n in &[1usize, 4, 5, 50, 250] {
            let text: Vec<u64> = (0..n).map(|_| rng.random_range(0..64u64)).collect();
            for _ in 0..20 {
                let p = rng.random_range(0..n);
                let q = rng.random_range(0..n);
                let want = lcp_scalar(&text, p, q, usize::MAX);
                let got = lcp(&text, p, q, usize::MAX);
                assert_eq!(got, want, "u64 p={p} q={q} n={n}");
            }
        }
    }
}
