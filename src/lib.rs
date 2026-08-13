//! Cache-friendly, parallel, sample-sort-based suffix array construction.
//!
//! This crate is a Rust port of [CaPS-SA] (Khan et al., WABI 2023), a parallel
//! and cache-friendly suffix-array constructor based on sample sort with
//! LCP-enhanced comparison.
//!
//! The crate is generic over the symbol type (`u8`, `u16`, …; any `Ord + Copy`)
//! and the index type (`u32`, `u64`; via the [`Index`] trait). It produces a
//! standard lexicographic suffix array. Callers who need a *generalized* suffix
//! array can supply a [`LimitProvider`] (for example [`SegmentedText`]) to stop
//! comparisons at string boundaries and define their boundary ordering. This
//! avoids enlarging the symbol alphabet with one distinct sentinel per input
//! string.
//!
//! Phase 1 of the port provides the **in-memory** algorithm; the external-memory
//! variant (disk-spilling buckets) is layered on top in a later phase.
//!
//! [CaPS-SA]: https://github.com/jamshed/CaPS-SA

mod ext_bucket;
mod ext_mem;
mod lcp;
mod lcp_memo;
mod limits;
mod radix;
mod sample_sort;

pub use ext_mem::{
    BuildError, ExtMemOpts, build_ext_mem, build_ext_mem_for_filter, build_ext_mem_for_filter_with,
    build_ext_mem_for_positions, build_ext_mem_for_positions_with, build_ext_mem_with,
    build_in_memory_sample_sort, build_in_memory_sample_sort_for_positions,
    build_in_memory_sample_sort_for_positions_with, build_in_memory_sample_sort_with,
    try_build_ext_mem, try_build_ext_mem_for_filter, try_build_ext_mem_for_filter_with,
    try_build_ext_mem_for_positions, try_build_ext_mem_for_positions_with, try_build_ext_mem_with,
    try_build_in_memory_sample_sort, try_build_in_memory_sample_sort_for_positions,
    try_build_in_memory_sample_sort_for_positions_with, try_build_in_memory_sample_sort_with,
};
pub use lcp::{LcpDispatch, Symbol, lcp, lcp_scalar, lcp_u8, suffix_cmp};
pub use lcp_memo::{GeometricMemoizationConfig, LcpMemoizationPolicy};
pub use limits::{LimitProvider, PlainText, SegmentedText};
pub use sample_sort::{
    Opts, build_in_memory, build_in_memory_for_positions, build_in_memory_for_positions_with,
    build_in_memory_for_positions_with_opts, build_in_memory_with, build_in_memory_with_opts,
};

/// Check that `sa` really is the suffix array of `text`, in `O(n)` time and
/// without re-running any construction algorithm.
///
/// Comparing a candidate against a second implementation only shows the two
/// agree; comparing adjacent suffixes directly is `O(n · lcp)` and becomes
/// unusable on the repetitive inputs that matter most. This instead uses the
/// standard fixpoint characterisation: let `rank` be the inverse of `sa`, and
/// define `f(p) = (text[p], rank[p + 1])`, with `rank[n]` taken as less than
/// every real rank. A permutation is the suffix array of `text` if and only if
/// `f` is strictly increasing along it, because suffix `p` precedes suffix `q`
/// exactly when `f(p) < f(q)`.
///
/// Returns `Err` with a description of the first violation found.
///
/// ```
/// let text = b"banana";
/// let sa: Vec<u32> = caps_sa::build_in_memory(text);
/// assert!(caps_sa::verify_sa(text, &sa).is_ok());
/// assert!(caps_sa::verify_sa(text, &[0u32, 1, 2, 3, 4, 5]).is_err());
/// ```
pub fn verify_sa<S, I>(text: &[S], sa: &[I]) -> Result<(), String>
where
    S: Ord,
    I: Index,
{
    let n = text.len();
    if sa.len() != n {
        return Err(format!("sa has {} entries, text has {n} symbols", sa.len()));
    }
    if n == 0 {
        return Ok(());
    }

    // Invert `sa`, checking along the way that it is a permutation of `0..n`.
    let mut rank = vec![usize::MAX; n];
    for (i, entry) in sa.iter().enumerate() {
        let p = entry.to_usize();
        if p >= n {
            return Err(format!("sa[{i}] = {p} is out of range for text length {n}"));
        }
        if rank[p] != usize::MAX {
            return Err(format!(
                "position {p} appears at sa[{}] and sa[{i}]",
                rank[p]
            ));
        }
        rank[p] = i;
    }

    // `None` stands for the end of the text, which sorts before every rank:
    // the shorter suffix is the smaller one.
    let successor =
        |p: usize| -> Option<usize> { if p + 1 < n { Some(rank[p + 1]) } else { None } };
    for i in 1..n {
        let a = sa[i - 1].to_usize();
        let b = sa[i].to_usize();
        let key_a = (&text[a], successor(a));
        let key_b = (&text[b], successor(b));
        if key_a >= key_b {
            return Err(format!(
                "suffixes out of order at sa[{}] = {a} and sa[{i}] = {b}",
                i - 1,
            ));
        }
    }
    Ok(())
}

/// Trait implemented by integer types usable as suffix array indices.
///
/// Provided for `u32`, `u64`, and `usize`. Callers pick the narrowest type
/// large enough to address their text.
pub trait Index:
    Copy
    + Eq
    + Ord
    + Send
    + Sync
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
{
    /// Convert from `usize`.
    ///
    /// Current primitive implementations use Rust's `as` casts and
    /// therefore truncate if the value does not fit. Public constructors
    /// dispatch to an index width large enough for their generated
    /// positions; callers that invoke generic internals directly must
    /// choose an `I` that can represent every position they pass.
    fn from_usize(v: usize) -> Self;
    /// Convert to `usize`. Lossless for `u32`/`u64`/`usize` on 64-bit targets.
    fn to_usize(self) -> usize;
    /// The zero value.
    fn zero() -> Self;
}

macro_rules! impl_index {
    ($t:ty) => {
        impl Index for $t {
            #[inline]
            fn from_usize(v: usize) -> Self {
                v as $t
            }
            #[inline]
            fn to_usize(self) -> usize {
                self as usize
            }
            #[inline]
            fn zero() -> Self {
                0
            }
        }
    };
}

impl_index!(u32);
impl_index!(u64);
impl_index!(usize);
