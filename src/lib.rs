//! Cache-friendly, parallel, sample-sort-based suffix array construction.
//!
//! This crate is a Rust port of [CaPS-SA] (Khan et al., WABI 2023), a parallel
//! and cache-friendly suffix-array constructor based on sample sort with
//! LCP-enhanced comparison.
//!
//! The crate is generic over the symbol type (`u8`, `u16`, …; any `Ord + Copy`)
//! and the index type (`u32`, `u64`; via the [`Index`] trait). It produces a
//! standard lexicographic suffix array. Callers who need a *generalized* suffix
//! array (multiple strings, sentinel-terminated) should rewrite their text with
//! distinct sentinels before invoking — the resulting standard SA over the
//! transformed text is then the generalized SA they want.
//!
//! Phase 1 of the port provides the **in-memory** algorithm; the external-memory
//! variant (disk-spilling buckets) is layered on top in a later phase.
//!
//! [CaPS-SA]: https://github.com/jamshed/CaPS-SA

mod ext_bucket;
mod lcp;
mod sample_sort;

pub use lcp::{lcp, suffix_cmp};
pub use sample_sort::{Opts, build_in_memory, build_in_memory_with_opts};

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
    /// Convert from `usize`. Panics if the value does not fit.
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
