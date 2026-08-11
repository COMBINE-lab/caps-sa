//! Radix-seeded prefix doubling for the plain in-memory suffix array.
//!
//! The LCP-enhanced merge sort in [`crate::sample_sort`] is the CaPS-SA
//! kernel and stays the general path: it is the only one that honours a
//! [`LimitProvider`][crate::limits::LimitProvider], a finite `max_context`,
//! and symbol types wider than a byte, and it is the only one that produces
//! an LCP array (which the external-memory path needs).
//!
//! But for the single most common request — the standard lexicographic
//! suffix array of a byte text, with no segmentation and no context bound —
//! that kernel is doing far more work than the problem requires, in two
//! distinct ways that the benchmarks separate cleanly:
//!
//! * **Step count.** The merge sort performs `n log n` merge steps, and a
//!   large majority of them are resolved by an actual symbol comparison at a
//!   random text address. On 80 MB of N-free DNA that is ~2.1e9 steps at
//!   ~13 ns each.
//! * **Scan length.** Every leaf merge starts with `m = 0`, so comparing two
//!   suffixes that share a long prefix costs a scan proportional to that
//!   prefix. Genome assemblies contain megabyte-scale runs of `N` (and the
//!   period-61 `N`-then-newline pattern of wrapped FASTA), where a single
//!   comparison scans millions of bytes. On a 47.5 MB chr21 FASTA this
//!   pushes the cost per merge step from 13 ns to 222 ns — a 16x penalty
//!   that is entirely scan time.
//!
//! This module attacks both. It sorts by a packed fixed-depth key first
//! (killing the step count), then resolves the remainder by **prefix
//! doubling** on ranks (killing the scan length: after the seed, no
//! comparison ever reads the text again, so a megabyte-long run of `N` costs
//! exactly as much as random DNA).
//!
//! ## The algorithm
//!
//! 1. **Pack.** Rank the bytes that actually occur onto a dense code range,
//!    then choose the smallest field width in `{1, 2, 4, 8}` bits that holds
//!    the alphabet, so `k = 64 / bits` symbols fit in one `u64` key. Ranking
//!    matters: raw FASTA uses six symbols but its largest byte is `'T'` (84),
//!    so packing raw bytes would force 8-bit fields and 8 symbols per key,
//!    against 16 after ranking. DNA over `{0,1,2,3}` gets 2-bit fields and
//!    **32 symbols per key**.
//! 2. **Seed.** Sort `(key, position)`. This is a full sort of the suffixes
//!    by their first `k` symbols, and it touches the text only in one
//!    sequential pass.
//! 3. **Double.** Suffixes still tied after depth `d` are ordered by the pair
//!    `(rank_d(p), rank_d(p + d))`, which resolves them to depth `2d`. Repeat
//!    until every group is a singleton. Each round reads only the rank array.
//!
//! ## Ordering convention
//!
//! Keys are big-endian in the field sense (the first symbol occupies the most
//! significant field) and short suffixes are zero-padded. Since `0` is the
//! minimum of `u8`, a padding field can never exceed a real symbol's field,
//! so a padded key compares less-or-equal to any key it shares a prefix with.
//! That is exactly the crate's "shorter suffix is smaller" convention. A real
//! `0` symbol is indistinguishable from padding *in the key*, which can only
//! make two suffixes tie — never invert them — and ties are resolved by the
//! doubling rounds, which use the true remaining length via the end-of-text
//! sentinel. So `A = 0` DNA encodings and STAR's `0..5` codes are both safe.

use crate::Index;
use crate::ext_mem::profile_log;
use crate::lcp::Symbol;
use crate::limits::LimitProvider;
use crate::runs::Cmp;
use crate::sample_sort;
use rayon::prelude::*;
use std::time::Instant;

/// An order-preserving remap of the bytes that actually occur in a text onto
/// a dense code range, plus the resulting key geometry.
///
/// The field width is driven by how many *distinct* symbols a text uses, not
/// by the largest byte value in it, and the difference is not academic. A raw
/// FASTA uses six symbols, but the largest is `'T'` (84), so packing raw bytes
/// forces 8-bit fields and fits only 8 symbols per key. Ranking those six
/// bytes to `0..6` gives 4-bit fields and 16 symbols per key, which halves the
/// number of doubling rounds needed downstream. The DNA-coded input is already
/// dense, so it is unaffected.
///
/// The map is monotone by construction, since codes are assigned in ascending
/// byte order. That is what keeps a packed key order-preserving: `key_a <
/// key_b` still implies `suffix_a < suffix_b`, and the zero-padding argument
/// carries over because code `0` remains the minimum.
pub(crate) struct Packer {
    /// Byte to dense code. Bytes absent from the text map to `0`; they never
    /// appear in a key.
    code: [u8; 256],
    /// Bits per packed field.
    bits: u32,
    /// Symbols per `u64` key.
    k: usize,
}

impl Packer {
    /// Build the map for `text`.
    fn new(text: &[u8]) -> Self {
        // Which bytes occur? One parallel pass, folded into a 256-bit set.
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
        for (b, &seen) in present.iter().enumerate() {
            if seen {
                code[b] = next as u8;
                next += 1;
            }
        }
        // `next` is the alphabet size; the largest code is `next - 1`.
        let bits: u32 = match next.saturating_sub(1) {
            0..=1 => 1,
            2..=3 => 2,
            4..=15 => 4,
            _ => 8,
        };
        Self {
            code,
            bits,
            k: 64 / bits as usize,
        }
    }

    #[inline]
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// Pack the `k` symbols at `text[p..]` into one order-preserving `u64`,
    /// zero-padding past the end of the text.
    #[inline]
    pub(crate) fn key_at(&self, text: &[u8], p: usize) -> u64 {
        let end = (p + self.k).min(text.len());
        let mut key: u64 = 0;
        for &s in &text[p..end] {
            key = (key << self.bits) | self.code[s as usize] as u64;
        }
        // Shift the packed prefix up so the missing trailing fields read zero.
        key << (self.bits as usize * (self.k - (end - p)))
    }
}

/// The alphabet map for `text`, or `None` when a packed key cannot represent
/// this text's order.
///
/// Computed once per build and handed to [`seed_subarray`], which would
/// otherwise re-scan the whole text for every subarray.
pub(crate) fn seed_params<S: Symbol>(text: &[S]) -> Option<Packer> {
    if size_of::<S>() != 1 {
        return None;
    }
    // SAFETY: `S` is one byte wide, so a byte view over the same memory is
    // valid for reads of the same length.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };
    Some(Packer::new(bytes))
}

/// Sort `sa` into suffix order and fill `lcp`, using a packed fixed-depth key
/// so that most of the ordering costs no text access at all.
///
/// This is the external-memory and sample-sort counterpart to [`build_sa`].
/// Those paths cannot use prefix doubling, which needs a rank for every
/// position in the text and would break the memory bound they exist to
/// provide. But they can still avoid the part of the merge kernel that hurts
/// most: sorting a subarray from singletons, where every leaf merge starts at
/// `m = 0` and compares two suffixes by scanning the text.
///
/// Sorting by the packed key resolves the first `k` symbols with no text
/// access (32 symbols for DNA), and hands back the LCP for adjacent entries
/// for free from `(key_a ^ key_b).leading_zeros()`. Only suffixes that agree
/// through all `k` symbols reach the merge kernel, on the small slice they
/// occupy.
///
/// Returns `false` without touching anything when the comparator is not plain
/// lexicographic, so the caller falls back to a plain `merge_sort`.
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
    cmp: Cmp<'_>,
) -> bool {
    let Some(packer) = packer else {
        return false;
    };
    let (bits, k) = (packer.bits(), packer.k());
    if max_ctx != usize::MAX || lp.plain_lex_len() != Some(text.len()) {
        return false;
    }
    let len = sa.len();
    if len < 2 {
        if len == 1 {
            lcp[0] = I::zero();
        }
        return true;
    }
    // SAFETY: `params` is `Some` only when `S` is one byte wide.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };

    // Order by (key, visible length), the same comparator `build_sa` seeds
    // with and for the same reason: zero padding makes a short suffix share a
    // key with any suffix continuing in zeros, and `0` is a real symbol.
    let n = bytes.len();
    let visible = |p: usize| -> usize { (n - p).min(k) };
    let mut keyed: Vec<(u64, u32, I)> = sa
        .iter()
        .map(|&p| {
            let p = p.to_usize();
            (packer.key_at(bytes, p), visible(p) as u32, p_as(p))
        })
        .collect();
    keyed.sort_unstable();
    for (slot, e) in sa.iter_mut().zip(keyed.iter()) {
        *slot = I::from_usize(e.2.to_usize());
    }

    // Walk equal-key runs. Between runs the LCP falls straight out of the key
    // difference; inside one it needs the merge kernel.
    let mut i = 0usize;
    while i < len {
        let mut j = i + 1;
        while j < len && (keyed[j].0, keyed[j].1) == (keyed[i].0, keyed[i].1) {
            j += 1;
        }
        if j - i > 1 {
            sample_sort::merge_sort(
                text,
                lp,
                &mut sa[i..j],
                &mut sa_w[i..j],
                &mut lcp[i..j],
                &mut lcp_w[i..j],
                max_ctx,
                cmp,
            );
        } else {
            lcp[i] = I::zero();
        }
        // Boundary entry: LCP against the last element of the previous run.
        if i > 0 {
            let a = sa[i - 1].to_usize();
            let b = sa[i].to_usize();
            let xor = keyed[i - 1].0 ^ keyed[i].0;
            debug_assert_ne!(xor, 0, "distinct runs must differ in the key");
            // `leading_zeros / bits` counts whole matching fields. Cap by both
            // suffixes' lengths: padding can agree with a real `0` symbol past
            // the end of the shorter one.
            let shared = (xor.leading_zeros() / bits) as usize;
            lcp[i] = I::from_usize(shared.min(lp.lim_at(a)).min(lp.lim_at(b)));
        }
        i = j;
    }
    lcp[0] = I::zero();
    true
}

/// Round-trip a position through the index type used in the seed vector.
#[inline]
fn p_as<I: Index>(p: usize) -> I {
    I::from_usize(p)
}

/// Bits of the key used for the MSD counting-sort pass, and the resulting
/// bucket count.
///
/// 2048 buckets keeps the write-combining state at 2048 × 2 streams × 128 B
/// ≈ 512 KB, which stays inside a core's private cache. Going to 16 bits
/// would need 16 MB of open write lines and thrashes the TLB instead.
const RADIX_BITS: u32 = 11;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Sort every suffix position by `(packed key, visible length)`, returning the
/// sorted keys alongside the sorted positions.
///
/// This is an MSD counting sort rather than a comparison sort, for three
/// reasons that all matter at genome scale:
///
/// * The source is never materialised. Keys are recomputed from `text` in
///   both the histogram and the scatter pass, which is a sequential read of
///   the text instead of a random read of an `n`-element key array.
/// * Peak memory is the two destination buffers only, 12 bytes per position
///   with `I = u32`, against the 16 a `(u64, u32, I)` record costs.
/// * The top-level partition is a counting pass, so it parallelises evenly.
///   A parallel comparison sort's first partitioning steps are close to
///   serial, which is exactly where a `n log n` sort loses on many cores.
fn seed_sort<I: Index>(
    text: &[u8],
    packer: &Packer,
    visible_len: &(dyn Fn(usize) -> usize + Sync),
) -> (Vec<u64>, Vec<I>) {
    let n = text.len();
    let bucket_of = |key: u64| -> usize { (key >> (64 - RADIX_BITS)) as usize };

    // Chunk the position range so each worker builds a private histogram.
    let n_chunks = (rayon::current_num_threads() * 4).clamp(1, 1024);
    let chunk_len = n.div_ceil(n_chunks);
    let bounds: Vec<(usize, usize)> = (0..n)
        .step_by(chunk_len)
        .map(|s| (s, (s + chunk_len).min(n)))
        .collect();

    // Pass 1: per-chunk histograms over the top `RADIX_BITS` of each key.
    let histograms: Vec<Vec<u32>> = bounds
        .par_iter()
        .map(|&(start, end)| {
            let mut counts = vec![0u32; RADIX_BUCKETS];
            for p in start..end {
                counts[bucket_of(packer.key_at(text, p))] += 1;
            }
            counts
        })
        .collect();

    // Exclusive prefix sum, bucket-major then chunk-minor, so every (chunk,
    // bucket) pair gets a disjoint destination range and the buckets come out
    // in ascending key order.
    let mut offsets = vec![0usize; bounds.len() * RADIX_BUCKETS];
    let mut bucket_start = vec![0usize; RADIX_BUCKETS + 1];
    {
        let mut running = 0usize;
        for b in 0..RADIX_BUCKETS {
            bucket_start[b] = running;
            for (c, hist) in histograms.iter().enumerate() {
                offsets[c * RADIX_BUCKETS + b] = running;
                running += hist[b] as usize;
            }
        }
        bucket_start[RADIX_BUCKETS] = running;
        debug_assert_eq!(running, n);
    }

    // Pass 2: scatter. Each chunk owns a disjoint slice of every bucket, so
    // the writes never collide even though they are not contiguous.
    let mut keys: Vec<u64> = vec![0; n];
    let mut sa: Vec<I> = vec![I::zero(); n];
    {
        let key_out = Scatter::new(&mut keys);
        let sa_out = Scatter::new(&mut sa);
        bounds
            .par_iter()
            .enumerate()
            .for_each(|(c, &(start, end))| {
                let mut cursor: Vec<usize> =
                    offsets[c * RADIX_BUCKETS..(c + 1) * RADIX_BUCKETS].to_vec();
                for p in start..end {
                    let key = packer.key_at(text, p);
                    let slot = &mut cursor[bucket_of(key)];
                    // SAFETY: the prefix sum gives this (chunk, bucket) pair a
                    // range of exactly its own histogram count, and the cursor
                    // never leaves it, so no other thread writes this index.
                    unsafe {
                        key_out.set(*slot, key);
                        sa_out.set(*slot, I::from_usize(p));
                    }
                    *slot += 1;
                }
            });
    }

    // Pass 3: order within each bucket. Buckets share their top `RADIX_BITS`,
    // so what remains is the low bits of the key and then the visible-length
    // tie-break. Buckets are contiguous and independent.
    let mut rest: &mut [u64] = &mut keys;
    let mut rest_sa: &mut [I] = &mut sa;
    let mut slices: Vec<(&mut [u64], &mut [I])> = Vec::with_capacity(RADIX_BUCKETS);
    for b in 0..RADIX_BUCKETS {
        let len = bucket_start[b + 1] - bucket_start[b];
        let (kb, kt) = rest.split_at_mut(len);
        let (sb, st) = rest_sa.split_at_mut(len);
        slices.push((kb, sb));
        rest = kt;
        rest_sa = st;
    }
    slices.into_par_iter().for_each(|(kb, sb)| {
        if kb.len() < 2 {
            return;
        }
        let mut pairs: Vec<(u64, I)> = kb.iter().copied().zip(sb.iter().copied()).collect();
        pairs.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| visible_len(a.1.to_usize()).cmp(&visible_len(b.1.to_usize())))
        });
        for (i, &(key, pos)) in pairs.iter().enumerate() {
            kb[i] = key;
            sb[i] = pos;
        }
    });

    (keys, sa)
}

/// Build the standard lexicographic suffix array of `text` by radix-seeded
/// prefix doubling.
///
/// The caller is responsible for the guards: `text` must be the whole,
/// non-segmented text, the comparator must be plain lexicographic with
/// shorter-is-smaller, and there must be no `max_context` bound. See
/// [`crate::sample_sort::build_in_memory_with`] for where those are checked.
pub(crate) fn build_sa<I: Index>(text: &[u8]) -> Vec<I> {
    let n = text.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![I::zero()];
    }

    let t0 = Instant::now();
    let packer = Packer::new(text);
    let k = packer.k();

    // ---- Seed: sort by the first `k` symbols, then by visible length. ----
    //
    // The second component is `min(n - p, k)`, and it is load-bearing rather
    // than cosmetic. Zero-padding makes a suffix shorter than `k` share a key
    // with any suffix whose symbols continue with zeros, and `0` is a real
    // symbol in every DNA encoding. Ordering those by visible length puts the
    // proper prefix first, which is the shorter-is-smaller convention. For
    // suffixes at least `k` long the component is `k` for all of them, so it
    // never separates suffixes that the doubling rounds still need to see as
    // tied. Without it, `[0, 0]` leaves positions 0 and 1 permanently tied
    // and the doubling loop cannot terminate.
    // Only the last `k - 1` positions can have a visible length below `k`, so
    // the tie-break is a function of the position alone and never has to be
    // stored alongside the key.
    let visible_len = |p: usize| -> usize { (n - p).min(k) };

    profile_log(&format!(
        "radix setup     {:.3}s",
        t0.elapsed().as_secs_f64()
    ));
    let t1 = Instant::now();
    let (keys, mut sa) = seed_sort::<I>(text, &packer, &visible_len);
    profile_log(&format!(
        "radix seed sort {:.3}s",
        t1.elapsed().as_secs_f64()
    ));
    let t2 = Instant::now();

    // Two seeded entries tie exactly when key and visible length both match.
    let seed_eq = |a: usize, b: usize| -> bool {
        keys[a] == keys[b] && visible_len(sa[a].to_usize()) == visible_len(sa[b].to_usize())
    };

    // `rank[p]` is the index in `sa` of the first element of `p`'s group, so
    // two suffixes tie at the current depth exactly when their ranks match,
    // and rank order is the current partial order.
    let mut rank: Vec<I> = vec![I::zero(); n];
    // Non-singleton `sa` ranges, the only ones any later round touches.
    //
    // Each index decides for itself whether it starts a group; the index that
    // does then owns the whole group, walks it to find the end, and writes its
    // members' ranks. Every group has exactly one owner and groups partition
    // `0..n`, so the scattered writes never collide. `collect` on an indexed
    // parallel iterator preserves order, so `groups` comes out sorted.
    let ranks = Scatter::new(&mut rank);
    let groups: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .filter_map(|h| {
            if h > 0 && seed_eq(h - 1, h) {
                return None;
            }
            let mut e = h + 1;
            while e < n && seed_eq(e, h) {
                e += 1;
            }
            let g = I::from_usize(h);
            for entry in &sa[h..e] {
                // SAFETY: `sa` is a permutation of `0..n`, and this thread
                // owns the whole group `h..e`, so `entry` is a distinct index
                // no other thread writes.
                unsafe { ranks.set(entry.to_usize(), g) };
            }
            (e - h > 1).then_some((h, e))
        })
        .collect();
    let mut groups = groups;
    drop(keys);
    profile_log(&format!(
        "radix grouping  {:.3}s",
        t2.elapsed().as_secs_f64()
    ));
    let t3 = Instant::now();

    // ---- Double: (rank_d(p), rank_d(p + d)) resolves to depth 2d. ----
    let mut depth = k;
    // Scratch for the new rank of each `sa` slot, so a round's reads of
    // `rank` never observe that same round's writes.
    let mut next_rank: Vec<I> = vec![I::zero(); n];

    while !groups.is_empty() {
        // Phase A: sort each tied group by the successor rank, and record the
        // ranks it should get. Groups are disjoint `sa` ranges, so this is
        // data-parallel with no synchronisation.
        let sub: Vec<Vec<(usize, usize)>> = split_disjoint(&mut sa, &mut next_rank, &groups)
            .into_par_iter()
            .zip(groups.par_iter())
            .map(|((sa_g, nr_g), &(start, _))| {
                let succ = |p: usize| -> u64 {
                    // End-of-text sorts first: the shorter suffix is smaller.
                    match p.checked_add(depth) {
                        Some(q) if q < n => rank[q].to_usize() as u64 + 1,
                        _ => 0,
                    }
                };
                // Materialise the successor ranks once. `sort_unstable_by_key`
                // re-evaluates its key function O(len log len) times, and each
                // evaluation is a random probe into `rank`; paying for it once
                // per element turns the sort's memory traffic sequential.
                let mut keyed: Vec<(u64, I)> =
                    sa_g.iter().map(|&e| (succ(e.to_usize()), e)).collect();
                keyed.sort_unstable();

                let mut fresh = Vec::new();
                let mut i = 0;
                while i < keyed.len() {
                    let key = keyed[i].0;
                    let mut j = i + 1;
                    while j < keyed.len() && keyed[j].0 == key {
                        j += 1;
                    }
                    let g = I::from_usize(start + i);
                    for slot in &mut nr_g[i..j] {
                        *slot = g;
                    }
                    if j - i > 1 {
                        fresh.push((start + i, start + j));
                    }
                    i = j;
                }
                for (slot, &(_, e)) in sa_g.iter_mut().zip(keyed.iter()) {
                    *slot = e;
                }
                fresh
            })
            .collect();

        // Phase B: publish the new ranks, now that every read is done.
        // Groups are disjoint and `sa` is a permutation, so each `rank` slot
        // is written by exactly one group.
        let ranks = Scatter::new(&mut rank);
        groups.par_iter().for_each(|&(start, end)| {
            for i in start..end {
                // SAFETY: `sa[start..end]` are distinct positions owned solely
                // by this group, and the groups partition their index range.
                unsafe { ranks.set(sa[i].to_usize(), next_rank[i]) };
            }
        });

        let before: usize = groups.iter().map(|&(s, e)| e - s).sum();
        groups = sub.into_iter().flatten().collect();
        let after: usize = groups.iter().map(|&(s, e)| e - s).sum();

        // A doubling round can only ever refine, so `after <= before`. If a
        // round refines nothing at all the text has a run longer than the
        // whole remaining depth budget; doubling still terminates because
        // `depth` grows geometrically and every suffix eventually runs off
        // the end of the text, which the sentinel orders. Guard against
        // overflow rather than against non-progress.
        debug_assert!(after <= before);
        match depth.checked_mul(2) {
            Some(d) if d <= n.saturating_mul(2) => depth = d,
            _ => {
                debug_assert!(groups.is_empty(), "doubling exhausted with ties left");
                break;
            }
        }
    }

    profile_log(&format!(
        "radix doubling  {:.3}s",
        t3.elapsed().as_secs_f64()
    ));
    sa
}

/// Write access to disjoint slots of one slice from several rayon threads.
///
/// Both users here scatter through a permutation: the target index is
/// `sa[i]`, not `i`, so the writes cannot be expressed as disjoint sub-slices
/// and `split_at_mut` does not apply. What makes them safe is that `sa` is a
/// permutation and the ranges being processed partition its index space, so
/// every slot is written exactly once across all threads.
struct Scatter<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: `Scatter` hands out writes only through `set`, whose contract is
// that no two calls target the same index. Under that contract there is no
// aliasing between threads, so the pointer is safe to share.
unsafe impl<T: Send> Send for Scatter<T> {}
unsafe impl<T: Send> Sync for Scatter<T> {}

impl<T> Scatter<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Write `value` at `index`.
    ///
    /// # Safety
    ///
    /// No two concurrent calls may pass the same `index`, and the borrow the
    /// `Scatter` was built from must still be live.
    #[inline]
    unsafe fn set(&self, index: usize, value: T) {
        debug_assert!(index < self.len);
        unsafe { self.ptr.add(index).write(value) };
    }
}

/// Borrow each `(start, end)` range of `sa` and `next_rank` mutably and
/// simultaneously. The ranges come from a scan of `sa` so they are sorted and
/// non-overlapping, which is what makes the repeated `split_at_mut` sound.
fn split_disjoint<'a, I: Index>(
    sa: &'a mut [I],
    next_rank: &'a mut [I],
    groups: &[(usize, usize)],
) -> Vec<(&'a mut [I], &'a mut [I])> {
    let mut out = Vec::with_capacity(groups.len());
    let mut sa_rest = sa;
    let mut nr_rest = next_rank;
    let mut consumed = 0usize;
    for &(start, end) in groups {
        debug_assert!(
            start >= consumed,
            "group ranges must be sorted and disjoint"
        );
        let (_, sa_tail) = sa_rest.split_at_mut(start - consumed);
        let (_, nr_tail) = nr_rest.split_at_mut(start - consumed);
        let (sa_g, sa_tail) = sa_tail.split_at_mut(end - start);
        let (nr_g, nr_tail) = nr_tail.split_at_mut(end - start);
        out.push((sa_g, nr_g));
        sa_rest = sa_tail;
        nr_rest = nr_tail;
        consumed = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(text: &[u8]) -> Vec<u32> {
        let mut sa: Vec<u32> = (0..text.len() as u32).collect();
        sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        sa
    }

    fn check(text: &[u8]) {
        let got: Vec<u32> = build_sa(text);
        assert_eq!(got, brute(text), "mismatch on {text:?}");
    }

    #[test]
    fn fixtures() {
        check(b"");
        check(b"a");
        check(b"banana");
        check(b"mississippi");
        check(b"abracadabra");
    }

    /// The field width follows the number of *distinct* symbols, not the
    /// largest byte value. Raw FASTA is the case that matters: six symbols
    /// whose largest is `'T'` (84) would force 8-bit fields without ranking,
    /// fitting only 8 symbols per key instead of 16.
    #[test]
    fn packer_width_follows_alphabet_size_not_byte_value() {
        let two = Packer::new(b"abababab");
        assert_eq!((two.bits(), two.k()), (1, 64));
        let four = Packer::new(&[0u8, 1, 2, 3, 3, 2, 1, 0]);
        assert_eq!((four.bits(), four.k()), (2, 32));

        let mut fasta: Vec<u8> = b"ACGTN".to_vec();
        fasta.push(b'\n');
        let f = Packer::new(&fasta);
        assert_eq!((f.bits(), f.k()), (4, 16), "6 symbols should pack 4 bits");

        let dense: Vec<u8> = (0..=255u8).collect();
        let d = Packer::new(&dense);
        assert_eq!((d.bits(), d.k()), (8, 8));
    }

    /// The remap must be monotone, or a packed key would stop being
    /// order-preserving and the whole seed would be wrong.
    #[test]
    fn packer_remap_is_monotone() {
        let text: Vec<u8> = b"TGCAN\nZq".to_vec();
        let p = Packer::new(&text);
        let mut present: Vec<u8> = text.clone();
        present.sort_unstable();
        present.dedup();
        for w in present.windows(2) {
            assert!(
                p.code[w[0] as usize] < p.code[w[1] as usize],
                "codes must follow byte order: {:?}",
                w
            );
        }
    }

    /// Texts whose symbols include a real `0`, so padding and a genuine
    /// minimum symbol are indistinguishable in the packed key. This is the
    /// case DNA encodings hit (`A = 0`) and the one the ordering argument in
    /// the module docs turns on.
    #[test]
    fn real_zero_symbol_is_not_confused_with_padding() {
        check(&[0]);
        check(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        check(&[1, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        check(&[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        check(&[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        let mut t: Vec<u8> = (0..200).map(|i| (i % 4) as u8).collect();
        t.extend(std::iter::repeat_n(0u8, 100));
        check(&t);
    }

    /// Every text of length <= 10 over a binary alphabet, plus every text of
    /// length <= 6 over a ternary one. Total coverage of the padding and
    /// end-of-text logic at the sizes where exhaustive checking is free.
    #[test]
    fn exhaustive_small_alphabets() {
        for n in 0..=10u32 {
            for mask in 0..(1u32 << n) {
                let t: Vec<u8> = (0..n).map(|i| ((mask >> i) & 1) as u8).collect();
                check(&t);
            }
        }
        for n in 0..=6u32 {
            let total = 3u32.pow(n);
            for mut code in 0..total {
                let mut t = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    t.push((code % 3) as u8);
                    code /= 3;
                }
                check(&t);
            }
        }
    }

    #[test]
    fn random_across_alphabet_widths() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xADD1);
        for &sigma in &[2u8, 3, 4, 6, 16, 17, 255] {
            for &n in &[
                2usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 20_000,
            ] {
                let t: Vec<u8> = (0..n).map(|_| rng.random_range(0..sigma)).collect();
                check(&t);
            }
        }
    }

    /// Long runs and periodic text are the inputs that make the merge kernel
    /// quadratic. Doubling must handle them and must terminate.
    #[test]
    fn long_runs_and_periodic_text() {
        check(&vec![0u8; 5000]);
        check(&vec![7u8; 5000]);
        check(&(0..5000).map(|i| (i % 2) as u8).collect::<Vec<u8>>());
        check(&(0..5000).map(|i| (i % 61) as u8).collect::<Vec<u8>>());
        // A long run flanked by noise: the shape of a poly-N genome block.
        let mut t: Vec<u8> = (0..500).map(|i| (i % 4) as u8).collect();
        t.extend(std::iter::repeat_n(4u8, 4000));
        t.extend((0..500).map(|i| (i % 4) as u8));
        check(&t);
        // Wrapped-FASTA shape: 60 `N`s then a newline, repeated.
        let mut fasta: Vec<u8> = Vec::new();
        for _ in 0..100 {
            fasta.extend(std::iter::repeat_n(b'N', 60));
            fasta.push(b'\n');
        }
        check(&fasta);
    }

    #[test]
    fn u64_index_matches_u32_index() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD0AB);
        let t: Vec<u8> = (0..5000).map(|_| rng.random_range(0..4u8)).collect();
        let a: Vec<u32> = build_sa(&t);
        let b: Vec<u64> = build_sa(&t);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(&x, &y)| x as u64 == y));
    }
}
