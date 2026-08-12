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
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// The text with every byte replaced by its code, when the identity map
    /// does not already do that. Materialising it once removes a dependent
    /// table load from the packing loop, which is otherwise the chain that
    /// sets the cost of building a key. `None` when the text is already dense
    /// (a `0..3` DNA encoding, say), so the common pre-coded input pays no
    /// extra memory.
    ranked: Option<Vec<u8>>,
    /// Bits per packed field.
    bits: u32,
    /// Symbols per `u64` key.
    k: usize,
}

impl Packer {
    /// Build the map for `text`.
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
        let bits: u32 = match next.saturating_sub(1) {
            0..=1 => 1,
            2..=3 => 2,
            4..=15 => 4,
            _ => 8,
        };
        // An identity map, or 8-bit fields where the code never changes the
        // packed value's order, both let the original text be read directly.
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
        }
    }

    /// Bits per packed field. Used by the key-geometry tests.
    #[cfg(test)]
    #[inline]
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// Gather the low `bits` of each of 8 ranked bytes into one contiguous
    /// field, most-significant byte first.
    ///
    /// A binary-tree SWAR shuffle: each step folds neighbouring fields
    /// together and halves the stride, so eight symbols cost three
    /// shift-or-mask pairs instead of eight dependent shift-or steps. The
    /// input is a big-endian load, so the text's first byte lands in the
    /// result's most significant field, which is the order the key needs.
    #[inline(always)]
    fn gather8(v: u64, bits: u32) -> u64 {
        match bits {
            1 => {
                let mut x = v & 0x0101_0101_0101_0101;
                x = (x | (x >> 7)) & 0x0003_0003_0003_0003;
                x = (x | (x >> 14)) & 0x0000_000F_0000_000F;
                (x | (x >> 28)) & 0xFF
            }
            2 => {
                let mut x = v & 0x0303_0303_0303_0303;
                x = (x | (x >> 6)) & 0x000F_000F_000F_000F;
                x = (x | (x >> 12)) & 0x0000_00FF_0000_00FF;
                (x | (x >> 24)) & 0xFFFF
            }
            4 => {
                let mut x = v & 0x0F0F_0F0F_0F0F_0F0F;
                x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
                x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
                (x | (x >> 16)) & 0xFFFF_FFFF
            }
            _ => v,
        }
    }

    /// Pack the `k` symbols at `text[p..]` into one order-preserving `u64`,
    /// zero-padding past the end of the text.
    #[inline]
    pub(crate) fn key_at(&self, text: &[u8], p: usize) -> u64 {
        let src = self.ranked.as_deref().unwrap_or(text);
        let n = src.len();

        // Fast path: a whole key's worth of symbols is available, so it is
        // `k / 8` big-endian loads and their gathers, with no bounds fuss.
        if p + self.k <= n {
            return self
                .fold(|i| u64::from_be_bytes(src[p + 8 * i..p + 8 * i + 8].try_into().unwrap()));
        }

        // Tail: fewer than `k` symbols remain. Pad with zero codes, which are
        // the alphabet's minimum, matching shorter-is-smaller.
        let mut buf = [0u8; 64];
        buf[..n - p].copy_from_slice(&src[p..n]);
        self.fold(|i| u64::from_be_bytes(buf[8 * i..8 * i + 8].try_into().unwrap()))
    }

    /// Concatenate the gathers of the `k / 8` words produced by `word`.
    ///
    /// The first group is assigned rather than shifted in. With 8-bit fields
    /// there is exactly one group and `8 * bits` is 64, which is not a legal
    /// shift distance for `u64`; release builds mask it to 0 and happen to
    /// give the right answer, debug builds panic. Assigning avoids relying on
    /// either behaviour.
    #[inline(always)]
    fn fold(&self, word: impl Fn(usize) -> u64) -> u64 {
        let shift = 8 * self.bits;
        let mut key = 0u64;
        for i in 0..self.k / 8 {
            let g = Self::gather8(word(i), self.bits);
            key = if i == 0 { g } else { (key << shift) | g };
        }
        key
    }
}

/// The LCP array of `sa`, computed from the suffix array itself in `O(n)`.
///
/// Kasai's algorithm. `lcp[i]` is the number of symbols
/// `text[sa[i - 1]..]` and `text[sa[i]..]` share, and `lcp[0]` is `0`.
///
/// This exists because prefix doubling answers comparisons from ranks and so
/// never produces the LCP array the merge kernel yields as a byproduct, which
/// is the structural reason the external-memory path could not be routed
/// through it. Deriving it afterwards costs one linear pass.
///
/// The pass is sequential and looks random-access, but it is not quadratic:
/// `h` falls by at most one per position and rises only while matching, so the
/// total number of symbol comparisons is at most `2n`. That bound holds
/// regardless of how repetitive the text is, which is the property the
/// scanning merge lacks.
pub(crate) fn kasai_lcp<I: Index>(text: &[u8], sa: &[I]) -> Vec<I> {
    let n = sa.len();
    let mut lcp = vec![I::zero(); n];
    if n == 0 {
        return lcp;
    }
    let mut rank = vec![0usize; n];
    for (i, entry) in sa.iter().enumerate() {
        rank[entry.to_usize()] = i;
    }
    let mut h = 0usize;
    for p in 0..n {
        let i = rank[p];
        if i == 0 {
            h = 0;
            continue;
        }
        let q = sa[i - 1].to_usize();
        while p + h < n && q + h < n && text[p + h] == text[q + h] {
            h += 1;
        }
        lcp[i] = I::from_usize(h);
        h = h.saturating_sub(1);
    }
    lcp
}

/// Largest tied group whose key vector is built on the stack. Groups average
/// about four elements, so nearly every one avoids the allocator entirely.
const DOUBLING_STACK_GROUP: usize = 32;

/// Bits of the key used for the MSD counting-sort pass, and the resulting
/// bucket count.
///
/// 2048 buckets keeps the write-combining state at 2048 × 2 streams × 128 B
/// ≈ 512 KB, which stays inside a core's private cache. Going to 16 bits
/// would need 16 MB of open write lines and thrashes the TLB instead.
const RADIX_BITS: u32 = 11;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;

/// Sort every suffix position by `(packed key, visible length)`, returning the
/// sorted positions and a bit per slot marking where a tied group starts.
///
/// The key array does not come back. Grouping only needs to know where one
/// group ends and the next begins, which is one bit per slot rather than the
/// eight bytes a key costs, and dropping the keys here rather than after the
/// grouping pass is what keeps the doubling path's peak below the merge
/// kernel's. See [`build_sa`] for the accounting.
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
) -> (Vec<AtomicU64>, Vec<I>) {
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

    // Pass 2: scatter positions into their buckets. Each chunk owns a
    // disjoint slice of every bucket, so the writes never collide even though
    // they are not contiguous.
    //
    // The key is computed here to pick the bucket and then thrown away. An
    // `n`-entry key array would be the largest allocation in the whole build,
    // 8 bytes per position against the 4 that `sa` costs, and it would still
    // be resident when the doubling rounds need their own arrays. Pass 3
    // recomputes the keys of one bucket at a time instead, which is one extra
    // key per position spread across the workers.
    let mut sa: Vec<I> = vec![I::zero(); n];
    {
        let sa_out = Scatter::new(&mut sa);
        bounds
            .par_iter()
            .enumerate()
            .for_each(|(c, &(start, end))| {
                let mut cursor: Vec<usize> =
                    offsets[c * RADIX_BUCKETS..(c + 1) * RADIX_BUCKETS].to_vec();
                for p in start..end {
                    let slot = &mut cursor[bucket_of(packer.key_at(text, p))];
                    // SAFETY: the prefix sum gives this (chunk, bucket) pair a
                    // range of exactly its own histogram count, and the cursor
                    // never leaves it, so no other thread writes this index.
                    unsafe { sa_out.set(*slot, I::from_usize(p)) };
                    *slot += 1;
                }
            });
    }

    // Pass 3: order within each bucket, and record where tied groups start.
    //
    // Buckets share their top `RADIX_BITS`, so what remains is the low bits of
    // the key and then the visible-length tie-break. Buckets are contiguous
    // and independent, and two positions in different buckets differ in the
    // key by construction, so a bucket's first slot always starts a group and
    // the group-start bits can be filled in here, per bucket, while that
    // bucket's keys exist.
    // Atomic words: two buckets can share the word their boundary bits live
    // in, so the bits are set with relaxed fetch-or. Reads are relaxed loads
    // and cost nothing once the fill is done.
    let starts: Vec<AtomicU64> = (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
    let mut rest_sa: &mut [I] = &mut sa;
    let mut slices: Vec<(usize, &mut [I])> = Vec::with_capacity(RADIX_BUCKETS);
    for b in 0..RADIX_BUCKETS {
        let len = bucket_start[b + 1] - bucket_start[b];
        let (sb, st) = rest_sa.split_at_mut(len);
        slices.push((bucket_start[b], sb));
        rest_sa = st;
    }
    {
        let start_bits = &starts;
        let set_bit = |h: usize| {
            start_bits[h / 64].fetch_or(1 << (h % 64), Ordering::Relaxed);
        };
        slices.into_par_iter().for_each(|(base, sb)| {
            if sb.is_empty() {
                return;
            }
            set_bit(base);
            if sb.len() > 1 {
                let mut pairs: Vec<(u64, I)> = sb
                    .iter()
                    .map(|&p| (packer.key_at(text, p.to_usize()), p))
                    .collect();
                pairs.sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then_with(|| visible_len(a.1.to_usize()).cmp(&visible_len(b.1.to_usize())))
                });
                for (i, &(_, pos)) in pairs.iter().enumerate() {
                    sb[i] = pos;
                }
                for i in 1..pairs.len() {
                    let (ka, pa) = pairs[i - 1];
                    let (kb, pb) = pairs[i];
                    if ka != kb || visible_len(pa.to_usize()) != visible_len(pb.to_usize()) {
                        set_bit(base + i);
                    }
                }
            }
        });
    }

    (starts, sa)
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
    let (starts, mut sa) = seed_sort::<I>(text, &packer, &visible_len);
    profile_log(&format!(
        "radix seed sort {:.3}s",
        t1.elapsed().as_secs_f64()
    ));
    let t2 = Instant::now();

    // Slot `h` begins a tied group exactly when the seed marked it, so two
    // adjacent slots tie exactly when the later one is not a start.
    let starts_group =
        |h: usize| -> bool { starts[h / 64].load(Ordering::Relaxed) >> (h % 64) & 1 == 1 };

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
    let groups: Vec<(I, I)> = (0..n)
        .into_par_iter()
        .filter_map(|h| {
            if !starts_group(h) {
                return None;
            }
            let mut e = h + 1;
            while e < n && !starts_group(e) {
                e += 1;
            }
            let g = I::from_usize(h);
            for entry in &sa[h..e] {
                // SAFETY: `sa` is a permutation of `0..n`, and this thread
                // owns the whole group `h..e`, so `entry` is a distinct index
                // no other thread writes.
                unsafe { ranks.set(entry.to_usize(), g) };
            }
            (e - h > 1).then_some((I::from_usize(h), I::from_usize(e)))
        })
        .collect();
    let mut groups = groups;
    profile_log(&format!(
        "radix groups    {} groups, {} MB",
        groups.len(),
        groups.len() * std::mem::size_of::<(I, I)>() / (1 << 20)
    ));
    drop(starts);
    profile_log(&format!(
        "radix grouping  {:.3}s",
        t2.elapsed().as_secs_f64()
    ));
    let t3 = Instant::now();

    // ---- Double: (rank_d(p), rank_d(p + d)) resolves to depth 2d. ----
    let mut depth = k;
    // Scratch for the new rank of each *tied* `sa` slot, so a round's reads
    // of `rank` never observe that same round's writes.
    //
    // Sized to the tied population rather than to `n`. Only slots inside a
    // group are rewritten, and even the first round has far fewer of those
    // than the text has positions (18% of them on chr21), so a full second
    // rank array would be mostly untouched pages. Each group gets a
    // contiguous window at its prefix-sum offset, and the buffer is reused
    // across rounds, which shrink monotonically.
    let mut next_rank: Vec<I> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();

    while !groups.is_empty() {
        let round_t = Instant::now();
        // Phase A: sort each tied group by the successor rank, and record the
        // ranks it should get. Groups are disjoint `sa` ranges, so this is
        // data-parallel with no synchronisation.
        // Groups average about four elements, and there are millions of them
        // per round, so the two things that dominated here were not the sort
        // or the rank probes but the bookkeeping around them: one heap
        // allocation per group for the key vector, and a sequential
        // `split_at_mut` chain to hand each group its sub-slices.
        //
        // Both go. `Scatter` already encodes "disjoint ranges, one owner
        // each", which is exactly the property the groups have, so each group
        // takes its own sub-slices directly with no sequential prepass. And a
        // group that fits the stack buffer never touches the allocator.
        // Window each group takes in the tied-slot buffer.
        offsets.clear();
        offsets.reserve(groups.len() + 1);
        let mut tied = 0usize;
        for &(start, end) in &groups {
            offsets.push(tied);
            tied += end.to_usize() - start.to_usize();
        }
        offsets.push(tied);
        if next_rank.len() < tied {
            next_rank.resize(tied, I::zero());
        }

        let sa_cell = Scatter::new(&mut sa);
        let nr_cell = Scatter::new(&mut next_rank);
        let rank_ref = &rank;
        let offsets_ref = &offsets;
        let sub: Vec<(I, I)> = groups
            .par_iter()
            .enumerate()
            .flat_map_iter(|(gi, &(start, end))| {
                let (start, end) = (start.to_usize(), end.to_usize());
                let len = end - start;
                // SAFETY: `groups` are disjoint, sorted `sa` ranges, so this
                // group is the sole owner of `start..end` in `sa` and of its
                // own prefix-sum window in the tied-slot buffer.
                let (sa_g, nr_g) = unsafe {
                    (
                        sa_cell.slice_mut(start, len),
                        nr_cell.slice_mut(offsets_ref[gi], len),
                    )
                };
                let succ = |p: usize| -> u64 {
                    // End-of-text sorts first: the shorter suffix is smaller.
                    match p.checked_add(depth) {
                        Some(q) if q < n => rank_ref[q].to_usize() as u64 + 1,
                        _ => 0,
                    }
                };

                let mut stack = [(0u64, I::zero()); DOUBLING_STACK_GROUP];
                let mut heap: Vec<(u64, I)>;
                let keyed: &mut [(u64, I)] = if len <= DOUBLING_STACK_GROUP {
                    let slot = &mut stack[..len];
                    for (dst, &e) in slot.iter_mut().zip(sa_g.iter()) {
                        *dst = (succ(e.to_usize()), e);
                    }
                    slot
                } else {
                    heap = sa_g.iter().map(|&e| (succ(e.to_usize()), e)).collect();
                    &mut heap
                };
                keyed.sort_unstable();

                let mut fresh: Vec<(I, I)> = Vec::new();
                let mut i = 0;
                while i < len {
                    let key = keyed[i].0;
                    let mut j = i + 1;
                    while j < len && keyed[j].0 == key {
                        j += 1;
                    }
                    let g = I::from_usize(start + i);
                    for slot in &mut nr_g[i..j] {
                        *slot = g;
                    }
                    if j - i > 1 {
                        fresh.push((I::from_usize(start + i), I::from_usize(start + j)));
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
        groups
            .par_iter()
            .enumerate()
            .for_each(|(gi, &(start, end))| {
                let base = offsets[gi];
                for (i, slot) in (start.to_usize()..end.to_usize()).enumerate() {
                    // SAFETY: `sa[start..end]` are distinct positions owned
                    // solely by this group, and the groups partition their
                    // index range.
                    unsafe { ranks.set(sa[slot].to_usize(), next_rank[base + i]) };
                }
            });

        let before: usize = groups
            .iter()
            .map(|&(s, e)| e.to_usize() - s.to_usize())
            .sum();
        let n_groups = groups.len();
        groups = sub;
        let after: usize = groups
            .iter()
            .map(|&(s, e)| e.to_usize() - s.to_usize())
            .sum();

        // A doubling round can only ever refine, so `after <= before`. If a
        // round refines nothing at all the text has a run longer than the
        // whole remaining depth budget; doubling still terminates because
        // `depth` grows geometrically and every suffix eventually runs off
        // the end of the text, which the sentinel orders. Guard against
        // overflow rather than against non-progress.
        profile_log(&format!(
            "  doubling round depth={depth}: {before} tied in {} groups              (avg {:.1}) -> {after} tied, {:.3}s",
            n_groups,
            before as f64 / n_groups.max(1) as f64,
            round_t.elapsed().as_secs_f64()
        ));
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

    /// Borrow `len` elements starting at `index` mutably.
    ///
    /// # Safety
    ///
    /// No other live borrow may overlap `index..index + len`, and the borrow
    /// the `Scatter` was built from must still be live.
    #[inline]
    unsafe fn slice_mut<'a>(&self, index: usize, len: usize) -> &'a mut [T] {
        debug_assert!(index + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(index), len) }
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

    /// `Symbol` is implemented for `i8`, and a one-byte-wide check alone lets a
    /// signed text through a packer that orders bytes as unsigned. `-1` has
    /// byte `0xFF`, so it would sort above `1`, inverting the true order.
    #[test]
    fn signed_symbols_are_not_eligible_for_packing() {
        let text: Vec<i8> = vec![-1, 0, -1, 1, -2, 0, 1, -1, 0, -2, 1, 0];
        assert!(
            seed_params(&text).is_none(),
            "i8 texts must not get a packed key"
        );
        // u8 of the same width still qualifies.
        let bytes: Vec<u8> = vec![1, 0, 1, 2, 3, 0];
        assert!(seed_params(&bytes).is_some());
    }

    /// Kasai's output must match a naive per-pair scan, on the inputs that
    /// make the naive version expensive: long runs and periodic text.
    #[test]
    fn kasai_matches_naive() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCA5A1);
        let mut fixtures: Vec<Vec<u8>> = vec![
            b"banana".to_vec(),
            b"mississippi".to_vec(),
            vec![7u8; 500],
            (0..500).map(|i| (i % 3) as u8).collect(),
            (0..500).map(|i| (i % 61) as u8).collect(),
            Vec::new(),
            vec![1],
        ];
        for &sigma in &[2u8, 4, 200] {
            for &n in &[7usize, 64, 1000] {
                fixtures.push((0..n).map(|_| rng.random_range(0..sigma)).collect());
            }
        }
        for text in fixtures {
            let sa: Vec<u32> = build_sa(&text);
            let lcp = kasai_lcp(&text, &sa);
            assert_eq!(lcp.len(), sa.len());
            if sa.is_empty() {
                continue;
            }
            assert_eq!(lcp[0], 0, "lcp[0] must be 0");
            for i in 1..sa.len() {
                let (a, b) = (sa[i - 1] as usize, sa[i] as usize);
                let want = (0..)
                    .take_while(|&j| {
                        a + j < text.len() && b + j < text.len() && text[a + j] == text[b + j]
                    })
                    .count();
                assert_eq!(lcp[i] as usize, want, "lcp[{i}] on {text:?}");
            }
        }
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
    /// order-preserving and the whole seed would be wrong. Checked through
    /// the observable behaviour: over a text whose bytes ascend and are all
    /// distinct, successive suffixes must produce strictly increasing keys.
    #[test]
    fn packer_keys_follow_byte_order() {
        for text in [
            b"\nACGNTZq".to_vec(),
            (0..40u8)
                .map(|i| i.wrapping_mul(6).wrapping_add(3))
                .collect(),
            b"ACGT".to_vec(),
        ] {
            let mut ascending: Vec<u8> = text.clone();
            ascending.sort_unstable();
            ascending.dedup();
            let p = Packer::new(&ascending);
            let keys: Vec<u64> = (0..ascending.len())
                .map(|i| p.key_at(&ascending, i))
                .collect();
            for (i, w) in keys.windows(2).enumerate() {
                assert!(
                    w[0] < w[1],
                    "key({i}) = {:#x} should precede key({}) = {:#x} for {ascending:?}",
                    w[0],
                    i + 1,
                    w[1],
                );
            }
        }
    }

    /// The SWAR gather must agree with the obvious shift-or loop for every
    /// field width and every alignment, including the zero-padded tail.
    #[test]
    fn swar_gather_matches_scalar_packing() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5AA5);
        for &sigma in &[2u8, 4, 16, 200] {
            for &n in &[1usize, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200] {
                let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..sigma)).collect();
                let p = Packer::new(&text);
                let (bits, k) = (p.bits(), p.k());
                let ranked = p.ranked.as_deref().unwrap_or(&text);
                for pos in 0..n {
                    let end = (pos + k).min(n);
                    let mut want: u64 = 0;
                    for &c in &ranked[pos..end] {
                        want = (want << bits) | c as u64;
                    }
                    want <<= bits as usize * (k - (end - pos));
                    assert_eq!(
                        p.key_at(&text, pos),
                        want,
                        "sigma={sigma} n={n} pos={pos} bits={bits}",
                    );
                }
            }
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
