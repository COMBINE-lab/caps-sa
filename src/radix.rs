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
//! 1. **Pack.** Find the maximum symbol and choose the smallest field width
//!    in `{1, 2, 4, 8}` bits that can hold it, so `k = 64 / bits` symbols fit
//!    in one `u64` key. DNA over `{0,1,2,3}` gets 2-bit fields and therefore
//!    resolves **32 symbols per key** rather than the 8 a raw byte key would.
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
use rayon::prelude::*;

/// Field width in bits and the number of symbols that fit in a `u64` key.
///
/// Restricted to divisors of 64 so a key is an exact number of whole fields
/// and no symbol ever straddles the key boundary.
fn pack_params(max_sym: u8) -> (u32, usize) {
    let bits: u32 = match max_sym {
        0..=1 => 1,
        2..=3 => 2,
        4..=15 => 4,
        _ => 8,
    };
    (bits, 64 / bits as usize)
}

/// Pack the `k` symbols at `text[p..]` into one order-preserving `u64`,
/// zero-padding past the end of the text.
#[inline]
fn key_at(text: &[u8], p: usize, bits: u32, k: usize) -> u64 {
    if bits == 8 {
        // Whole-byte fields: this is just a big-endian load, and the common
        // case (p + 8 <= n) is a single unaligned u64 read plus a bswap.
        let mut buf = [0u8; 8];
        let end = (p + 8).min(text.len());
        buf[..end - p].copy_from_slice(&text[p..end]);
        return u64::from_be_bytes(buf);
    }
    let end = (p + k).min(text.len());
    let mut key: u64 = 0;
    for &s in &text[p..end] {
        key = (key << bits) | s as u64;
    }
    // Shift the packed prefix up so the missing trailing fields read as zero.
    key << (bits as usize * (k - (end - p)))
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

    let max_sym = text.par_iter().copied().max().unwrap_or(0);
    let (bits, k) = pack_params(max_sym);

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
    let mut seeded: Vec<(u64, u32, I)> = (0..n)
        .into_par_iter()
        .map(|p| {
            (
                key_at(text, p, bits, k),
                (n - p).min(k) as u32,
                I::from_usize(p),
            )
        })
        .collect();
    seeded.par_sort_unstable();

    let mut sa: Vec<I> = Vec::with_capacity(n);
    sa.par_extend(seeded.par_iter().map(|&(_, _, p)| p));

    // `rank[p]` is the index in `sa` of the first element of `p`'s group, so
    // two suffixes tie at the current depth exactly when their ranks match,
    // and rank order is the current partial order.
    let mut rank: Vec<I> = vec![I::zero(); n];
    // Non-singleton `sa` ranges, the only ones any later round touches.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    {
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && (seeded[j].0, seeded[j].1) == (seeded[i].0, seeded[i].1) {
                j += 1;
            }
            let g = I::from_usize(i);
            for e in &sa[i..j] {
                rank[e.to_usize()] = g;
            }
            if j - i > 1 {
                groups.push((i, j));
            }
            i = j;
        }
    }
    drop(seeded);

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
                sa_g.sort_unstable_by_key(|e| succ(e.to_usize()));

                let mut fresh = Vec::new();
                let mut i = 0;
                while i < sa_g.len() {
                    let key = succ(sa_g[i].to_usize());
                    let mut j = i + 1;
                    while j < sa_g.len() && succ(sa_g[j].to_usize()) == key {
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
                fresh
            })
            .collect();

        // Phase B: publish the new ranks, now that every read is done.
        for &(start, end) in &groups {
            for i in start..end {
                rank[sa[i].to_usize()] = next_rank[i];
            }
        }

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

    sa
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

    #[test]
    fn pack_params_covers_every_width() {
        assert_eq!(pack_params(0), (1, 64));
        assert_eq!(pack_params(1), (1, 64));
        assert_eq!(pack_params(3), (2, 32));
        assert_eq!(pack_params(4), (4, 16));
        assert_eq!(pack_params(15), (4, 16));
        assert_eq!(pack_params(16), (8, 8));
        assert_eq!(pack_params(255), (8, 8));
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
