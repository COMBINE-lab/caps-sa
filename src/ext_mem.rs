//! External-memory suffix array construction.
//!
//! Implements upstream CaPS-SA's *sample-sort + LCP-enhanced merge*
//! external-memory algorithm:
//!
//! 1. **Sort + sample + spill.** Split positions `0..n` into `p` subarrays
//!    of size `n / p`. In parallel, sort each with the in-memory
//!    merge-sort kernel (`sample_sort::merge_sort`), sample ~`c · ln n`
//!    positions uniformly from each sorted subarray, and spill the sorted
//!    `(position, lcp)` records to a per-subarray
//!    [`ExtMemBucket`][crate::ext_bucket::ExtMemBucket].
//! 2. **Select pivots.** Globally sort the pooled samples and pick
//!    `p − 1` pivots at evenly-spaced ranks, splitting the suffix order
//!    into `p` partitions.
//! 3. **Distribute.** For each sorted subarray, binary-search the pivots
//!    to find its `p` sub-subarray split points; append each sub-subarray
//!    to the corresponding *partition* bucket, marking a boundary after
//!    each contribution.
//! 4. **Per-partition merge.** Load each partition's bucket into RAM
//!    (≈ `n / p` records); cascade 2-way LCP-enhanced merges across the
//!    `p` sub-subarrays to produce that partition's globally-sorted slice.
//! 5. **Stream output.** Iterate partitions in order and emit each
//!    position via the caller's closure.
//!
//! Peak RAM ≈ `text` + `O(n / p)` per active worker (one partition's
//! merge working set). With `p = 4 × rayon::current_num_threads()` and
//! `num_threads = 8`, that's a few hundred MB on a `n = 6e9` (human-scale)
//! input — well below in-memory's `~4 × n × 8 = ~200 GB`. The full SA is
//! never materialized in RAM.

use std::cmp::Ordering;
use std::io;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::ext_bucket::{ExtMemBucket, SaLcp};
use crate::lcp::suffix_cmp;
use crate::sample_sort;

/// Tunable options for [`build_ext_mem`].
#[derive(Clone, Debug)]
pub struct ExtMemOpts {
    /// Bound on LCP-extension comparisons inside merges. `usize::MAX`
    /// (default) is unbounded.
    pub max_context: usize,
    /// Number of subarrays (`p` in upstream CaPS-SA). `0` (default) picks
    /// `4 × rayon::current_num_threads()`, clamped to `[1, n]`.
    pub subproblem_count: usize,
    /// Directory for temp files. Defaults to [`std::env::temp_dir`].
    pub work_dir: PathBuf,
}

impl Default for ExtMemOpts {
    fn default() -> Self {
        Self {
            max_context: usize::MAX,
            subproblem_count: 0,
            work_dir: std::env::temp_dir(),
        }
    }
}

impl ExtMemOpts {
    /// Convenience constructor with the supplied `work_dir` and defaults
    /// for everything else.
    pub fn with_work_dir(work_dir: impl AsRef<Path>) -> Self {
        Self {
            work_dir: work_dir.as_ref().to_path_buf(),
            ..Self::default()
        }
    }
}

/// Build the suffix array of `text` with bounded RAM, streaming each
/// output position to `emit` in lexicographic order.
///
/// Returns an [`io::Error`] if temp-file I/O fails. The callback may also
/// return an error to abort construction; partial work is discarded and
/// temp files are cleaned up when their bucket drops.
///
/// Equivalent in semantics to [`crate::build_in_memory`]: produces a
/// standard lexicographic suffix array with the "shorter suffix is
/// smaller when one runs off the end" tie-break.
pub fn build_ext_mem<S, F>(text: &[S], opts: &ExtMemOpts, mut emit: F) -> io::Result<()>
where
    S: Ord + Copy + Sync,
    F: FnMut(u64) -> io::Result<()>,
{
    let n = text.len();
    if n == 0 {
        return Ok(());
    }

    let p = effective_subproblem_count(n, opts.subproblem_count);

    // Phase 1: sort each subarray; sample uniformly; spill (pos, lcp).
    let (mut subarray_buckets, samples) = phase1_sort_sample_spill(text, n, p, opts)?;

    // Phase 2: globally sort samples; pick p-1 pivots.
    let pivots = phase2_select_pivots(text, samples, p, opts.max_context);

    // Phase 3: distribute each subarray into per-partition buckets.
    let mut partition_buckets = phase3_distribute(text, &mut subarray_buckets, &pivots, p, opts)?;

    // Subarray buckets are no longer needed — drop early to release
    // their disk space.
    drop(subarray_buckets);

    // Phase 4 & 5: per-partition cascade merge + stream output.
    phase4_merge_and_emit(text, &mut partition_buckets, opts.max_context, &mut emit)
}

fn effective_subproblem_count(n: usize, requested: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let raw = if requested == 0 {
        rayon::current_num_threads().saturating_mul(4)
    } else {
        requested
    };
    raw.clamp(1, n)
}

/// Phase 1: sort each subarray in parallel, sample from it, and spill
/// `(position, lcp)` records to its own [`ExtMemBucket`].
fn phase1_sort_sample_spill<S>(
    text: &[S],
    n: usize,
    p: usize,
    opts: &ExtMemOpts,
) -> io::Result<(Vec<ExtMemBucket<SaLcp>>, Vec<u64>)>
where
    S: Ord + Copy + Sync,
{
    let chunk_size = n.div_ceil(p);
    let samples_target_total = sample_target_total(n, p);

    let per_subarray: Vec<(ExtMemBucket<SaLcp>, Vec<u64>)> = (0..p)
        .into_par_iter()
        .map(|i| {
            let start = (i * chunk_size).min(n);
            let end = ((i + 1) * chunk_size).min(n);
            let len = end - start;

            let mut bucket = ExtMemBucket::new(&opts.work_dir, format!("sub{i}"));
            if len == 0 {
                return Ok::<_, io::Error>((bucket, Vec::new()));
            }

            // In-memory sort of this subarray with LCP maintenance.
            let mut sa: Vec<u64> = (start as u64..end as u64).collect();
            let mut sa_w = vec![0u64; len];
            let mut lcp_arr = vec![0u64; len];
            let mut lcp_w = vec![0u64; len];
            sample_sort::merge_sort(
                text,
                &mut sa,
                &mut sa_w,
                &mut lcp_arr,
                &mut lcp_w,
                opts.max_context,
            );

            // Pull `samples_per_subarray` evenly-spaced positions out of
            // the now-sorted subarray. Deterministic — no RNG needed for
            // pivot selection to be globally well-distributed.
            let samples_per_subarray = samples_target_total.div_ceil(p).min(len);
            let samples = evenly_spaced(&sa, samples_per_subarray);

            // Spill (position, lcp) records to the bucket. `lcp[0]`
            // remains 0 (set by the merge-sort base case), making each
            // subarray its own well-formed LCP-annotated sorted run.
            let records: Vec<SaLcp> = sa
                .iter()
                .zip(lcp_arr.iter())
                .map(|(&p, &l)| SaLcp { pos: p, lcp: l })
                .collect();
            bucket.add_slice(&records)?;

            Ok((bucket, samples))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut buckets = Vec::with_capacity(p);
    let mut all_samples = Vec::with_capacity(samples_target_total);
    for (bucket, samples) in per_subarray {
        buckets.push(bucket);
        all_samples.extend(samples);
    }
    Ok((buckets, all_samples))
}

/// Target sample count *across all subarrays*. Matches upstream CaPS-SA's
/// "`c · ln n`" rule per subarray with `c = 4`, so the global pool is
/// `p · 4 · ln n` samples.
fn sample_target_total(n: usize, p: usize) -> usize {
    let ln_n = (n as f64).ln().max(1.0);
    let per = (4.0 * ln_n).ceil() as usize;
    // At least p (so we have enough to pick p-1 pivots) and at most n.
    p.saturating_mul(per).clamp(p, n)
}

/// Pick `count` evenly-spaced elements from a slice. Deterministic, which
/// keeps the algorithm reproducible without an RNG dependency.
fn evenly_spaced<T: Copy>(xs: &[T], count: usize) -> Vec<T> {
    let n = xs.len();
    if count == 0 || n == 0 {
        return Vec::new();
    }
    if count >= n {
        return xs.to_vec();
    }
    // Pick indices at positions (i + 0.5) · n / count for i in 0..count,
    // i.e. evenly-spaced midpoints. Avoids both endpoints — keeps pivots
    // away from extreme corners of the order.
    (0..count)
        .map(|i| xs[(2 * i + 1) * n / (2 * count)])
        .collect()
}

/// Phase 2: globally sort the pooled samples and pick `p − 1` pivots at
/// evenly-spaced ranks.
fn phase2_select_pivots<S>(text: &[S], mut samples: Vec<u64>, p: usize, max_ctx: usize) -> Vec<u64>
where
    S: Ord + Copy + Sync,
{
    if p <= 1 || samples.is_empty() {
        return Vec::new();
    }
    let n_samples = samples.len();
    let mut sa_w = vec![0u64; n_samples];
    let mut lcp = vec![0u64; n_samples];
    let mut lcp_w = vec![0u64; n_samples];
    sample_sort::merge_sort(text, &mut samples, &mut sa_w, &mut lcp, &mut lcp_w, max_ctx);

    // p-1 pivots at evenly-spaced ranks across the sorted sample pool.
    (1..p).map(|j| samples[(j * n_samples) / p]).collect()
}

/// Phase 3: walk each subarray (sequentially in this v1 — locks would be
/// the only complication of a parallel version, deferred to a follow-up),
/// load it into RAM, binary-search the pivots to find its `p` sub-subarray
/// boundaries, and append each sub-subarray to the corresponding partition
/// bucket. Marks a [`ExtMemBucket::mark_boundary`] after each non-empty
/// contribution so the per-partition merge can recover the runs.
fn phase3_distribute<S>(
    text: &[S],
    subarray_buckets: &mut [ExtMemBucket<SaLcp>],
    pivots: &[u64],
    p: usize,
    opts: &ExtMemOpts,
) -> io::Result<Vec<ExtMemBucket<SaLcp>>>
where
    S: Ord + Copy,
{
    let mut partition_buckets: Vec<ExtMemBucket<SaLcp>> = (0..p)
        .map(|j| ExtMemBucket::new(&opts.work_dir, format!("part{j}")))
        .collect();

    for sub_bucket in subarray_buckets.iter_mut() {
        if sub_bucket.total_records() == 0 {
            continue;
        }
        let records = sub_bucket.load_all()?;

        // Find p-1 split points by binary-searching each pivot's *upper
        // bound* in the sorted subarray.
        let mut splits = Vec::with_capacity(p + 1);
        splits.push(0usize);
        for &pivot in pivots {
            splits.push(upper_bound_by_pivot(
                &records,
                pivot,
                text,
                opts.max_context,
            ));
        }
        splits.push(records.len());

        // Distribute each sub-subarray. Reset the first record's `lcp` to
        // 0 so the per-partition merge sees a well-formed boundary.
        for j in 0..p {
            let lo = splits[j];
            let hi = splits[j + 1];
            if lo >= hi {
                continue;
            }
            let mut sub: Vec<SaLcp> = records[lo..hi].to_vec();
            sub[0].lcp = 0;
            partition_buckets[j].add_slice(&sub)?;
            partition_buckets[j].mark_boundary();
        }
        // `records` is dropped at end of loop iteration; the subarray
        // bucket's underlying file is removed when it eventually drops.
    }

    Ok(partition_buckets)
}

/// Upper-bound binary search: returns the first index `i` such that the
/// suffix at `records[i].pos` is **strictly greater than** the suffix at
/// `pivot`.
fn upper_bound_by_pivot<S>(records: &[SaLcp], pivot: u64, text: &[S], max_ctx: usize) -> usize
where
    S: Ord + Copy,
{
    let mut lo = 0;
    let mut hi = records.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match suffix_cmp(text, records[mid].pos as usize, pivot as usize, max_ctx) {
            Ordering::Greater => hi = mid,
            Ordering::Equal | Ordering::Less => lo = mid + 1,
        }
    }
    lo
}

/// Phase 4 + 5: for each partition, load its bucket into RAM, cascade
/// 2-way LCP-enhanced merge across the sub-subarrays, and stream each
/// merged position to `emit`.
fn phase4_merge_and_emit<S, F>(
    text: &[S],
    partition_buckets: &mut [ExtMemBucket<SaLcp>],
    max_ctx: usize,
    emit: &mut F,
) -> io::Result<()>
where
    S: Ord + Copy + Sync,
    F: FnMut(u64) -> io::Result<()>,
{
    for bucket in partition_buckets.iter_mut() {
        if bucket.total_records() == 0 {
            continue;
        }
        let records = bucket.load_all()?;
        let boundaries: Vec<usize> = bucket.boundaries().to_vec();

        let merged = cascade_merge(text, &records, &boundaries, max_ctx);
        for pos in merged {
            emit(pos)?;
        }
    }
    Ok(())
}

/// Cascade 2-way LCP-enhanced merges across the sub-subarrays of one
/// partition (delimited by `boundaries`) until a single sorted run
/// remains. Returns its position list.
fn cascade_merge<S>(
    text: &[S],
    records: &[SaLcp],
    boundaries: &[usize],
    max_ctx: usize,
) -> Vec<u64>
where
    S: Ord + Copy,
{
    // Extract each sub-subarray into separate SA / LCP slices — the
    // existing `merge` kernel is SOA-shaped.
    let mut runs: Vec<(Vec<u64>, Vec<u64>)> = (0..boundaries.len().saturating_sub(1))
        .filter_map(|i| {
            let lo = boundaries[i];
            let hi = boundaries[i + 1];
            if lo >= hi {
                return None;
            }
            let sa: Vec<u64> = records[lo..hi].iter().map(|r| r.pos).collect();
            let lcp: Vec<u64> = records[lo..hi].iter().map(|r| r.lcp).collect();
            Some((sa, lcp))
        })
        .collect();

    if runs.is_empty() {
        return Vec::new();
    }

    while runs.len() > 1 {
        let mut next: Vec<(Vec<u64>, Vec<u64>)> = Vec::with_capacity(runs.len().div_ceil(2));
        let mut iter = runs.into_iter();
        while let Some((sa1, lcp1)) = iter.next() {
            match iter.next() {
                None => next.push((sa1, lcp1)),
                Some((sa2, lcp2)) => {
                    let total = sa1.len() + sa2.len();
                    let mut z_sa = vec![0u64; total];
                    let mut z_lcp = vec![0u64; total];
                    sample_sort::merge(
                        text, &sa1, &sa2, &lcp1, &lcp2, &mut z_sa, &mut z_lcp, max_ctx,
                    );
                    next.push((z_sa, z_lcp));
                }
            }
        }
        runs = next;
    }

    let (sa, _lcp) = runs.into_iter().next().unwrap();
    sa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_in_memory;
    use tempfile::tempdir;

    fn ext_mem_sa(text: &[u8], p: usize) -> Vec<u64> {
        let dir = tempdir().unwrap();
        let opts = ExtMemOpts {
            subproblem_count: p,
            work_dir: dir.path().to_path_buf(),
            ..ExtMemOpts::default()
        };
        let mut out: Vec<u64> = Vec::with_capacity(text.len());
        build_ext_mem(text, &opts, |pos| {
            out.push(pos);
            Ok(())
        })
        .unwrap();
        out
    }

    fn assert_matches_in_memory(text: &[u8], p: usize) {
        let want: Vec<u32> = build_in_memory(text);
        let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
        let got = ext_mem_sa(text, p);
        assert_eq!(got, want64, "mismatch on text {text:?} with p={p}");
    }

    #[test]
    fn ext_mem_empty() {
        let got = ext_mem_sa(b"", 4);
        assert!(got.is_empty());
    }

    #[test]
    fn ext_mem_single_partition() {
        assert_matches_in_memory(b"banana", 1);
    }

    #[test]
    fn ext_mem_p_greater_than_n() {
        assert_matches_in_memory(b"abc", 10);
    }

    #[test]
    fn ext_mem_banana_p4() {
        assert_matches_in_memory(b"banana", 4);
    }

    #[test]
    fn ext_mem_mississippi_p3() {
        assert_matches_in_memory(b"mississippi", 3);
    }

    #[test]
    fn ext_mem_random_byte_texts() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE);
        for &n in &[16usize, 100, 1000, 5000] {
            for &p in &[1usize, 2, 4, 16] {
                let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
                assert_matches_in_memory(&text, p);
            }
        }
    }

    #[test]
    fn ext_mem_with_unique_terminator() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xF00D);
        for &n in &[10usize, 200, 2000] {
            for &p in &[1usize, 3, 8] {
                let mut text: Vec<u8> = (0..n).map(|_| rng.random_range(0..5u8)).collect();
                text.push(200);
                assert_matches_in_memory(&text, p);
            }
        }
    }

    #[test]
    fn ext_mem_repetitive_does_not_blow_up() {
        // Many copies of a long repeat — what killed the Phase 2 v1
        // linear-scan merge. The sample-sort + LCP-enhanced cascade
        // should handle it in proportional time.
        use std::time::Instant;
        let unit = b"ACGTACGTACGTACGTACGTACGTACGT"; // 28 bases
        let mut text: Vec<u8> = Vec::new();
        for _ in 0..100 {
            text.extend_from_slice(unit);
        }
        text.push(200);
        let start = Instant::now();
        let got = ext_mem_sa(&text, 8);
        let elapsed = start.elapsed();
        let want: Vec<u32> = build_in_memory(&text);
        let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
        assert_eq!(got, want64);
        // Sanity: should finish in well under a second on this input.
        assert!(
            elapsed.as_secs() < 2,
            "ext-mem build on a tiny repetitive text took {elapsed:?}"
        );
    }
}
