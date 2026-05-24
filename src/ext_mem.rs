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
use crate::lcp::LcpDispatch;
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
pub fn build_ext_mem<S, F>(text: &[S], opts: &ExtMemOpts, emit: F) -> io::Result<()>
where
    S: Ord + Copy + Sync + 'static,
    F: FnMut(u64) -> io::Result<()>,
{
    build_ext_mem_inner(text, PositionSource::Identity(text.len()), opts, emit)
}

/// Like [`build_ext_mem`] but sorts only the caller-supplied
/// `positions` by the lexicographic order of their suffixes in `text`.
/// Suffix content is always `text[position..]`; no positions are
/// dropped from the input. To filter, the caller constructs
/// `positions` with only the indices they want.
///
/// This lets STAR-style genome indexing skip the bin-padding
/// pathology: pass only the ACGT-starting positions and the
/// spacer-suffix work disappears.
///
/// `positions` does not need to be pre-sorted in any way.
pub fn build_ext_mem_for_positions<S, F>(
    text: &[S],
    positions: Vec<u64>,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Ord + Copy + Sync + 'static,
    F: FnMut(u64) -> io::Result<()>,
{
    // We hold a reference to `positions` for the duration of the build;
    // `phase1_sort_sample_spill` copies each chunk out. The Vec is
    // dropped once phase 1 returns.
    build_ext_mem_inner(text, PositionSource::Subset(&positions), opts, emit)
}

fn build_ext_mem_inner<S, F>(
    text: &[S],
    source: PositionSource<'_>,
    opts: &ExtMemOpts,
    mut emit: F,
) -> io::Result<()>
where
    S: Ord + Copy + Sync + 'static,
    F: FnMut(u64) -> io::Result<()>,
{
    let n = source.len();
    if n == 0 {
        return Ok(());
    }

    let p = effective_subproblem_count(n, opts.subproblem_count);

    // Choose the LCP implementation once for the whole build. The
    // captured function pointer rides through every phase (and across
    // rayon thread boundaries — `LcpDispatch` is `Copy + Send + Sync`),
    // so inner merge loops pay no atomic load or feature-detection
    // branch per call.
    let dispatch = LcpDispatch::detect();

    // Phase 1: sort each subarray; sample uniformly; spill (pos, lcp).
    let (mut subarray_buckets, samples) =
        phase1_sort_sample_spill(text, &source, p, opts, dispatch)?;

    // Phase 2: globally sort samples; pick p-1 pivots.
    let pivots = phase2_select_pivots(text, samples, p, opts.max_context, dispatch);

    // Phase 3: distribute each subarray into per-partition buckets.
    let mut partition_buckets =
        phase3_distribute(text, &mut subarray_buckets, &pivots, p, opts, dispatch)?;

    // Subarray buckets are no longer needed — drop early to release
    // their disk space.
    drop(subarray_buckets);

    // Phase 4 & 5: per-partition cascade merge + stream output.
    phase4_merge_and_emit(
        text,
        &mut partition_buckets,
        opts.max_context,
        &mut emit,
        dispatch,
    )
}

/// Source of the positions to sort. The all-suffixes case
/// ([`PositionSource::Identity`]) avoids materialising a `Vec<u64>` of
/// length `n`, which on the human genome would itself be ~25 GB.
enum PositionSource<'a> {
    Identity(usize),
    Subset(&'a [u64]),
}

impl<'a> PositionSource<'a> {
    fn len(&self) -> usize {
        match self {
            Self::Identity(n) => *n,
            Self::Subset(p) => p.len(),
        }
    }

    /// Fill `dst` with positions for the half-open subarray range
    /// `[start, start + dst.len())`. For [`PositionSource::Identity`]
    /// this generates the contiguous integer range on the fly; for
    /// [`PositionSource::Subset`] it copies from the caller's slice.
    fn fill_chunk(&self, start: usize, dst: &mut [u64]) {
        match self {
            Self::Identity(_) => {
                for (i, slot) in dst.iter_mut().enumerate() {
                    *slot = (start + i) as u64;
                }
            }
            Self::Subset(p) => {
                let end = start + dst.len();
                dst.copy_from_slice(&p[start..end]);
            }
        }
    }
}

/// Target subarray size used by [`effective_subproblem_count`] when
/// auto-picking `p`. Smaller means more (smaller) subarrays — lower
/// per-task phase-1 scratch, at the cost of more phase-3 distribute
/// work (which scales as `O(p² · log(n/p))`, sequentially) and a
/// higher temp-file count.
const PHASE1_TARGET_CHUNK: usize = 65_536;
/// Hard cap on the number of subarrays. With phase 3 currently
/// sequential, scaling `p` past a couple of thousand causes the
/// `O(p²)` distribute pass to swamp the rest of the algorithm at
/// human-genome scale (measured: `p = 8192` doubled the wall time
/// relative to `p = 128` on GRCh38 at 32 threads, even though it cut
/// peak RSS to ~6 GB). 2048 keeps phase 3 sub-second on a 3 GB input
/// while pushing per-task scratch low enough for `T × n/p` workspace
/// to fit comfortably alongside the loaded text.
const PHASE1_MAX_PARTITIONS: usize = 2048;

fn effective_subproblem_count(n: usize, requested: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let raw = if requested == 0 {
        let nthreads = rayon::current_num_threads().max(1);
        let p_from_size = n.div_ceil(PHASE1_TARGET_CHUNK);
        // At least one chunk per thread (otherwise we leave cores idle),
        // at most `PHASE1_MAX_PARTITIONS` (so phase 3's sequential
        // `O(p²)` sweep and the temp-file count stay manageable). For
        // small inputs `p_from_size` is well below the cap, so the
        // formula degrades gracefully to roughly "one chunk per thread";
        // for human-scale inputs the cap binds and per-task scratch
        // stays in the tens-of-MB range.
        p_from_size.clamp(nthreads, PHASE1_MAX_PARTITIONS)
    } else {
        requested
    };
    raw.clamp(1, n)
}

/// Phase 1: sort each subarray in parallel, sample from it, and spill
/// `(position, lcp)` records to its own [`ExtMemBucket`].
///
/// One rayon task per subarray; rayon's work-stealing scheduler keeps
/// all worker threads busy and lets `merge_sort`'s inner
/// [`rayon::join`] recursion steal idle slots. With the auto-picked
/// `p` (target chunk ~ 64 K records), per-task scratch is ~18 MiB on
/// human-scale inputs, so the `num_threads × per_task_scratch` peak
/// stays bounded even though we don't reuse buffers across iterations.
fn phase1_sort_sample_spill<S>(
    text: &[S],
    source: &PositionSource<'_>,
    p: usize,
    opts: &ExtMemOpts,
    dispatch: LcpDispatch,
) -> io::Result<(Vec<ExtMemBucket<SaLcp>>, Vec<u64>)>
where
    S: Ord + Copy + Sync + 'static,
{
    let n = source.len();
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
            let mut sa: Vec<u64> = vec![0u64; len];
            source.fill_chunk(start, &mut sa);
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
                dispatch,
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
fn phase2_select_pivots<S>(
    text: &[S],
    mut samples: Vec<u64>,
    p: usize,
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> Vec<u64>
where
    S: Ord + Copy + Sync + 'static,
{
    if p <= 1 || samples.is_empty() {
        return Vec::new();
    }
    let n_samples = samples.len();
    let mut sa_w = vec![0u64; n_samples];
    let mut lcp = vec![0u64; n_samples];
    let mut lcp_w = vec![0u64; n_samples];
    sample_sort::merge_sort(
        text,
        &mut samples,
        &mut sa_w,
        &mut lcp,
        &mut lcp_w,
        max_ctx,
        dispatch,
    );

    // p-1 pivots at evenly-spaced ranks across the sorted sample pool.
    (1..p).map(|j| samples[(j * n_samples) / p]).collect()
}

/// Phase 3: walk each subarray (sequentially in this v1 — locks would be
/// the only complication of a parallel version, deferred to a follow-up),
/// load it into RAM, binary-search the pivots to find its `p` sub-subarray
/// boundaries, and append each sub-subarray to the corresponding partition
/// bucket. Marks a [`ExtMemBucket::mark_boundary`] after each non-empty
/// contribution so the per-partition merge can recover the runs.
#[allow(clippy::too_many_arguments)]
fn phase3_distribute<S>(
    text: &[S],
    subarray_buckets: &mut [ExtMemBucket<SaLcp>],
    pivots: &[u64],
    p: usize,
    opts: &ExtMemOpts,
    dispatch: LcpDispatch,
) -> io::Result<Vec<ExtMemBucket<SaLcp>>>
where
    S: Ord + Copy + 'static,
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
                dispatch,
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
fn upper_bound_by_pivot<S>(
    records: &[SaLcp],
    pivot: u64,
    text: &[S],
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> usize
where
    S: Ord + Copy + 'static,
{
    let mut lo = 0;
    let mut hi = records.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match dispatch.suffix_cmp(text, records[mid].pos as usize, pivot as usize, max_ctx) {
            Ordering::Greater => hi = mid,
            Ordering::Equal | Ordering::Less => lo = mid + 1,
        }
    }
    lo
}

/// Phase 4 + 5: parallel-merge partitions in chunks of `num_threads`,
/// emitting each chunk's results in lex order before starting the next.
///
/// Each worker thread holds its own [`CascadeWorkspace`] for the duration
/// of one partition merge. Within a chunk, all `T` workspaces live in
/// parallel; between chunks, they are dropped (so peak workspace memory
/// scales with `T`, not with the number of partitions). The merged result
/// for each partition is then drained sequentially via `emit` to preserve
/// streaming-output order without ever holding the full SA in RAM.
///
/// Peak transient RAM ≈ `T × max_partition_size × 16 bytes` for the
/// merged-result buffers, plus the workspaces themselves (~`2 × T ×
/// max_partition_size × 16` bytes). On a typical run with `p = 4 × T`
/// subarrays the per-partition size is `≈ n / p`, so this stays
/// proportional to `n / 4 = 0.25 n` even at the peak — well below the
/// in-memory path's `~4 n` working set.
fn phase4_merge_and_emit<S, F>(
    text: &[S],
    partition_buckets: &mut [ExtMemBucket<SaLcp>],
    max_ctx: usize,
    emit: &mut F,
    dispatch: LcpDispatch,
) -> io::Result<()>
where
    S: Ord + Copy + Sync + 'static,
    F: FnMut(u64) -> io::Result<()>,
{
    let n_partitions = partition_buckets.len();
    if n_partitions == 0 {
        return Ok(());
    }
    let chunk_size = rayon::current_num_threads().max(1);

    let mut start = 0;
    while start < n_partitions {
        let end = (start + chunk_size).min(n_partitions);
        let chunk = &mut partition_buckets[start..end];

        // Parallel-merge each non-empty bucket in this chunk. `par_iter_mut`
        // preserves index order in the collected `Vec`, so the subsequent
        // sequential emit yields positions in lex order.
        let merged: Vec<Vec<u64>> = chunk
            .par_iter_mut()
            .map(|bucket| -> io::Result<Vec<u64>> {
                if bucket.total_records() == 0 {
                    return Ok(Vec::new());
                }
                let records = bucket.load_all()?;
                let boundaries: Vec<usize> = bucket.boundaries().to_vec();
                let mut workspace = CascadeWorkspace::new();
                let result =
                    workspace.cascade_merge(text, &records, &boundaries, max_ctx, dispatch);
                Ok(result.to_vec())
            })
            .collect::<Result<Vec<_>, io::Error>>()?;

        for positions in merged {
            for pos in positions {
                emit(pos)?;
            }
        }

        start = end;
    }
    Ok(())
}

/// Reusable ping-pong scratch for the partition cascade merge.
///
/// Holds two `(sa, lcp)` buffers each sized to the largest partition seen.
/// The cascade alternates reads from one side and writes to the other,
/// flipping a `src_is_a` flag after each level. Avoids the
/// per-level allocations that the previous immutable-`Vec` cascade
/// performed for every pair of sub-subarrays.
struct CascadeWorkspace {
    a_sa: Vec<u64>,
    a_lcp: Vec<u64>,
    b_sa: Vec<u64>,
    b_lcp: Vec<u64>,
}

impl CascadeWorkspace {
    fn new() -> Self {
        Self {
            a_sa: Vec::new(),
            a_lcp: Vec::new(),
            b_sa: Vec::new(),
            b_lcp: Vec::new(),
        }
    }

    /// Grow all four buffers to at least `n` elements. The contents past
    /// the cascade's actual run lengths are don't-care.
    fn ensure_capacity(&mut self, n: usize) {
        if self.a_sa.len() < n {
            self.a_sa.resize(n, 0);
            self.a_lcp.resize(n, 0);
            self.b_sa.resize(n, 0);
            self.b_lcp.resize(n, 0);
        }
    }

    /// Cascade 2-way LCP-enhanced merges across the sub-subarrays of one
    /// partition (delimited by `boundaries`) until a single sorted run
    /// remains. Returns a borrow of the buffer slot holding the final
    /// sorted positions.
    fn cascade_merge<'a, S>(
        &'a mut self,
        text: &[S],
        records: &[SaLcp],
        boundaries: &[usize],
        max_ctx: usize,
        dispatch: LcpDispatch,
    ) -> &'a [u64]
    where
        S: Ord + Copy + 'static,
    {
        let n = records.len();
        if n == 0 {
            return &self.a_sa[..0];
        }
        self.ensure_capacity(n);

        // Initialize side A in SOA form from the AOS `records`, and
        // collect the lengths of the non-empty sub-subarrays.
        let mut run_lens: Vec<usize> = boundaries
            .windows(2)
            .filter_map(|w| {
                let l = w[1] - w[0];
                if l > 0 { Some(l) } else { None }
            })
            .collect();
        for (i, r) in records.iter().enumerate() {
            self.a_sa[i] = r.pos;
            self.a_lcp[i] = r.lcp;
        }

        let mut src_is_a = true;
        while run_lens.len() > 1 {
            run_lens = self.merge_one_level(src_is_a, &run_lens, text, max_ctx, dispatch);
            src_is_a = !src_is_a;
        }

        if src_is_a {
            &self.a_sa[..n]
        } else {
            &self.b_sa[..n]
        }
    }

    /// Pair the runs in `run_lens` (last odd one passes through unchanged),
    /// running each pair through the LCP-enhanced 2-way merge from the
    /// `src_is_a`-selected buffer side into the other. Returns the new
    /// run-length list (each entry is the sum of the two it replaced, or
    /// the carry-over for an odd tail).
    fn merge_one_level<S>(
        &mut self,
        src_is_a: bool,
        run_lens: &[usize],
        text: &[S],
        max_ctx: usize,
        dispatch: LcpDispatch,
    ) -> Vec<usize>
    where
        S: Ord + Copy + 'static,
    {
        // Destructure self so the borrow checker can see the two sides as
        // disjoint locals — we borrow one immutably and the other mutably.
        let Self {
            a_sa,
            a_lcp,
            b_sa,
            b_lcp,
        } = self;
        let (src_sa, src_lcp, dst_sa, dst_lcp) = if src_is_a {
            (
                a_sa.as_slice(),
                a_lcp.as_slice(),
                b_sa.as_mut_slice(),
                b_lcp.as_mut_slice(),
            )
        } else {
            (
                b_sa.as_slice(),
                b_lcp.as_slice(),
                a_sa.as_mut_slice(),
                a_lcp.as_mut_slice(),
            )
        };

        let mut new_lens = Vec::with_capacity(run_lens.len().div_ceil(2));
        let mut src_off = 0usize;
        let mut dst_off = 0usize;
        let mut i = 0;
        while i < run_lens.len() {
            let l1 = run_lens[i];
            if i + 1 < run_lens.len() {
                let l2 = run_lens[i + 1];
                let x_end = src_off + l1;
                let xy_end = x_end + l2;
                let dst_end = dst_off + l1 + l2;
                sample_sort::merge(
                    text,
                    &src_sa[src_off..x_end],
                    &src_sa[x_end..xy_end],
                    &src_lcp[src_off..x_end],
                    &src_lcp[x_end..xy_end],
                    &mut dst_sa[dst_off..dst_end],
                    &mut dst_lcp[dst_off..dst_end],
                    max_ctx,
                    dispatch,
                );
                new_lens.push(l1 + l2);
                src_off = xy_end;
                dst_off = dst_end;
                i += 2;
            } else {
                // Odd run carries over unchanged.
                let end = dst_off + l1;
                dst_sa[dst_off..end].copy_from_slice(&src_sa[src_off..src_off + l1]);
                dst_lcp[dst_off..end].copy_from_slice(&src_lcp[src_off..src_off + l1]);
                new_lens.push(l1);
                src_off += l1;
                dst_off = end;
                i += 1;
            }
        }
        new_lens
    }
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

    fn ext_mem_for_positions(text: &[u8], positions: Vec<u64>, p: usize) -> Vec<u64> {
        let dir = tempdir().unwrap();
        let opts = ExtMemOpts {
            subproblem_count: p,
            work_dir: dir.path().to_path_buf(),
            ..ExtMemOpts::default()
        };
        let mut out: Vec<u64> = Vec::with_capacity(positions.len());
        build_ext_mem_for_positions(text, positions, &opts, |pos| {
            out.push(pos);
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn ext_mem_for_positions_full_set_matches_ext_mem() {
        let text = b"mississippi";
        let want = ext_mem_sa(text, 3);
        let positions: Vec<u64> = (0..text.len() as u64).collect();
        let got = ext_mem_for_positions(text, positions, 3);
        assert_eq!(got, want);
    }

    #[test]
    fn ext_mem_for_positions_subset_matches_brute_force() {
        let text = b"mississippi";
        let positions: Vec<u64> = (0..text.len() as u64).step_by(2).collect();
        let mut want = positions.clone();
        want.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        let got = ext_mem_for_positions(text, positions, 4);
        assert_eq!(got, want);
    }

    #[test]
    fn ext_mem_for_positions_random_subsets() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0DE);
        for &n in &[50usize, 500, 2000] {
            for &p in &[1usize, 3, 8] {
                let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..5u8)).collect();
                let mut positions: Vec<u64> = (0..n as u64).collect();
                positions.retain(|_| rng.random_range(0..10) < 7);
                let mut want = positions.clone();
                want.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
                let got = ext_mem_for_positions(&text, positions, p);
                assert_eq!(got, want, "subset ext-mem mismatch n={n} p={p}");
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
