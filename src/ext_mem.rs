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
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::Index;
use crate::ext_bucket::{BucketPool, BucketRecord, BucketStore, InMemBucket, SaLcp};
use crate::lcp::{LcpDispatch, Symbol};
use crate::sample_sort;

/// Emit a phase-timing line to stderr if `CAPS_SA_PROFILE` is set in
/// the environment. Used to localise where the ext-mem path spends its
/// time without paying the cost of always logging — see
/// `bench/README.md` "Where AVX-512 helps and where it doesn't" for
/// how this is used.
fn profile_log(message: &str) {
    if std::env::var_os("CAPS_SA_PROFILE").is_some() {
        eprintln!("caps-sa profile  {message}");
    }
}

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
    /// Number of physical files in the bucket pool (one pool for the
    /// phase-1 subarray buckets and a second for the phase-3 partition
    /// buckets). `0` (default) picks `rayon::current_num_threads()` —
    /// the right answer in practice: one writable inode per worker
    /// keeps kernel-level write contention bounded.
    ///
    /// The `2 × p` logical buckets (typically thousands at genome
    /// scale) collapse onto this pool of anonymous tempfiles via
    /// `bucket_id % physical_file_count`. Larger values lower kernel
    /// write contention; smaller values are kinder to networked
    /// filesystems with high metadata cost. The `CAPS_SA_N_PHYS` env
    /// var overrides this for one-off benches.
    pub physical_file_count: usize,
}

impl Default for ExtMemOpts {
    fn default() -> Self {
        Self {
            max_context: usize::MAX,
            subproblem_count: 0,
            work_dir: std::env::temp_dir(),
            physical_file_count: 0,
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
    S: Symbol,
    F: FnMut(u64) -> io::Result<()>,
{
    // Dispatch on text size: when every suffix position fits in `u32`
    // (n ≤ 2^32), use the narrow record type to halve all the SaLcp
    // bytes — bucket disk I/O, phase-1 records, and the per-partition
    // load in phase 4.
    if text.len() <= u32::MAX as usize + 1 {
        build_ext_mem_inner::<S, u32, F>(text, PositionSource::Identity(text.len()), opts, emit)
    } else {
        build_ext_mem_inner::<S, u64, F>(text, PositionSource::Identity(text.len()), opts, emit)
    }
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
    S: Symbol,
    F: FnMut(u64) -> io::Result<()>,
{
    // We hold a reference to `positions` for the duration of the build;
    // `phase1_sort_sample_spill` copies each chunk out. The Vec is
    // dropped once phase 1 returns.
    if text.len() <= u32::MAX as usize + 1 {
        build_ext_mem_inner::<S, u32, F>(text, PositionSource::Subset(&positions), opts, emit)
    } else {
        build_ext_mem_inner::<S, u64, F>(text, PositionSource::Subset(&positions), opts, emit)
    }
}

fn build_ext_mem_inner<S, I, F>(
    text: &[S],
    source: PositionSource<'_>,
    opts: &ExtMemOpts,
    mut emit: F,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    SaLcp<I>: BucketRecord,
    F: FnMut(u64) -> io::Result<()>,
{
    let n = source.len();
    if n == 0 {
        return Ok(());
    }
    let p = effective_subproblem_count(n, opts.subproblem_count);
    let dispatch = LcpDispatch::detect();
    let work_dir = opts.work_dir.clone();

    // Pool the `2 × p` bucket files into one anonymous tempfile per
    // worker thread. With `p` in the thousands and `num_threads` in
    // the dozens this collapses the openat/close/unlink budget from
    // ~3·p (per-bucket-file path) to N (per-worker-file pool),
    // eliminating the metadata-syscall pain on networked filesystems
    // and the open-file-handle limit headache on tiny inputs. Per-
    // bucket in-memory buffers and write volumes are unchanged, so
    // local-disk wall time is neutral or marginally improved. See
    // `bench/README.md` for the empirical sizing.
    let n_phys = effective_physical_file_count(opts.physical_file_count);
    let phase1_pool = BucketPool::new(n_phys, &work_dir)?;
    let phase3_pool = BucketPool::new(n_phys, &work_dir)?;

    profile_log(&format!(
        "build_ext_mem n={n} p={p} index_width={}b n_phys={n_phys}",
        std::mem::size_of::<I>() * 8
    ));

    let sub_factory = |i: usize| phase1_pool.new_bucket::<SaLcp<I>>(i);
    let part_factory = |j: usize| phase3_pool.new_bucket::<SaLcp<I>>(j);

    let t = Instant::now();
    let (mut subarray_buckets, samples) =
        phase1_sort_sample_spill::<S, I, _, _>(text, &source, p, opts, dispatch, sub_factory)?;
    profile_log(&format!(
        "phase1 (sort+sample+spill) {:.3}s",
        t.elapsed().as_secs_f64()
    ));

    let t = Instant::now();
    let pivots = phase2_select_pivots::<S, I>(text, samples, p, opts.max_context, dispatch);
    profile_log(&format!(
        "phase2 (select pivots)      {:.3}s",
        t.elapsed().as_secs_f64()
    ));

    let t = Instant::now();
    let mut partition_buckets = phase3_distribute::<S, I, _, _>(
        text,
        &mut subarray_buckets,
        &pivots,
        p,
        opts,
        dispatch,
        part_factory,
    )?;
    profile_log(&format!(
        "phase3 (distribute)          {:.3}s",
        t.elapsed().as_secs_f64()
    ));

    drop(subarray_buckets);

    let t = Instant::now();
    let result = phase4_merge_and_emit::<S, I, _, F>(
        text,
        &mut partition_buckets,
        opts.max_context,
        &mut emit,
        dispatch,
    );
    profile_log(&format!(
        "phase4 (merge+emit)          {:.3}s",
        t.elapsed().as_secs_f64()
    ));
    result
}

/// Same algorithm as [`build_ext_mem_inner`] but with the disk-backed
/// [`ExtMemBucket`] replaced by [`InMemBucket`] throughout — phase 1
/// sorts each subarray and keeps the result in a `Vec<SaLcp<I>>`,
/// phase 3 distributes into in-RAM partition Vecs, phase 4 cascade-
/// merges the in-RAM partitions. No disk I/O.
///
/// Trades RAM for wall time: peak memory is ~`n × sizeof(SaLcp<I>)`
/// (the post-phase-1 records sitting around until phase 3 consumes
/// them), so ~25 GB on the human genome with `I = u32`. In exchange,
/// the disk-spill / distribute-write / partition-load round-trip is
/// gone — useful on machines with enough RAM to hold the working set.
fn build_in_memory_ss_inner<S, I, F>(
    text: &[S],
    source: PositionSource<'_>,
    opts: &ExtMemOpts,
    mut emit: F,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    SaLcp<I>: BucketRecord,
    F: FnMut(u64) -> io::Result<()>,
{
    let n = source.len();
    if n == 0 {
        return Ok(());
    }
    let p = effective_subproblem_count(n, opts.subproblem_count);
    let dispatch = LcpDispatch::detect();

    let factory = |_i: usize| InMemBucket::<SaLcp<I>>::new();

    let (mut subarray_buckets, samples) =
        phase1_sort_sample_spill::<S, I, _, _>(text, &source, p, opts, dispatch, factory)?;
    let pivots = phase2_select_pivots::<S, I>(text, samples, p, opts.max_context, dispatch);
    let mut partition_buckets = phase3_distribute::<S, I, _, _>(
        text,
        &mut subarray_buckets,
        &pivots,
        p,
        opts,
        dispatch,
        factory,
    )?;
    drop(subarray_buckets);
    phase4_merge_and_emit::<S, I, _, F>(
        text,
        &mut partition_buckets,
        opts.max_context,
        &mut emit,
        dispatch,
    )
}

/// In-memory variant of the sample-sort algorithm used by
/// [`build_ext_mem`]. Skips all disk I/O at the cost of holding the
/// (`pos`, `lcp`) records in RAM throughout. Picks `u32` records when
/// `n ≤ 2³²`, falls back to `u64` otherwise. The caller's `emit`
/// closure is called once per output position in lex order, just like
/// in the ext-mem path.
pub fn build_in_memory_sample_sort<S, F>(text: &[S], opts: &ExtMemOpts, emit: F) -> io::Result<()>
where
    S: Symbol,
    F: FnMut(u64) -> io::Result<()>,
{
    if text.len() <= u32::MAX as usize + 1 {
        build_in_memory_ss_inner::<S, u32, F>(
            text,
            PositionSource::Identity(text.len()),
            opts,
            emit,
        )
    } else {
        build_in_memory_ss_inner::<S, u64, F>(
            text,
            PositionSource::Identity(text.len()),
            opts,
            emit,
        )
    }
}

/// Subset-positions variant of [`build_in_memory_sample_sort`]. Same
/// shape as [`build_ext_mem_for_positions`].
pub fn build_in_memory_sample_sort_for_positions<S, F>(
    text: &[S],
    positions: Vec<u64>,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    F: FnMut(u64) -> io::Result<()>,
{
    if text.len() <= u32::MAX as usize + 1 {
        build_in_memory_ss_inner::<S, u32, F>(text, PositionSource::Subset(&positions), opts, emit)
    } else {
        build_in_memory_ss_inner::<S, u64, F>(text, PositionSource::Subset(&positions), opts, emit)
    }
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
    /// `[start, start + dst.len())`, narrowing the caller's `u64`
    /// positions into `I` via [`Index::from_usize`]. For
    /// [`PositionSource::Identity`] this generates the contiguous
    /// integer range on the fly; for [`PositionSource::Subset`] it
    /// reads from the caller's slice.
    fn fill_chunk<I: Index>(&self, start: usize, dst: &mut [I]) {
        match self {
            Self::Identity(_) => {
                for (i, slot) in dst.iter_mut().enumerate() {
                    *slot = I::from_usize(start + i);
                }
            }
            Self::Subset(p) => {
                let end = start + dst.len();
                for (slot, &v) in dst.iter_mut().zip(p[start..end].iter()) {
                    *slot = I::from_usize(v as usize);
                }
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
/// Hard cap on the number of subarrays. Matches upstream CaPS-SA's
/// default of 8192 — phase 3 is now parallelised across rayon
/// workers (each subarray distributes independently into per-partition
/// `Mutex<ExtMemBucket>` slots), so the `O(p²)` sequential distribute
/// of the original design no longer constrains us. The cap is still
/// finite to keep the temp-file count bounded.
const PHASE1_MAX_PARTITIONS: usize = 8192;

/// Resolve [`ExtMemOpts::physical_file_count`] for the current build.
/// `0` (the default) means "let the runtime decide"; we pick
/// `rayon::current_num_threads()` so the pool has one inode per
/// concurrent writer, which empirically matches per-bucket-file wall
/// time while collapsing thousands of small files into dozens of
/// large ones. The `CAPS_SA_N_PHYS` env var overrides at the call
/// site for benchmarks.
fn effective_physical_file_count(requested: usize) -> usize {
    if let Some(v) = std::env::var("CAPS_SA_N_PHYS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 1)
    {
        return v;
    }
    if requested >= 1 {
        return requested;
    }
    rayon::current_num_threads().max(1)
}

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
#[allow(clippy::too_many_arguments)]
fn phase1_sort_sample_spill<S, I, B, MkB>(
    text: &[S],
    source: &PositionSource<'_>,
    p: usize,
    opts: &ExtMemOpts,
    dispatch: LcpDispatch,
    mk_bucket: MkB,
) -> io::Result<(Vec<B>, Vec<I>)>
where
    S: Symbol,
    I: Index,
    SaLcp<I>: BucketRecord,
    B: BucketStore<SaLcp<I>> + Send,
    MkB: Fn(usize) -> B + Send + Sync,
{
    let n = source.len();
    let chunk_size = n.div_ceil(p);
    let samples_target_total = sample_target_total(n, p);

    let per_subarray: Vec<(B, Vec<I>)> = (0..p)
        .into_par_iter()
        .map(|i| {
            let start = (i * chunk_size).min(n);
            let end = ((i + 1) * chunk_size).min(n);
            let len = end - start;

            let mut bucket = mk_bucket(i);
            if len == 0 {
                return Ok::<_, io::Error>((bucket, Vec::new()));
            }

            // In-memory sort of this subarray with LCP maintenance.
            let mut sa: Vec<I> = vec![I::zero(); len];
            source.fill_chunk(start, &mut sa);
            let mut sa_w = vec![I::zero(); len];
            let mut lcp_arr = vec![I::zero(); len];
            let mut lcp_w = vec![I::zero(); len];
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
            let records: Vec<SaLcp<I>> = sa
                .iter()
                .zip(lcp_arr.iter())
                .map(|(&pos, &lcp)| SaLcp { pos, lcp })
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
fn phase2_select_pivots<S, I>(
    text: &[S],
    mut samples: Vec<I>,
    p: usize,
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> Vec<I>
where
    S: Symbol,
    I: Index,
{
    if p <= 1 || samples.is_empty() {
        return Vec::new();
    }
    let n_samples = samples.len();
    let mut sa_w = vec![I::zero(); n_samples];
    let mut lcp = vec![I::zero(); n_samples];
    let mut lcp_w = vec![I::zero(); n_samples];
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

/// Phase 3: walk each subarray *in parallel*, load it into RAM,
/// binary-search the pivots to find its `p` sub-subarray boundaries,
/// and append each sub-subarray to the corresponding partition bucket.
///
/// Partition buckets are wrapped in a [`Mutex`] each so multiple
/// threads can write to different partitions concurrently without
/// shard-merging afterwards. With `p` in the thousands and `T` in the
/// tens, lock contention is negligible (probability that two threads
/// want the same partition at the same instant is `~T/p`); the lock
/// scope per acquisition is one `add_slice` + `mark_boundary` of a
/// few-KB sub-subarray.
///
/// Phase 4 doesn't care about the relative order of sub-subarrays
/// within a partition — only that each one between consecutive
/// boundaries is internally sorted. Both properties hold under
/// arbitrary thread interleaving.
#[allow(clippy::too_many_arguments)]
fn phase3_distribute<S, I, B, MkB>(
    text: &[S],
    subarray_buckets: &mut [B],
    pivots: &[I],
    p: usize,
    opts: &ExtMemOpts,
    dispatch: LcpDispatch,
    mk_bucket: MkB,
) -> io::Result<Vec<B>>
where
    S: Symbol,
    I: Index,
    SaLcp<I>: BucketRecord,
    B: BucketStore<SaLcp<I>> + Send,
    MkB: Fn(usize) -> B + Send + Sync,
{
    let _ = opts; // work_dir is used only by the ext-mem factory closure now
    let partition_buckets: Vec<Mutex<B>> = (0..p).map(|j| Mutex::new(mk_bucket(j))).collect();

    subarray_buckets
        .par_iter_mut()
        .try_for_each(|sub_bucket| -> io::Result<()> {
            if sub_bucket.total_records() == 0 {
                return Ok(());
            }
            let records = sub_bucket.load_all()?;

            // Find p-1 split points by binary-searching each pivot's
            // *upper bound* in the sorted subarray.
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

            // Distribute each sub-subarray. Reset the first record's
            // `lcp` to 0 so the per-partition merge sees a well-formed
            // boundary.
            for j in 0..p {
                let lo = splits[j];
                let hi = splits[j + 1];
                if lo >= hi {
                    continue;
                }
                let mut sub: Vec<SaLcp<I>> = records[lo..hi].to_vec();
                sub[0].lcp = I::zero();
                let mut bucket = partition_buckets[j].lock().unwrap();
                bucket.add_slice(&sub)?;
                bucket.mark_boundary();
            }
            Ok(())
        })?;

    // Unwrap the Mutexes — at this point only this thread holds
    // references, so the locks are uncontended.
    Ok(partition_buckets
        .into_iter()
        .map(|m| m.into_inner().expect("partition mutex poisoned"))
        .collect())
}

/// Upper-bound binary search: returns the first index `i` such that the
/// suffix at `records[i].pos` is **strictly greater than** the suffix at
/// `pivot`.
fn upper_bound_by_pivot<S, I>(
    records: &[SaLcp<I>],
    pivot: I,
    text: &[S],
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> usize
where
    S: Symbol,
    I: Index,
{
    let mut lo = 0;
    let mut hi = records.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match dispatch.suffix_cmp(text, records[mid].pos.to_usize(), pivot.to_usize(), max_ctx) {
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
fn phase4_merge_and_emit<S, I, B, F>(
    text: &[S],
    partition_buckets: &mut [B],
    max_ctx: usize,
    emit: &mut F,
    dispatch: LcpDispatch,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    SaLcp<I>: BucketRecord,
    B: BucketStore<SaLcp<I>> + Send,
    F: FnMut(u64) -> io::Result<()>,
{
    let n_partitions = partition_buckets.len();
    if n_partitions == 0 {
        return Ok(());
    }
    // `chunk_size = 4 × num_threads` (not `= num_threads`): with one
    // partition per thread per chunk, rayon's `par_iter_mut` assigns
    // 1-to-1 with no opportunity to steal, and the chunk's wall is
    // set by its slowest partition. Sample-sort partition sizes vary
    // ~2× from random sampling, so the slow tail leaves ~half the
    // cores idle waiting (observed: 52% parallel efficiency on
    // GRCh38 / 32 t).
    //
    // Bumping the chunk to `4 × num_threads` gives rayon four
    // partitions per thread to dispatch — fast threads can steal from
    // slow neighbours, smoothing out the size variance. Peak RAM
    // grows linearly: each in-flight merged partition holds its
    // result `Vec<I>` (~3 MB at human-genome scale with `u32`
    // indices), so the chunk's transient cost goes from `32 × 3 MB =
    // 96 MB` to `128 × 3 MB = 384 MB` — well within the budget we
    // already spend on phase 1.
    let chunk_size = rayon::current_num_threads().max(1) * 4;

    // Per-thread CPU-µs accumulators for the two parallel sub-steps. They
    // add across threads, so the printed values are CPU-time (sum), not
    // wall-time; the ratio between them still tells us where the work is.
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    let profile = std::env::var_os("CAPS_SA_PROFILE").is_some();
    let load_us = AtomicU64::new(0);
    let merge_us = AtomicU64::new(0);
    let mut emit_secs: f64 = 0.0;

    let mut start = 0;
    while start < n_partitions {
        let end = (start + chunk_size).min(n_partitions);
        let chunk = &mut partition_buckets[start..end];

        // Parallel-merge each non-empty bucket in this chunk. `par_iter_mut`
        // preserves index order in the collected `Vec`, so the subsequent
        // sequential emit yields positions in lex order.
        let merged: Vec<Vec<I>> = chunk
            .par_iter_mut()
            .map(|bucket| -> io::Result<Vec<I>> {
                if bucket.total_records() == 0 {
                    return Ok(Vec::new());
                }
                let t = Instant::now();
                let records = bucket.load_all()?;
                let boundaries: Vec<usize> = bucket.boundaries().to_vec();
                if profile {
                    load_us.fetch_add(t.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
                }

                let t = Instant::now();
                let workspace = CascadeWorkspace::<I>::new();
                // `cascade_merge` consumes the workspace and returns
                // the result side directly — the other three buffers
                // drop along with `workspace` here, without an
                // intermediate `to_vec()` copy.
                let result =
                    workspace.cascade_merge(text, &records, &boundaries, max_ctx, dispatch);
                if profile {
                    merge_us.fetch_add(t.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
                }
                Ok(result)
            })
            .collect::<Result<Vec<_>, io::Error>>()?;

        let t = Instant::now();
        for positions in merged {
            for pos in positions {
                // Widen back to the public `u64` emit contract.
                emit(pos.to_usize() as u64)?;
            }
        }
        if profile {
            emit_secs += t.elapsed().as_secs_f64();
        }

        start = end;
    }
    if profile {
        profile_log(&format!(
            "phase4 breakdown CPU: load {:.3}s merge {:.3}s; wall emit {:.3}s",
            load_us.load(AtomicOrdering::Relaxed) as f64 * 1e-6,
            merge_us.load(AtomicOrdering::Relaxed) as f64 * 1e-6,
            emit_secs,
        ));
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
struct CascadeWorkspace<I> {
    a_sa: Vec<I>,
    a_lcp: Vec<I>,
    b_sa: Vec<I>,
    b_lcp: Vec<I>,
}

impl<I: Index> CascadeWorkspace<I> {
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
            self.a_sa.resize(n, I::zero());
            self.a_lcp.resize(n, I::zero());
            self.b_sa.resize(n, I::zero());
            self.b_lcp.resize(n, I::zero());
        }
    }

    /// Cascade 2-way LCP-enhanced merges across the sub-subarrays of one
    /// partition (delimited by `boundaries`) until a single sorted run
    /// remains. **Consumes the workspace** and returns the result side
    /// as a `Vec<u64>`; the other three buffers (`a_lcp`, the opposing
    /// `*_sa`, the opposing `*_lcp`) drop immediately. This shape lets
    /// the caller skip the per-partition `to_vec()` round-trip that
    /// would otherwise sit briefly alongside all four workspace buffers
    /// at peak.
    fn cascade_merge<S>(
        mut self,
        text: &[S],
        records: &[SaLcp<I>],
        boundaries: &[usize],
        max_ctx: usize,
        dispatch: LcpDispatch,
    ) -> Vec<I>
    where
        S: Symbol,
    {
        let n = records.len();
        if n == 0 {
            return Vec::new();
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

        // Take ownership of the buffer holding the result, truncate to
        // the actual record count, drop the other three buffers with
        // `self` going out of scope.
        let mut result = if src_is_a { self.a_sa } else { self.b_sa };
        result.truncate(n);
        result
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
        S: Symbol,
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

    fn in_memory_sample_sort(text: &[u8], p: usize) -> Vec<u64> {
        let dir = tempdir().unwrap();
        let opts = ExtMemOpts {
            subproblem_count: p,
            work_dir: dir.path().to_path_buf(),
            ..ExtMemOpts::default()
        };
        let mut out: Vec<u64> = Vec::with_capacity(text.len());
        build_in_memory_sample_sort(text, &opts, |pos| {
            out.push(pos);
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn in_memory_sample_sort_matches_in_memory() {
        for text in [b"banana" as &[u8], b"mississippi", b"abracadabra"] {
            let want: Vec<u32> = build_in_memory(text);
            let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
            let got = in_memory_sample_sort(text, 0);
            assert_eq!(got, want64, "in-mem sample-sort mismatch on {text:?}");
        }
    }

    #[test]
    fn in_memory_sample_sort_random_byte_texts() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0DE_C0DE);
        for &n in &[16usize, 200, 2000] {
            for &p in &[1usize, 4, 16] {
                let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
                let want: Vec<u32> = build_in_memory(&text);
                let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
                let got = in_memory_sample_sort(&text, p);
                assert_eq!(got, want64, "in-mem ss mismatch n={n} p={p}");
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
