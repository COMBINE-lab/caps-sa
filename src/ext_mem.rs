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
use crate::limits::{LimitProvider, PlainText};
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
    build_ext_mem_with(text, &PlainText::new(text.len()), opts, emit)
}

/// Variant of [`build_ext_mem`] that accepts a [`LimitProvider`]. With
/// [`PlainText`] this matches [`build_ext_mem`] exactly (and
/// monomorphizes to identical assembly); with
/// [`SegmentedText`][crate::limits::SegmentedText] the LCP scans stop
/// at segment boundaries.
pub fn build_ext_mem_with<S, L, F>(text: &[S], lp: &L, opts: &ExtMemOpts, emit: F) -> io::Result<()>
where
    S: Symbol,
    L: LimitProvider,
    F: FnMut(u64) -> io::Result<()>,
{
    // Dispatch on text size: when every suffix position fits in `u32`
    // (n ≤ 2^32), use the narrow record type to halve all the SaLcp
    // bytes — bucket disk I/O, phase-1 records, and the per-partition
    // load in phase 4.
    if text.len() <= u32::MAX as usize + 1 {
        build_ext_mem_inner::<S, u32, L, F>(
            text,
            PositionSource::Identity(text.len()),
            lp,
            opts,
            emit,
        )
    } else {
        build_ext_mem_inner::<S, u64, L, F>(
            text,
            PositionSource::Identity(text.len()),
            lp,
            opts,
            emit,
        )
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
    build_ext_mem_for_positions_with(text, positions, &PlainText::new(text.len()), opts, emit)
}

/// Variant of [`build_ext_mem_for_positions`] that accepts a
/// [`LimitProvider`]. See [`build_ext_mem_with`] for the semantics.
pub fn build_ext_mem_for_positions_with<S, L, F>(
    text: &[S],
    positions: Vec<u64>,
    lp: &L,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    L: LimitProvider,
    F: FnMut(u64) -> io::Result<()>,
{
    // We hold a reference to `positions` for the duration of the build;
    // `phase1_sort_sample_spill` copies each chunk out. The Vec is
    // dropped once phase 1 returns.
    if text.len() <= u32::MAX as usize + 1 {
        build_ext_mem_inner::<S, u32, L, F>(
            text,
            PositionSource::Subset(&positions),
            lp,
            opts,
            emit,
        )
    } else {
        build_ext_mem_inner::<S, u64, L, F>(
            text,
            PositionSource::Subset(&positions),
            lp,
            opts,
            emit,
        )
    }
}

/// Like [`build_ext_mem_for_positions`] but takes a **predicate** over
/// text positions instead of a pre-materialised `Vec<u64>` of kept
/// positions.
///
/// caps-sa walks the predicate **once** to build a bitmap of kept
/// positions + a tiny per-block popcount prefix-sum (together ~`n / 8`
/// bytes — ~770 MB on the human genome, vs the ~50 GB the equivalent
/// `Vec<u64>` would take). Phase 1's per-subarray fill is then driven
/// by popcount-walking the bitmap; the predicate is **never invoked
/// again** after the initial build. See [`FilteredSource`] for the
/// memory accounting and the inner loop.
///
/// Use this entry when the caller already has the text in RAM and
/// the kept positions are described by a cheap per-position
/// predicate (e.g. STAR's `text[p] < 4` for ACGT-only suffix
/// sampling). It is **the right entry for genome-scale inputs** —
/// the `Vec<u64>` path can dominate peak RSS otherwise.
///
/// `keep` is invoked from rayon worker threads in parallel during
/// the bitmap build; it must be `Send + Sync` (typically a plain
/// closure capturing only `&[u8]` references is fine).
pub fn build_ext_mem_for_filter<S, F, Pred>(
    text: &[S],
    keep: Pred,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    F: FnMut(u64) -> io::Result<()>,
    Pred: Fn(u64) -> bool + Send + Sync,
{
    build_ext_mem_for_filter_with(text, keep, &PlainText::new(text.len()), opts, emit)
}

/// Variant of [`build_ext_mem_for_filter`] that accepts a
/// [`LimitProvider`]. See [`build_ext_mem_with`] for the semantics.
pub fn build_ext_mem_for_filter_with<S, L, F, Pred>(
    text: &[S],
    keep: Pred,
    lp: &L,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    L: LimitProvider,
    F: FnMut(u64) -> io::Result<()>,
    Pred: Fn(u64) -> bool + Send + Sync,
{
    let filtered = FilteredSource::new(text.len(), keep);
    if text.len() <= u32::MAX as usize + 1 {
        build_ext_mem_inner::<S, u32, L, F>(
            text,
            PositionSource::Filtered(filtered),
            lp,
            opts,
            emit,
        )
    } else {
        build_ext_mem_inner::<S, u64, L, F>(
            text,
            PositionSource::Filtered(filtered),
            lp,
            opts,
            emit,
        )
    }
}

fn build_ext_mem_inner<S, I, L, F>(
    text: &[S],
    source: PositionSource<'_>,
    lp: &L,
    opts: &ExtMemOpts,
    mut emit: F,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
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
    let (mut subarray_buckets, samples) = phase1_sort_sample_spill::<S, I, L, _, _>(
        text,
        lp,
        &source,
        p,
        opts,
        dispatch,
        sub_factory,
    )?;
    profile_log(&format!(
        "phase1 (sort+sample+spill) {:.3}s",
        t.elapsed().as_secs_f64()
    ));

    // Drop the position source as soon as phase 1 returns — phases
    // 2/3/4 don't touch it. For `PositionSource::Subset` this frees
    // the caller's `Vec<u64>` (e.g. ~47 GB on a human-scale
    // _for_positions build); for `PositionSource::Filtered` it
    // frees the bitmap + cumsum (~770 MB); for `Identity` it's a
    // no-op. The text and the spilled `subarray_buckets` are all
    // phase 2+ needs.
    drop(source);

    let t = Instant::now();
    let pivots = phase2_select_pivots::<S, I, L>(text, lp, samples, p, opts.max_context, dispatch);
    profile_log(&format!(
        "phase2 (select pivots)      {:.3}s",
        t.elapsed().as_secs_f64()
    ));

    let t = Instant::now();
    let mut partition_buckets = phase3_distribute::<S, I, L, _, _>(
        text,
        lp,
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
    let result = phase4_merge_and_emit::<S, I, L, _, F>(
        text,
        lp,
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
fn build_in_memory_ss_inner<S, I, L, F>(
    text: &[S],
    source: PositionSource<'_>,
    lp: &L,
    opts: &ExtMemOpts,
    mut emit: F,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
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
        phase1_sort_sample_spill::<S, I, L, _, _>(text, lp, &source, p, opts, dispatch, factory)?;
    // Same rationale as in `build_ext_mem_inner` — drop the source
    // as soon as phase 1's `fill_chunk` calls have stopped.
    drop(source);
    let pivots = phase2_select_pivots::<S, I, L>(text, lp, samples, p, opts.max_context, dispatch);
    let mut partition_buckets = phase3_distribute::<S, I, L, _, _>(
        text,
        lp,
        &mut subarray_buckets,
        &pivots,
        p,
        opts,
        dispatch,
        factory,
    )?;
    drop(subarray_buckets);
    phase4_merge_and_emit::<S, I, L, _, F>(
        text,
        lp,
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
    build_in_memory_sample_sort_with(text, &PlainText::new(text.len()), opts, emit)
}

/// Variant of [`build_in_memory_sample_sort`] that accepts a
/// [`LimitProvider`].
pub fn build_in_memory_sample_sort_with<S, L, F>(
    text: &[S],
    lp: &L,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    L: LimitProvider,
    F: FnMut(u64) -> io::Result<()>,
{
    if text.len() <= u32::MAX as usize + 1 {
        build_in_memory_ss_inner::<S, u32, L, F>(
            text,
            PositionSource::Identity(text.len()),
            lp,
            opts,
            emit,
        )
    } else {
        build_in_memory_ss_inner::<S, u64, L, F>(
            text,
            PositionSource::Identity(text.len()),
            lp,
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
    build_in_memory_sample_sort_for_positions_with(
        text,
        positions,
        &PlainText::new(text.len()),
        opts,
        emit,
    )
}

/// Variant of [`build_in_memory_sample_sort_for_positions`] that
/// accepts a [`LimitProvider`].
pub fn build_in_memory_sample_sort_for_positions_with<S, L, F>(
    text: &[S],
    positions: Vec<u64>,
    lp: &L,
    opts: &ExtMemOpts,
    emit: F,
) -> io::Result<()>
where
    S: Symbol,
    L: LimitProvider,
    F: FnMut(u64) -> io::Result<()>,
{
    if text.len() <= u32::MAX as usize + 1 {
        build_in_memory_ss_inner::<S, u32, L, F>(
            text,
            PositionSource::Subset(&positions),
            lp,
            opts,
            emit,
        )
    } else {
        build_in_memory_ss_inner::<S, u64, L, F>(
            text,
            PositionSource::Subset(&positions),
            lp,
            opts,
            emit,
        )
    }
}

/// Source of the positions to sort.
///
/// - [`PositionSource::Identity`] is the all-suffixes case `0..n`;
///   avoids materialising a `Vec<u64>` of length `n`, which on the
///   human genome would itself be ~50 GB.
/// - [`PositionSource::Subset`] holds a caller-supplied `&[u64]` of
///   the positions to sort, in any order. Random-access by index.
/// - [`PositionSource::Filtered`] is the streaming-predicate variant:
///   the caller hands in a `Fn(u64) -> bool` over text positions and
///   caps-sa walks the filter forward through the text on demand. A
///   tiny prefix-sum (`text.len() / BLOCK_SIZE × 8` B ≈ 760 KB on
///   the human genome at the 64 KB block size) lets `fill_chunk`
///   locate the i-th kept position in `O(log n_blocks + BLOCK_SIZE +
///   chunk_size)`. **No kept-positions list is materialised** — the
///   memory saving over `Subset` is `≈ 8 × n_kept` bytes, dominant
///   on genome-scale inputs.
enum PositionSource<'a> {
    Identity(usize),
    Subset(&'a [u64]),
    Filtered(FilteredSource),
}

/// Block size in **u64 words** for [`FilteredSource`]'s popcount
/// prefix-sum. With 1024 words per block (64 K text bits) the
/// prefix-sum stores one `u64` per block — ~760 KB for the human
/// genome — and each `fill_chunk` walks at most one block (≈ 8 KB)
/// in cache before emitting the first kept position.
const FILTERED_WORDS_PER_BLOCK: usize = 1024;

/// Streaming position source backed by a **bitmap** of kept positions
/// + a per-block popcount prefix-sum. No `Vec<u64>` of kept positions
/// is ever materialised.
///
/// Memory: `(n + 7) / 8` bytes for the bitmap (~770 MB on the human
/// genome at `n ≈ 6.2 B`) + `8 × n_blocks` for the prefix sum
/// (~760 KB). The 47 GB `Vec<u64>` the [`Subset`][PositionSource::Subset]
/// variant requires goes away entirely.
///
/// Lookup path inside [`fill_chunk`]:
/// 1. `partition_point` the prefix-sum to find the block containing
///    the `start`-th kept position (`O(log n_blocks)` ≈ 17 ops).
/// 2. Walk that block's bitmap words `popcount`-by-`popcount` until
///    we've skipped the right number of set bits to reach `start`.
/// 3. Walk forward through bitmap words using `trailing_zeros`-style
///    iteration, emitting each set bit's position to `dst`. Inner
///    loop is `O(chunk_size / 64)` u64 ops — no per-text-position
///    closure calls, branch-light, cache-resident.
///
/// A truly `O(1)` select-1 structure (darray / Elias-Fano) on top
/// of the bitmap would shrink step (1)+(2) further; with
/// `chunk_size` ≈ 750 K and only `p` ≈ 8192 fill_chunk calls per
/// build, the `O(log n_blocks)` + at-most-one-block-walk cost is
/// already a rounding error. See `bench/README.md` for the
/// follow-up note if that ever changes.
struct FilteredSource {
    text_len: usize,
    total_kept: usize,
    /// Bitmap: `(bitmap[w] >> b) & 1 == 1` iff position `64 * w + b`
    /// is kept. Length = `text_len.div_ceil(64)`.
    bitmap: Vec<u64>,
    /// `cumsum[i] = sum of set bits in
    /// `bitmap[0 .. i * FILTERED_WORDS_PER_BLOCK]`. Length =
    /// `n_blocks + 1`; the last entry equals `total_kept`.
    cumsum: Vec<u64>,
}

impl FilteredSource {
    /// Build a [`FilteredSource`] by walking the predicate once over
    /// `0..text_len` to fill the bitmap, then accumulating per-block
    /// popcounts.
    ///
    /// The predicate is invoked exactly `text_len` times here; once
    /// the bitmap is built, `fill_chunk` never calls it again. This
    /// trades one full predicate pass for zero per-position calls
    /// during all subsequent random-access `fill_chunk`s — a clear
    /// win when (as in caps-sa's phase 1) every position is read at
    /// least once.
    fn new<Pred>(text_len: usize, keep: Pred) -> Self
    where
        Pred: Fn(u64) -> bool + Send + Sync,
    {
        let n_words = text_len.div_ceil(64);
        // Parallel per-word bitmap build. Each word reads 64 text
        // positions (clamped at `text_len`), packs them into a u64.
        let bitmap: Vec<u64> = (0..n_words)
            .into_par_iter()
            .map(|w| {
                let mut word: u64 = 0;
                let base = (w as u64) * 64;
                let limit = ((w + 1) * 64).min(text_len) - w * 64;
                for b in 0..limit {
                    if keep(base + b as u64) {
                        word |= 1u64 << b;
                    }
                }
                word
            })
            .collect();

        // Per-block popcount cumsum. Each block covers
        // `FILTERED_WORDS_PER_BLOCK` words = `FILTERED_BITS_PER_BLOCK`
        // text positions.
        let n_blocks = n_words.div_ceil(FILTERED_WORDS_PER_BLOCK);
        let per_block: Vec<u64> = (0..n_blocks)
            .into_par_iter()
            .map(|i| {
                let start = i * FILTERED_WORDS_PER_BLOCK;
                let end = ((i + 1) * FILTERED_WORDS_PER_BLOCK).min(n_words);
                let mut c: u64 = 0;
                for &word in &bitmap[start..end] {
                    c += word.count_ones() as u64;
                }
                c
            })
            .collect();
        let mut cumsum = Vec::with_capacity(n_blocks + 1);
        let mut s: u64 = 0;
        cumsum.push(0);
        for &k in &per_block {
            s += k;
            cumsum.push(s);
        }
        let total_kept = s as usize;
        Self {
            text_len,
            total_kept,
            bitmap,
            cumsum,
        }
    }

    /// Number of kept positions.
    #[inline]
    fn len(&self) -> usize {
        self.total_kept
    }

    /// Fill `dst` with the next `dst.len()` kept positions starting
    /// from the `start`-th (0-based) kept position. See type-level
    /// doc for the algorithm; this is the hot path during phase 1
    /// fill_chunk.
    fn fill_chunk<I: Index>(&self, start: usize, dst: &mut [I]) {
        debug_assert!(start + dst.len() <= self.total_kept);
        if dst.is_empty() {
            return;
        }

        // (1) Locate the block containing the `start`-th set bit.
        // `partition_point(|c| c <= start)` gives the first cumsum
        // entry strictly greater than `start`; previous index is the
        // containing block.
        let pp = self.cumsum.partition_point(|&c| c <= start as u64);
        debug_assert!(pp > 0);
        let block_idx = pp - 1;
        let mut word_idx = block_idx * FILTERED_WORDS_PER_BLOCK;
        let mut skip = start as u64 - self.cumsum[block_idx];

        // (2) Skip the first `skip` set bits — possibly spanning
        // several bitmap words. Whole words with `popcount ≤ skip`
        // are consumed wholesale; the final partial word has its
        // lowest `skip` set bits cleared so the emit loop sees only
        // un-skipped 1s.
        //
        // Note: a naive `while skip >= 64 { … }` is wrong because a
        // word's popcount can be far less than 64; we must subtract
        // the actual popcount each iteration, not 64. This matters
        // any time the bitmap is sparser than ~50%.
        let n_words = self.bitmap.len();
        let mut word: u64 = if word_idx < n_words { self.bitmap[word_idx] } else { 0 };
        while skip > 0 {
            let pc = word.count_ones() as u64;
            if skip < pc {
                // Consume `skip` lowest set bits inside the current
                // word; emit loop continues from the remaining ones.
                for _ in 0..skip {
                    word &= word - 1;
                }
                break;
            }
            // Skip ≥ pc: consume the whole word and advance.
            skip -= pc;
            word_idx += 1;
            word = if word_idx < n_words { self.bitmap[word_idx] } else { 0 };
        }

        // (3) Walk `word`+subsequent words, emitting one position per
        // set bit. Uses `trailing_zeros` to jump straight to the next
        // 1 inside a word, then clears it via `word &= word - 1`.
        let mut written = 0usize;
        let need = dst.len();
        loop {
            while word != 0 && written < need {
                let bit = word.trailing_zeros() as u64;
                let pos = (word_idx as u64) * 64 + bit;
                debug_assert!((pos as usize) < self.text_len);
                dst[written] = I::from_usize(pos as usize);
                written += 1;
                word &= word - 1;
            }
            if written == need {
                break;
            }
            word_idx += 1;
            debug_assert!(
                word_idx < n_words,
                "FilteredSource::fill_chunk: walked past bitmap end \
                 ({written}/{need} emitted, word_idx={word_idx}, n_words={n_words})"
            );
            word = self.bitmap[word_idx];
        }
    }
}

impl<'a> PositionSource<'a> {
    fn len(&self) -> usize {
        match self {
            Self::Identity(n) => *n,
            Self::Subset(p) => p.len(),
            Self::Filtered(f) => f.len(),
        }
    }

    /// Fill `dst` with positions for the half-open subarray range
    /// `[start, start + dst.len())`, narrowing the caller's `u64`
    /// positions into `I` via [`Index::from_usize`]. For
    /// [`PositionSource::Identity`] this generates the contiguous
    /// integer range on the fly; for [`PositionSource::Subset`] it
    /// reads from the caller's slice; for [`PositionSource::Filtered`]
    /// it walks the predicate forward from the right text block.
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
            Self::Filtered(f) => f.fill_chunk(start, dst),
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
fn phase1_sort_sample_spill<S, I, L, B, MkB>(
    text: &[S],
    lp: &L,
    source: &PositionSource<'_>,
    p: usize,
    opts: &ExtMemOpts,
    dispatch: LcpDispatch,
    mk_bucket: MkB,
) -> io::Result<(Vec<B>, Vec<I>)>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
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
                lp,
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
fn phase2_select_pivots<S, I, L>(
    text: &[S],
    lp: &L,
    mut samples: Vec<I>,
    p: usize,
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> Vec<I>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
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
        lp,
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
fn phase3_distribute<S, I, L, B, MkB>(
    text: &[S],
    lp: &L,
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
    L: LimitProvider,
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
                    lp,
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
fn upper_bound_by_pivot<S, I, L>(
    records: &[SaLcp<I>],
    pivot: I,
    text: &[S],
    lp: &L,
    max_ctx: usize,
    dispatch: LcpDispatch,
) -> usize
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
{
    let mut lo = 0;
    let mut hi = records.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match dispatch.suffix_cmp_with(
            text,
            lp,
            records[mid].pos.to_usize(),
            pivot.to_usize(),
            max_ctx,
        ) {
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
fn phase4_merge_and_emit<S, I, L, B, F>(
    text: &[S],
    lp: &L,
    partition_buckets: &mut [B],
    max_ctx: usize,
    emit: &mut F,
    dispatch: LcpDispatch,
) -> io::Result<()>
where
    S: Symbol,
    I: Index,
    L: LimitProvider,
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
                    workspace.cascade_merge(text, lp, &records, &boundaries, max_ctx, dispatch);
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
    fn cascade_merge<S, L>(
        mut self,
        text: &[S],
        lp: &L,
        records: &[SaLcp<I>],
        boundaries: &[usize],
        max_ctx: usize,
        dispatch: LcpDispatch,
    ) -> Vec<I>
    where
        S: Symbol,
        L: LimitProvider,
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
            run_lens = self.merge_one_level(src_is_a, &run_lens, text, lp, max_ctx, dispatch);
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
    fn merge_one_level<S, L>(
        &mut self,
        src_is_a: bool,
        run_lens: &[usize],
        text: &[S],
        lp: &L,
        max_ctx: usize,
        dispatch: LcpDispatch,
    ) -> Vec<usize>
    where
        S: Symbol,
        L: LimitProvider,
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
                    lp,
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

    /// Helper that drives [`build_ext_mem_for_filter`] and collects the
    /// emitted positions.
    fn ext_mem_for_filter<Pred>(text: &[u8], keep: Pred, p: usize) -> Vec<u64>
    where
        Pred: Fn(u64) -> bool + Send + Sync,
    {
        let dir = tempdir().unwrap();
        let opts = ExtMemOpts {
            subproblem_count: p,
            work_dir: dir.path().to_path_buf(),
            ..ExtMemOpts::default()
        };
        let mut out: Vec<u64> = Vec::new();
        build_ext_mem_for_filter(text, keep, &opts, |pos| {
            out.push(pos);
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn ext_mem_for_filter_matches_for_positions_on_full_set() {
        // Filter that accepts every position → must equal the
        // identity-positions ext-mem build.
        let text = b"mississippi";
        let want = ext_mem_sa(text, 3);
        let got = ext_mem_for_filter(text, |_p| true, 3);
        assert_eq!(got, want);
    }

    #[test]
    fn ext_mem_for_filter_matches_for_positions_on_dna_subset() {
        // STAR-style "keep ACGT (`< 4`), drop N (`4`)/spacer (`5`)"
        // filter. The filter API must produce exactly the same SA as
        // pre-materialising the kept positions and going through the
        // _for_positions path.
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCA75_5A);
        for &n in &[50usize, 500, 2000] {
            for &p in &[1usize, 3, 8] {
                let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
                let positions: Vec<u64> = (0..n as u64).filter(|&i| text[i as usize] < 4).collect();
                let want = ext_mem_for_positions(&text, positions, p);
                let got = ext_mem_for_filter(&text, |i| text[i as usize] < 4, p);
                assert_eq!(got, want, "filter vs positions mismatch n={n} p={p}");
            }
        }
    }

    #[test]
    fn ext_mem_for_filter_handles_block_aligned_boundaries() {
        // Exercise the bitmap word/block boundaries by using a text
        // longer than one popcount block (1024 × 64 bits = 64 K
        // positions) — but stay under that to keep the test fast.
        // 200 K positions touches the cumsum's second block too.
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xB10C_C0DE);
        let n = 200_000usize;
        let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..6u8)).collect();
        let positions: Vec<u64> = (0..n as u64).filter(|&i| text[i as usize] < 4).collect();
        let want = ext_mem_for_positions(&text, positions, 8);
        let got = ext_mem_for_filter(&text, |i| text[i as usize] < 4, 8);
        assert_eq!(got, want, "filter API mismatch across block boundaries");
    }

    #[test]
    fn ext_mem_for_filter_sparse_predicate() {
        // ~5% acceptance — exercises long runs of zero-bits in the
        // bitmap (skip-loop across whole words).
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5_AA_55);
        let n = 50_000usize;
        let text: Vec<u8> = (0..n).map(|_| rng.random_range(0..20u8)).collect();
        let positions: Vec<u64> = (0..n as u64).filter(|&i| text[i as usize] < 1).collect();
        let want = ext_mem_for_positions(&text, positions, 4);
        let got = ext_mem_for_filter(&text, |i| text[i as usize] < 1, 4);
        assert_eq!(got, want, "filter API mismatch on sparse predicate");
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
