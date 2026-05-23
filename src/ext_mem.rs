//! External-memory suffix array construction.
//!
//! The Phase 1 in-memory path holds the entire suffix array and several
//! work buffers in RAM (≈ 4n × sizeof(index) for the merge-sort ping-pong).
//! This module trades that for bounded RAM at the cost of disk traffic:
//!
//! 1. Split positions `0..n` into `p` subarrays of size `n / p`. Sort each
//!    in memory with the same merge-sort kernel as the in-memory path,
//!    then spill the sorted positions to a per-subarray
//!    [`ExtMemBucket`][crate::ext_bucket::ExtMemBucket].
//! 2. Stream a `p`-way merge by reading one head from each bucket through
//!    a [`BufReader`] and emitting the lexicographically smallest one via
//!    a caller-supplied closure.
//!
//! Peak RAM ≈ `text` + per-thread merge-sort scratch (`O(n/p)`) + a small
//! per-bucket read buffer. With the default `subproblem_count`, that's a
//! few percent of the input size — well below the in-memory path's
//! `~4 × n` working set.
//!
//! Phase 2 v1 uses a *linear scan* over `p` bucket heads to pick the next
//! emission (no heap, no LCP enhancement). For modest `p` (default = 4 ×
//! `rayon::current_num_threads()`) this is fine; a future Phase 2b will
//! upgrade to CaPS-SA's sample-sort partitioning + LCP-enhanced multi-way
//! merge to remove the `O(n · p)` factor for highly-parallel runs.

use std::cmp::Ordering;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::ext_bucket::ExtMemBucket;
use crate::lcp::suffix_cmp;
use crate::sample_sort;

/// Tunable options for [`build_ext_mem`].
#[derive(Clone, Debug)]
pub struct ExtMemOpts {
    /// Bound on LCP-extension comparisons inside the merge. `usize::MAX`
    /// (default) is unbounded.
    pub max_context: usize,
    /// Number of subarrays to split `0..n` into. `0` (default) picks a
    /// reasonable value based on `rayon::current_num_threads()`.
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
/// output index to `emit` in lexicographic order.
///
/// Returns an [`io::Error`] if the on-disk spilling fails. The callback
/// may also return an error to abort construction (the partial output is
/// discarded; temp files are cleaned up when their bucket drops).
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

    // Phase 1: sort each subarray in RAM, spill sorted positions to disk.
    let mut buckets = sort_and_spill_subarrays(text, n, p, opts)?;

    // Phase 2: stream p-way merge, emitting each output position via `emit`.
    streaming_p_way_merge(text, &mut buckets, opts.max_context, &mut emit)
}

fn effective_subproblem_count(n: usize, requested: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let raw = if requested == 0 {
        // Default: enough subarrays to keep each rayon worker busy and to
        // shrink the per-subarray working set, but not so many that the
        // O(p)-per-emit cost in the merge dominates. Four per thread is
        // the same heuristic upstream CaPS-SA uses.
        rayon::current_num_threads().saturating_mul(4)
    } else {
        requested
    };
    raw.clamp(1, n)
}

/// Phase 1: sort the `p` subarrays of `text` in parallel, spill each
/// sorted run to its own [`ExtMemBucket`].
fn sort_and_spill_subarrays<S>(
    text: &[S],
    n: usize,
    p: usize,
    opts: &ExtMemOpts,
) -> io::Result<Vec<ExtMemBucket<u64>>>
where
    S: Ord + Copy + Sync,
{
    let chunk_size = n.div_ceil(p);

    (0..p)
        .into_par_iter()
        .map(|i| {
            // When `n` isn't a clean multiple of `p`, trailing chunks may
            // start past `n`. Clamp both ends so `len` is always
            // non-negative.
            let start = (i * chunk_size).min(n);
            let end = ((i + 1) * chunk_size).min(n);
            let len = end - start;

            if len == 0 {
                // An empty trailing subarray (when n isn't divisible by p)
                // still gets a (logically empty) bucket so phase 2 sees a
                // uniform p-stream input.
                return Ok(ExtMemBucket::new(&opts.work_dir, format!("sub{i}")));
            }

            // Sort the positions [start..end) with the in-memory kernel.
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

            // Spill the sorted positions only. The LCP array is consumed
            // by the in-memory merge and discarded here — Phase 2 v1's
            // streaming merge doesn't carry LCP info.
            let mut bucket = ExtMemBucket::new(&opts.work_dir, format!("sub{i}"));
            bucket.add_slice(&sa)?;
            Ok::<_, io::Error>(bucket)
        })
        .collect()
}

/// Phase 2: stream a `p`-way merge from the per-subarray buckets, calling
/// `emit` for each output position in lex order.
fn streaming_p_way_merge<S, F>(
    text: &[S],
    buckets: &mut [ExtMemBucket<u64>],
    max_context: usize,
    emit: &mut F,
) -> io::Result<()>
where
    S: Ord,
    F: FnMut(u64) -> io::Result<()>,
{
    let p = buckets.len();
    let mut streams: Vec<Stream> = Vec::with_capacity(p);
    let mut heads: Vec<Option<u64>> = Vec::with_capacity(p);

    for bucket in buckets.iter_mut() {
        if bucket.total_records() == 0 {
            streams.push(Stream::Empty);
            heads.push(None);
            continue;
        }
        let mut reader = bucket.open_reader()?;
        let head = read_one_u64(&mut reader)?;
        streams.push(Stream::Active { reader });
        heads.push(Some(head));
    }

    loop {
        // Linear scan to find the lexicographically smallest active head.
        let mut min_idx: Option<usize> = None;
        for (i, head) in heads.iter().enumerate() {
            let Some(hp) = head else { continue };
            min_idx = Some(match min_idx {
                None => i,
                Some(j) => {
                    let jp = heads[j].unwrap();
                    match suffix_cmp(text, *hp as usize, jp as usize, max_context) {
                        Ordering::Less => i,
                        Ordering::Equal | Ordering::Greater => j,
                    }
                }
            });
        }

        let Some(i) = min_idx else {
            break; // all streams exhausted
        };

        let pos = heads[i].unwrap();
        emit(pos)?;

        // Advance stream `i`.
        match &mut streams[i] {
            Stream::Active { reader } => match read_one_u64(reader) {
                Ok(next) => heads[i] = Some(next),
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    heads[i] = None;
                    streams[i] = Stream::Empty;
                }
                Err(e) => return Err(e),
            },
            Stream::Empty => unreachable!("Empty streams have heads[i] == None"),
        }
    }

    Ok(())
}

/// Per-bucket reader state.
enum Stream {
    Active { reader: BufReader<std::fs::File> },
    Empty,
}

#[inline]
fn read_one_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
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
    fn ext_mem_single_subarray() {
        assert_matches_in_memory(b"banana", 1);
    }

    #[test]
    fn ext_mem_more_subarrays_than_text() {
        // p > n; effective_subproblem_count clamps to n.
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
                let want: Vec<u32> = build_in_memory(&text);
                let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
                let got = ext_mem_sa(&text, p);
                assert_eq!(got, want64, "n={n} p={p}");
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
                let want: Vec<u32> = build_in_memory(&text);
                let want64: Vec<u64> = want.iter().map(|&x| x as u64).collect();
                let got = ext_mem_sa(&text, p);
                assert_eq!(got, want64, "n={n} p={p}");
            }
        }
    }
}
