//! Periodic-run detection, and a run-aware suffix comparator.
//!
//! The LCP-enhanced merge resolves most steps without touching the text, but
//! when it does have to compare two suffixes it scans their shared prefix one
//! vector at a time. That is fine until the text contains a long *periodic
//! run*, at which point two suffixes inside the run agree for as far as the
//! run continues and a single comparison scans megabytes.
//!
//! Genome assemblies always contain these. An `N` block is the obvious case,
//! and note that in wrapped FASTA it is **not** a run of one symbol: 60 `N`s
//! followed by a newline is a run of period 61. So detecting only
//! single-symbol runs would miss the representation that actually shows up.
//!
//! The detector covers periods up to [`MAX_PERIOD`] and **not** beyond.
//! Measured on synthetic 1 MiB periodic inputs, periods 1, 2, 61 and 64 are
//! detected with full coverage and sort 5.8-6.9x faster; periods 65 and 171
//! are not detected at all and run at parity. Alpha-satellite arrays, whose
//! canonical monomer is 171 bases, are therefore *outside* this detector.
//! The sampling stage can also miss a run that is localised enough to fall
//! between its windows.
//!
//! The observation that makes this cheap: if `text[s..e)` has period `q`, and
//! two suffixes start at `a < b` inside it with `(b - a) % q == 0`, then they
//! agree until the later one reaches `e`. That is
//!
//! ```text
//! lcp(a, b) >= e - b
//! ```
//!
//! known in `O(1)` from the run's bounds, with no scanning at all. The scan
//! resumes at `e`, where the run's guarantee stops. When the phase does not
//! match (`(b - a) % q != 0`) the two suffixes must differ within `q` symbols,
//! so the ordinary scan is already short.
//!
//! Detection is two-stage so that texts without runs pay almost nothing. A
//! sampling pass looks for any periodic window at all and collects the set of
//! periods that actually occur; the full scan then runs only for those
//! periods. On N-free DNA the sample finds nothing and the table is empty, so
//! [`RunTable::skip`] returns immediately on a slice-empty check.
//!
//! This is what the external-memory and sample-sort paths use instead of the
//! prefix doubling in [`crate::radix`]: doubling needs a rank for every
//! position in the text, which would defeat the bounded memory those paths
//! exist to provide, while a run table costs a few dozen entries.

use crate::lcp::{LcpDispatch, Symbol};
use crate::limits::LimitProvider;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Shortest run worth recording. Below this the ordinary SIMD scan crosses
/// the run faster than the binary search that would find it.
const MIN_RUN: usize = 1024;

/// Longest period considered. Covers homopolymers (period 1) and
/// wrapped-FASTA `N` blocks (period 61), which are the cases that occur in
/// practice in assembly FASTA.
///
/// It does **not** cover alpha-satellite arrays: their canonical monomer is
/// 171 bases, and a synthetic period-171 input measures at parity with no
/// detection at all. Raising this is a constant change, but the detection
/// scan is `O(periods x n)`, so it is not free.
const MAX_PERIOD: usize = 64;

/// Window used by the sampling pass to decide whether a period occurs at all.
const SAMPLE_WINDOW: usize = 512;

/// Symbols an ordinary scan must match before the run table is consulted.
///
/// Two suffixes of real sequence that agree this far are already unusual, so
/// the table is reached only when it might actually help. Small enough that
/// the probe is a handful of vector compares, and the probe is not wasted
/// work: whatever it matches counts towards the answer.
const RUN_PROBE: usize = 256;

/// A maximal stretch `[start, end)` of the text with period `period`, meaning
/// `text[i] == text[i + period]` for every `i` in `start..end - period`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Run {
    start: usize,
    end: usize,
    period: usize,
}

/// Long periodic runs of a byte text, sorted by start and non-overlapping.
///
/// Empty for texts without long repeats, which is the common case for
/// randomised or `N`-free input, and empty for symbol types wider than a byte.
#[derive(Clone, Debug, Default)]
pub(crate) struct RunTable {
    runs: Vec<Run>,
}

impl RunTable {
    /// An empty table. Every query short-circuits.
    pub(crate) fn empty() -> Self {
        Self { runs: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Detect the long periodic runs of `text`.
    pub(crate) fn detect(text: &[u8]) -> Self {
        let n = text.len();
        if n < MIN_RUN {
            return Self::empty();
        }

        // Stage 1: which periods occur anywhere? Sample windows across the
        // text and record every period that makes one of them periodic. A
        // text with no long repeat contributes nothing and stops here.
        let n_samples = 4096.min(n / SAMPLE_WINDOW).max(1);
        let stride = (n / n_samples).max(1);
        let mut seen = [false; MAX_PERIOD + 1];
        let found: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|s| {
                let base = s * stride;
                let end = (base + SAMPLE_WINDOW).min(n);
                let mut periods = Vec::new();
                if end - base < MAX_PERIOD * 2 {
                    return periods;
                }
                for q in 1..=MAX_PERIOD {
                    if (base..end - q).all(|i| text[i] == text[i + q]) {
                        periods.push(q);
                        // The smallest period implies all its multiples; one
                        // per window is enough to trigger the full scan.
                        break;
                    }
                }
                periods
            })
            .collect();
        for q in found.into_iter().flatten() {
            seen[q] = true;
        }
        let periods: Vec<usize> = (1..=MAX_PERIOD).filter(|&q| seen[q]).collect();
        if periods.is_empty() {
            return Self::empty();
        }

        // Stage 2: for each period that occurs, find its maximal runs.
        let mut runs: Vec<Run> = periods
            .par_iter()
            .flat_map_iter(|&q| {
                let mut out = Vec::new();
                let mut i = 0usize;
                while i + q < n {
                    if text[i] != text[i + q] {
                        i += 1;
                        continue;
                    }
                    let start = i;
                    while i + q < n && text[i] == text[i + q] {
                        i += 1;
                    }
                    // Matching through `i` means the periodic stretch covers
                    // `start..i + q`.
                    let end = i + q;
                    if end - start >= MIN_RUN {
                        out.push(Run {
                            start,
                            end,
                            period: q,
                        });
                    }
                }
                out
            })
            .collect();

        // Keep a non-overlapping set, preferring the earliest start and then
        // the longest reach, so a lookup is a single binary search.
        runs.sort_unstable_by_key(|r| (r.start, std::cmp::Reverse(r.end)));
        let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
        for r in runs {
            match merged.last() {
                Some(last) if r.start < last.end => {
                    // Overlaps the previous run. Extending the previous run
                    // would break its period guarantee, so drop this one
                    // unless it reaches strictly further, in which case keep
                    // only the part past the previous end.
                    if r.end > last.end && r.end - last.end >= MIN_RUN {
                        merged.push(Run {
                            start: last.end,
                            end: r.end,
                            period: r.period,
                        });
                    }
                }
                _ => merged.push(r),
            }
        }
        Self { runs: merged }
    }

    /// The run containing `pos`, if any.
    #[inline]
    fn at(&self, pos: usize) -> Option<&Run> {
        if self.runs.is_empty() {
            return None;
        }
        let i = self.runs.partition_point(|r| r.start <= pos);
        let r = self.runs.get(i.checked_sub(1)?)?;
        (pos < r.end).then_some(r)
    }

    /// Start of the first run beginning at or after `pos`, or `usize::MAX`.
    #[inline]
    fn next_start(&self, pos: usize) -> usize {
        if self.runs.is_empty() {
            return usize::MAX;
        }
        let i = self.runs.partition_point(|r| r.start < pos);
        self.runs.get(i).map_or(usize::MAX, |r| r.start)
    }

    /// Symbols that suffixes `a` and `b` are guaranteed to share starting at
    /// their current offset, derived from run structure alone.
    ///
    /// Returns `0` when nothing can be concluded, which is always the answer
    /// for an empty table.
    #[inline]
    fn skip(&self, a: usize, b: usize) -> usize {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let Some(r) = self.at(lo) else { return 0 };
        if hi >= r.end || (hi - lo) % r.period != 0 {
            return 0;
        }
        // Both offsets sit in the same run, an exact whole number of periods
        // apart, so they agree until the later one leaves the run.
        r.end - hi
    }
}

/// A suffix comparator: the SIMD LCP kernel plus the run table that lets it
/// skip long periodic repeats instead of scanning them.
///
/// Threaded through the merge kernel in place of a bare [`LcpDispatch`]. It is
/// `Copy`, so it still travels through the recursion in registers.
#[derive(Copy, Clone)]
pub(crate) struct Cmp<'a> {
    pub(crate) dispatch: LcpDispatch,
    pub(crate) runs: &'a RunTable,
}

impl<'a> Cmp<'a> {
    pub(crate) fn new(dispatch: LcpDispatch, runs: &'a RunTable) -> Self {
        Self { dispatch, runs }
    }

    /// Whether the text contains long periodic repeats.
    ///
    /// Callers use this to decide whether a fixed-depth key can be expected
    /// to resolve most suffixes: a long run is precisely a stretch where it
    /// cannot, because every suffix inside it shares the whole key.
    #[inline]
    pub(crate) fn has_long_runs(&self) -> bool {
        !self.runs.is_empty()
    }

    /// LCP of `text[p..]` and `text[q..]` in symbols, bounded by `max_ctx`,
    /// using run structure to jump over long periodic stretches.
    ///
    /// With an empty run table this is exactly [`LcpDispatch::lcp`] plus one
    /// predictable branch.
    ///
    /// When a table *is* present, the ordinary bounded scan still runs first.
    /// Consulting the table costs up to three binary searches before any
    /// comparison happens, and the overwhelming majority of LCP calls in real
    /// sequence mismatch within a few symbols and never reach a run at all.
    /// Paying the lookup up front taxed every one of them: on a filtered,
    /// `N`-containing chr21 that alone turned a 1.80 s build into 2.72 s even
    /// though the answers were identical.
    ///
    /// So: probe first, and only once a match has survived [`RUN_PROBE`]
    /// symbols — which ordinary genomic difference does not — is it worth
    /// asking whether a run explains it.
    #[inline]
    pub(crate) fn lcp<S: Symbol>(&self, text: &[S], p: usize, q: usize, max_ctx: usize) -> usize {
        if self.runs.is_empty() || size_of::<S>() != 1 {
            return self.dispatch.lcp(text, p, q, max_ctx);
        }

        let probe = max_ctx.min(RUN_PROBE);
        let got = self.dispatch.lcp(text, p, q, probe);
        if got < probe || probe == max_ctx {
            // Either a real mismatch, or the caller's bound was reached. No
            // run can extend this, so the table is never touched.
            return got;
        }

        let mut i = got;
        while i < max_ctx {
            let jump = self.runs.skip(p + i, q + i);
            if jump > 0 {
                i = (i + jump).min(max_ctx);
                continue;
            }
            // Stop the scan where a run begins, so it never traverses one.
            // Scanning into a run is exactly the megabyte-long case.
            let next = self
                .runs
                .next_start(p + i)
                .saturating_sub(p + i)
                .min(self.runs.next_start(q + i).saturating_sub(q + i))
                .max(1);
            let window = max_ctx - i;
            let bounded = next.min(window);
            let got = self.dispatch.lcp(text, p + i, q + i, bounded);
            i += got;
            if got < bounded {
                // A real mismatch, not a window boundary.
                break;
            }
        }
        i.min(max_ctx)
    }

    /// Total order on two suffixes, mirroring [`LcpDispatch::suffix_cmp_with`]
    /// but going through the run-aware [`Self::lcp`].
    #[inline]
    pub(crate) fn suffix_cmp_with<S: Symbol, L: LimitProvider>(
        &self,
        text: &[S],
        lp: &L,
        p: usize,
        q: usize,
        max_ctx: usize,
    ) -> Ordering {
        let lim_p = lp.lim_at(p);
        let lim_q = lp.lim_at(q);
        let lim = lim_p.min(lim_q).min(max_ctx);
        let common = self.lcp(text, p, q, lim);
        if common < lim {
            text[p + common].cmp(&text[q + common])
        } else {
            lp.boundary_order(p, lim_p, q, lim_q)
        }
    }
}

/// Build a run table for `text` when the symbol type is a byte, otherwise an
/// empty one.
///
/// Detection is a sequential-read pass and only runs at all if the sampling
/// stage finds a periodic window, so texts without long repeats pay a single
/// sampling sweep.
pub(crate) fn detect_for<S: Symbol>(text: &[S]) -> RunTable {
    if size_of::<S>() != 1 {
        return RunTable::empty();
    }
    // SAFETY: `S` is one byte wide with no padding and no invalid bit
    // patterns (the `Symbol` contract), so a byte view over the same memory is
    // valid for reads of the same length.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(text.as_ptr() as *const u8, text.len()) };
    RunTable::detect(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_lcp(text: &[u8], a: usize, b: usize, max_ctx: usize) -> usize {
        let lim = (text.len() - a).min(text.len() - b).min(max_ctx);
        (0..lim).take_while(|&i| text[a + i] == text[b + i]).count()
    }

    /// The run-aware LCP must agree with a byte-at-a-time scan for every pair
    /// of positions, whether or not a run is involved.
    fn assert_lcp_agrees(text: &[u8]) {
        let runs = RunTable::detect(text);
        let cmp = Cmp::new(LcpDispatch::detect(), &runs);
        let n = text.len();
        let step = (n / 64).max(1);
        for a in (0..n).step_by(step) {
            for b in (0..n).step_by(step) {
                let want = naive_lcp(text, a, b, usize::MAX);
                let got = cmp.lcp(text, a, b, usize::MAX);
                assert_eq!(got, want, "lcp({a}, {b}) on len-{n} text");
            }
        }
    }

    #[test]
    fn empty_table_for_texts_without_runs() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0DD1);
        let text: Vec<u8> = (0..50_000).map(|_| rng.random_range(0..4u8)).collect();
        assert!(RunTable::detect(&text).is_empty());
    }

    #[test]
    fn detects_homopolymer() {
        let mut text: Vec<u8> = vec![1, 2, 3];
        text.extend(std::iter::repeat_n(0u8, 5000));
        text.extend([1, 2, 3]);
        let table = RunTable::detect(&text);
        assert!(!table.is_empty());
        assert!(table.runs.iter().any(|r| r.end - r.start >= 5000));
    }

    /// Wrapped FASTA: 60 `N`s then a newline. The single-symbol runs are only
    /// 60 long, so a homopolymer-only detector would find nothing; the real
    /// structure is period 61.
    #[test]
    fn detects_wrapped_fasta_n_block() {
        let mut text: Vec<u8> = b"ACGT".to_vec();
        for _ in 0..200 {
            text.extend(std::iter::repeat_n(b'N', 60));
            text.push(b'\n');
        }
        text.extend(b"ACGT");
        let table = RunTable::detect(&text);
        assert!(!table.is_empty(), "period-61 N block should be detected");
        assert!(table.runs.iter().any(|r| r.period == 61 || r.period == 1));
    }

    #[test]
    fn runs_are_sorted_and_disjoint() {
        let mut text: Vec<u8> = Vec::new();
        text.extend(std::iter::repeat_n(0u8, 3000));
        text.extend(b"ACGTACGT");
        text.extend((0..3000).map(|i| (i % 7) as u8));
        text.extend(b"TTTT");
        let table = RunTable::detect(&text);
        for w in table.runs.windows(2) {
            assert!(w[0].end <= w[1].start, "runs overlap: {:?}", w);
            assert!(w[0].start < w[1].start);
        }
        for r in &table.runs {
            for i in r.start..r.end - r.period {
                assert_eq!(text[i], text[i + r.period], "period claim is wrong");
            }
        }
    }

    #[test]
    fn lcp_agrees_on_homopolymer() {
        let mut text: Vec<u8> = b"ACGT".to_vec();
        text.extend(std::iter::repeat_n(0u8, 4000));
        text.extend(b"ACGT");
        assert_lcp_agrees(&text);
    }

    #[test]
    fn lcp_agrees_on_wrapped_fasta() {
        let mut text: Vec<u8> = b"ACGTAC".to_vec();
        for _ in 0..120 {
            text.extend(std::iter::repeat_n(b'N', 60));
            text.push(b'\n');
        }
        text.extend(b"GTGTGT");
        assert_lcp_agrees(&text);
    }

    #[test]
    fn lcp_agrees_on_multi_period_text() {
        let mut text: Vec<u8> = Vec::new();
        text.extend((0..3000).map(|i| (i % 3) as u8));
        text.extend(b"XYZ");
        text.extend(std::iter::repeat_n(9u8, 2500));
        text.extend((0..2000).map(|i| (i % 5) as u8));
        assert_lcp_agrees(&text);
    }

    #[test]
    fn lcp_respects_max_ctx_inside_a_run() {
        let mut text: Vec<u8> = b"AC".to_vec();
        text.extend(std::iter::repeat_n(0u8, 4000));
        let runs = RunTable::detect(&text);
        let cmp = Cmp::new(LcpDispatch::detect(), &runs);
        for &ctx in &[0usize, 1, 7, 100, 3000] {
            assert_eq!(cmp.lcp(&text, 2, 3, ctx), naive_lcp(&text, 2, 3, ctx));
        }
    }

    #[test]
    fn suffix_cmp_matches_slice_order() {
        use crate::limits::PlainText;
        let mut text: Vec<u8> = b"ACGT".to_vec();
        text.extend(std::iter::repeat_n(5u8, 3000));
        text.extend(b"ACGT");
        let runs = RunTable::detect(&text);
        let cmp = Cmp::new(LcpDispatch::detect(), &runs);
        let lp = PlainText::new(text.len());
        let step = (text.len() / 40).max(1);
        for a in (0..text.len()).step_by(step) {
            for b in (0..text.len()).step_by(step) {
                let want = text[a..].cmp(&text[b..]);
                let got = cmp.suffix_cmp_with(&text, &lp, a, b, usize::MAX);
                assert_eq!(got, want, "suffix_cmp({a}, {b})");
            }
        }
    }
}
