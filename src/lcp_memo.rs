//! Per-partition geometric memoization of exact LCP intervals.
//!
//! An entry `(delta, end) -> start` proves that positions on one text
//! diagonal match on `[start, end)` and mismatch at `end`.  The mismatch is
//! essential: comparisons stopped only by a context or segment cap are lower
//! bounds and are never admitted.

use crate::lcp::{LcpDispatch, Symbol};
use std::num::NonZeroUsize;

const DEFAULT_PROBE: usize = 256;
const DEFAULT_MIN_LCP: usize = 1_024;
const DEFAULT_CAPACITY: usize = 4_096;
const DEFAULT_ACTIVATE_ENTRIES: usize = 64;

/// Controls whether phase-4 LCP comparisons use geometric memoization.
///
/// Memoization is disabled by default. Callers whose inputs contain many
/// repeated long contexts can opt in with [`Self::Geometric`].
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LcpMemoizationPolicy {
    /// Use the ordinary LCP-enhanced merge kernel without allocating a table.
    #[default]
    Disabled,
    /// Learn and reuse exact LCP intervals within each partition cascade.
    Geometric(GeometricMemoizationConfig),
}

impl LcpMemoizationPolicy {
    /// Enable geometric memoization with the GRCh38-tuned defaults.
    ///
    /// This is equivalent to
    /// `LcpMemoizationPolicy::Geometric(GeometricMemoizationConfig::default())`
    /// without requiring callers to name the configuration type.
    pub const fn geometric() -> Self {
        Self::Geometric(GeometricMemoizationConfig::DEFAULT)
    }
}

impl From<GeometricMemoizationConfig> for LcpMemoizationPolicy {
    fn from(config: GeometricMemoizationConfig) -> Self {
        Self::Geometric(config)
    }
}

/// Tuning controls for [`LcpMemoizationPolicy::Geometric`].
///
/// The defaults were selected on a complete ruSTAR-shaped GRCh38 plus GENCODE
/// workload. Tables are local to a single phase-4 partition cascade and are
/// dropped when that partition is emitted.
///
/// ```
/// use caps_sa::{GeometricMemoizationConfig, LcpMemoizationPolicy};
/// use std::num::NonZeroUsize;
///
/// let config = GeometricMemoizationConfig::default()
///     .with_probe_symbols(NonZeroUsize::new(512).unwrap());
/// let policy = LcpMemoizationPolicy::from(config);
/// # let _ = policy;
/// ```
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GeometricMemoizationConfig {
    probe_symbols: NonZeroUsize,
    min_lcp_symbols: NonZeroUsize,
    activate_after_entries: NonZeroUsize,
    max_entries_per_partition: NonZeroUsize,
}

impl GeometricMemoizationConfig {
    /// The default configuration measured on ruSTAR-shaped GRCh38 plus
    /// GENCODE input.
    pub const DEFAULT: Self = Self {
        probe_symbols: NonZeroUsize::new(DEFAULT_PROBE).unwrap(),
        min_lcp_symbols: NonZeroUsize::new(DEFAULT_MIN_LCP).unwrap(),
        activate_after_entries: NonZeroUsize::new(DEFAULT_ACTIVATE_ENTRIES).unwrap(),
        max_entries_per_partition: NonZeroUsize::new(DEFAULT_CAPACITY).unwrap(),
    };

    /// Number of symbols compared normally before an active-table lookup.
    pub const fn probe_symbols(&self) -> usize {
        self.probe_symbols.get()
    }

    /// Return this configuration with a new ordinary-comparison probe length.
    ///
    /// Production callers should normally retain the measured default so
    /// short comparisons stay on the direct path.
    pub const fn with_probe_symbols(mut self, probe_symbols: NonZeroUsize) -> Self {
        self.probe_symbols = probe_symbols;
        self
    }

    /// Minimum exact LCP length, including any prefix already known by the
    /// merge, that is eligible for admission.
    pub const fn min_lcp_symbols(&self) -> usize {
        self.min_lcp_symbols.get()
    }

    /// Return this configuration with a new exact-LCP admission threshold.
    pub const fn with_min_lcp_symbols(mut self, min_lcp_symbols: NonZeroUsize) -> Self {
        self.min_lcp_symbols = min_lcp_symbols;
        self
    }

    /// Number of learned entries required before a partition starts looking
    /// entries up.
    pub const fn activate_after_entries(&self) -> usize {
        self.activate_after_entries.get()
    }

    /// Return this configuration with a new lazy-activation threshold.
    ///
    /// If this exceeds [`Self::max_entries_per_partition`], that partition
    /// remains in the training kernel for its entire lifetime.
    pub const fn with_activate_after_entries(
        mut self,
        activate_after_entries: NonZeroUsize,
    ) -> Self {
        self.activate_after_entries = activate_after_entries;
        self
    }

    /// Hard entry bound for one phase-4 partition table.
    pub const fn max_entries_per_partition(&self) -> usize {
        self.max_entries_per_partition.get()
    }

    /// Return this configuration with a new per-partition entry bound.
    ///
    pub const fn with_max_entries_per_partition(
        mut self,
        max_entries_per_partition: NonZeroUsize,
    ) -> Self {
        self.max_entries_per_partition = max_entries_per_partition;
        self
    }
}

impl Default for GeometricMemoizationConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Compact internal representation copied into the hot path once per build.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MemoConfig {
    pub(crate) probe: usize,
    pub(crate) min_lcp: usize,
    pub(crate) capacity: usize,
    pub(crate) activate_entries: usize,
}

impl From<GeometricMemoizationConfig> for MemoConfig {
    fn from(config: GeometricMemoizationConfig) -> Self {
        Self {
            probe: config.probe_symbols.get(),
            min_lcp: config.min_lcp_symbols.get(),
            capacity: config.max_entries_per_partition.get(),
            activate_entries: config.activate_after_entries.get(),
        }
    }
}

/// Counters accumulated without synchronization inside one partition.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoStats {
    pub(crate) tables: u64,
    pub(crate) active_tables: u64,
    pub(crate) tables_0_15: u64,
    pub(crate) tables_16_31: u64,
    pub(crate) tables_32_63: u64,
    pub(crate) tables_64_127: u64,
    pub(crate) tables_128_255: u64,
    pub(crate) tables_256_plus: u64,
    pub(crate) calls: u64,
    pub(crate) cold_direct: u64,
    pub(crate) probe_resolved: u64,
    pub(crate) lookups: u64,
    pub(crate) direct_hits: u64,
    pub(crate) gap_hits: u64,
    pub(crate) misses: u64,
    pub(crate) inserts: u64,
    pub(crate) extensions: u64,
    pub(crate) capacity_rejects: u64,
    pub(crate) scanned_matches: u64,
    pub(crate) skipped_matches: u64,
    pub(crate) final_entries: u64,
    pub(crate) max_entries: u64,
    pub(crate) unique_diagonals: u64,
    pub(crate) singleton_diagonals: u64,
    pub(crate) max_entries_per_diagonal: u64,
    pub(crate) lookup_steps: u64,
    pub(crate) insert_steps: u64,
    pub(crate) insert_shifts: u64,
    pub(crate) gap_mismatches: u64,
    pub(crate) gap_caps: u64,
}

impl MemoStats {
    pub(crate) fn add_assign(&mut self, other: Self) {
        self.tables = self.tables.saturating_add(other.tables);
        self.active_tables = self.active_tables.saturating_add(other.active_tables);
        self.tables_0_15 = self.tables_0_15.saturating_add(other.tables_0_15);
        self.tables_16_31 = self.tables_16_31.saturating_add(other.tables_16_31);
        self.tables_32_63 = self.tables_32_63.saturating_add(other.tables_32_63);
        self.tables_64_127 = self.tables_64_127.saturating_add(other.tables_64_127);
        self.tables_128_255 = self.tables_128_255.saturating_add(other.tables_128_255);
        self.tables_256_plus = self.tables_256_plus.saturating_add(other.tables_256_plus);
        self.calls = self.calls.saturating_add(other.calls);
        self.cold_direct = self.cold_direct.saturating_add(other.cold_direct);
        self.probe_resolved = self.probe_resolved.saturating_add(other.probe_resolved);
        self.lookups = self.lookups.saturating_add(other.lookups);
        self.direct_hits = self.direct_hits.saturating_add(other.direct_hits);
        self.gap_hits = self.gap_hits.saturating_add(other.gap_hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.inserts = self.inserts.saturating_add(other.inserts);
        self.extensions = self.extensions.saturating_add(other.extensions);
        self.capacity_rejects = self.capacity_rejects.saturating_add(other.capacity_rejects);
        self.scanned_matches = self.scanned_matches.saturating_add(other.scanned_matches);
        self.skipped_matches = self.skipped_matches.saturating_add(other.skipped_matches);
        self.final_entries = self.final_entries.saturating_add(other.final_entries);
        self.max_entries = self.max_entries.max(other.max_entries);
        self.unique_diagonals = self.unique_diagonals.saturating_add(other.unique_diagonals);
        self.singleton_diagonals = self
            .singleton_diagonals
            .saturating_add(other.singleton_diagonals);
        self.max_entries_per_diagonal = self
            .max_entries_per_diagonal
            .max(other.max_entries_per_diagonal);
        self.lookup_steps = self.lookup_steps.saturating_add(other.lookup_steps);
        self.insert_steps = self.insert_steps.saturating_add(other.insert_steps);
        self.insert_shifts = self.insert_shifts.saturating_add(other.insert_shifts);
        self.gap_mismatches = self.gap_mismatches.saturating_add(other.gap_mismatches);
        self.gap_caps = self.gap_caps.saturating_add(other.gap_caps);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MemoEntry {
    diagonal: usize,
    end: usize,
    start: usize,
}

/// A bounded successor map local to one phase-4 partition cascade.
pub(crate) struct GeometricMemo {
    config: MemoConfig,
    entries: Vec<MemoEntry>,
    stats: MemoStats,
}

impl GeometricMemo {
    pub(crate) fn new(config: MemoConfig) -> Self {
        let initial_capacity = config.capacity.min(256);
        Self {
            config,
            entries: Vec::with_capacity(initial_capacity),
            stats: MemoStats::default(),
        }
    }

    pub(crate) fn finish(mut self) -> MemoStats {
        self.stats.tables = 1;
        self.stats.final_entries = self.entries.len() as u64;
        self.stats.active_tables = u64::from(self.is_active());
        match self.entries.len() {
            0..=15 => self.stats.tables_0_15 = 1,
            16..=31 => self.stats.tables_16_31 = 1,
            32..=63 => self.stats.tables_32_63 = 1,
            64..=127 => self.stats.tables_64_127 = 1,
            128..=255 => self.stats.tables_128_255 = 1,
            _ => self.stats.tables_256_plus = 1,
        }
        let mut group_start = 0usize;
        while group_start < self.entries.len() {
            let diagonal = self.entries[group_start].diagonal;
            let mut group_end = group_start + 1;
            while group_end < self.entries.len() && self.entries[group_end].diagonal == diagonal {
                group_end += 1;
            }
            let group_len = group_end - group_start;
            self.stats.unique_diagonals += 1;
            self.stats.singleton_diagonals += u64::from(group_len == 1);
            self.stats.max_entries_per_diagonal =
                self.stats.max_entries_per_diagonal.max(group_len as u64);
            group_start = group_end;
        }
        self.stats
    }

    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        self.entries.len() >= self.config.activate_entries
    }

    #[inline]
    pub(crate) fn probe(&self, max_ext: usize) -> usize {
        max_ext.min(self.config.probe)
    }

    /// Extend the LCP of `text[p..]` and `text[q..]` after a prefix of
    /// `known` symbols has already been proved equal by the merge invariant.
    /// The returned extension is bounded by `max_ext`.
    #[cfg(test)]
    #[inline]
    pub(crate) fn lcp<S: Symbol>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        max_ext: usize,
    ) -> usize {
        if self.is_active() {
            self.lcp_active_impl::<S, false>(text, dispatch, p, q, known, max_ext)
        } else {
            let got = dispatch.lcp(text, p + known, q + known, max_ext);
            self.observe_training_impl::<false>(p, q, known, got, max_ext);
            got
        }
    }

    /// Instrumented form of [`Self::lcp`]. Keeping the choice outside the hot
    /// merge loop lets LLVM erase all counter branches from normal runs.
    #[cfg(test)]
    #[inline]
    pub(crate) fn lcp_profiled<S: Symbol>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        max_ext: usize,
    ) -> usize {
        if self.is_active() {
            self.lcp_active_impl::<S, true>(text, dispatch, p, q, known, max_ext)
        } else {
            let got = dispatch.lcp(text, p + known, q + known, max_ext);
            self.observe_training_impl::<true>(p, q, known, got, max_ext);
            got
        }
    }

    #[inline]
    pub(crate) fn observe_training(
        &mut self,
        p: usize,
        q: usize,
        known: usize,
        got: usize,
        max_ext: usize,
    ) {
        self.observe_training_impl::<false>(p, q, known, got, max_ext);
    }

    #[inline]
    pub(crate) fn observe_training_profiled(
        &mut self,
        p: usize,
        q: usize,
        known: usize,
        got: usize,
        max_ext: usize,
    ) {
        self.observe_training_impl::<true>(p, q, known, got, max_ext);
    }

    #[inline]
    fn observe_training_impl<const STATS: bool>(
        &mut self,
        p: usize,
        q: usize,
        known: usize,
        got: usize,
        max_ext: usize,
    ) {
        if STATS {
            self.stats.calls = self.stats.calls.saturating_add(1);
            self.stats.cold_direct = self.stats.cold_direct.saturating_add(1);
            self.stats.scanned_matches = self.stats.scanned_matches.saturating_add(got as u64);
        }
        let total = known + got;
        // Long LCPs are rare. Test the selective condition first so ordinary
        // short mismatches pay one branch, not the exactness and active-state
        // checks as well.
        if total >= self.config.min_lcp && got < max_ext && !self.is_active() {
            let (base, other) = if p < q { (p, q) } else { (q, p) };
            self.insert_exact::<STATS>(base, other - base, total);
        }
    }

    #[cfg(test)]
    #[inline]
    fn lcp_active_impl<S: Symbol, const STATS: bool>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        max_ext: usize,
    ) -> usize {
        if STATS {
            self.stats.calls = self.stats.calls.saturating_add(1);
        }
        if max_ext == 0 || p == q {
            return dispatch.lcp(
                text,
                p.saturating_add(known),
                q.saturating_add(known),
                max_ext,
            );
        }

        let scan_p = p.saturating_add(known);
        let scan_q = q.saturating_add(known);

        // Resolve ordinary short comparisons before paying for a table query.
        let probe = self.probe(max_ext);
        let got = dispatch.lcp(text, scan_p, scan_q, probe);
        self.add_scanned::<STATS>(got);
        if got < probe {
            if STATS {
                self.stats.probe_resolved = self.stats.probe_resolved.saturating_add(1);
            }
            return got;
        }
        if probe == max_ext {
            if STATS {
                self.stats.probe_resolved = self.stats.probe_resolved.saturating_add(1);
            }
            return got;
        }

        self.lcp_after_probe_impl::<S, STATS>(text, dispatch, p, q, known, probe, max_ext)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lcp_after_probe<S: Symbol>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        probe: usize,
        max_ext: usize,
    ) -> usize {
        debug_assert!(self.is_active());
        self.lcp_after_probe_impl::<S, false>(text, dispatch, p, q, known, probe, max_ext)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lcp_after_probe_profiled<S: Symbol>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        probe: usize,
        max_ext: usize,
    ) -> usize {
        debug_assert!(self.is_active());
        self.lcp_after_probe_impl::<S, true>(text, dispatch, p, q, known, probe, max_ext)
    }

    #[inline]
    pub(crate) fn record_probe_profiled(&mut self, got: usize, probe: usize, max_ext: usize) {
        self.stats.calls = self.stats.calls.saturating_add(1);
        self.stats.scanned_matches = self.stats.scanned_matches.saturating_add(got as u64);
        if got < probe || probe == max_ext {
            self.stats.probe_resolved = self.stats.probe_resolved.saturating_add(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lcp_after_probe_impl<S: Symbol, const STATS: bool>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        known: usize,
        probe: usize,
        max_ext: usize,
    ) -> usize {
        let scan_p = p.saturating_add(known);
        let scan_q = q.saturating_add(known);
        let (base, other) = if p < q { (p, q) } else { (q, p) };
        let diagonal = other - base;

        let query = base.saturating_add(known).saturating_add(probe);
        let remaining = max_ext - probe;

        if STATS {
            self.stats.lookups = self.stats.lookups.saturating_add(1);
        }
        let successor_index = if STATS {
            let mut steps = 0u64;
            let index = self.entries.partition_point(|entry| {
                steps += 1;
                (entry.diagonal, entry.end) < (diagonal, query)
            });
            self.stats.lookup_steps = self.stats.lookup_steps.saturating_add(steps);
            index
        } else {
            self.entries
                .partition_point(|entry| (entry.diagonal, entry.end) < (diagonal, query))
        };
        let successor = self.entries.get(successor_index).and_then(|entry| {
            (entry.diagonal == diagonal).then_some((successor_index, entry.end, entry.start))
        });

        let Some((successor_index, end, start)) = successor else {
            if STATS {
                self.stats.misses = self.stats.misses.saturating_add(1);
            }
            return probe
                + self.scan_tail_and_insert::<S, STATS>(
                    text, dispatch, p, q, base, diagonal, known, probe, remaining,
                );
        };

        if start <= query {
            // The query lies inside a proved interval.  Its endpoint is an
            // observed mismatch, unless the caller's cap stops us first.
            let available = end - query;
            let skipped = available.min(remaining);
            if STATS {
                self.stats.direct_hits = self.stats.direct_hits.saturating_add(1);
            }
            self.add_skipped::<STATS>(skipped);
            self.extend_start::<STATS>(successor_index, base);
            return probe + skipped;
        }

        // Scan the unknown gap before the stored interval.  If it matches,
        // the interval can be extended left through both the gap and the
        // merge's already-known prefix.
        let gap = start - query;
        let gap_cap = gap.min(remaining);
        let gap_lcp = dispatch.lcp(
            text,
            scan_p.saturating_add(probe),
            scan_q.saturating_add(probe),
            gap_cap,
        );
        self.add_scanned::<STATS>(gap_lcp);
        if gap_lcp < gap_cap {
            if STATS {
                self.stats.gap_mismatches = self.stats.gap_mismatches.saturating_add(1);
            }
            self.insert_exact::<STATS>(
                base,
                diagonal,
                known.saturating_add(probe).saturating_add(gap_lcp),
            );
            return probe + gap_lcp;
        }
        if gap_cap == remaining {
            if STATS {
                self.stats.gap_caps = self.stats.gap_caps.saturating_add(1);
            }
            return max_ext;
        }

        let interval_len = end - start;
        let skipped = interval_len.min(remaining - gap);
        if STATS {
            self.stats.gap_hits = self.stats.gap_hits.saturating_add(1);
        }
        self.add_skipped::<STATS>(skipped);
        self.extend_start::<STATS>(successor_index, base);
        probe + gap + skipped
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_tail_and_insert<S: Symbol, const STATS: bool>(
        &mut self,
        text: &[S],
        dispatch: LcpDispatch,
        p: usize,
        q: usize,
        base: usize,
        diagonal: usize,
        known: usize,
        already_scanned: usize,
        remaining: usize,
    ) -> usize {
        let got = dispatch.lcp(
            text,
            p.saturating_add(known).saturating_add(already_scanned),
            q.saturating_add(known).saturating_add(already_scanned),
            remaining,
        );
        self.add_scanned::<STATS>(got);
        if got < remaining {
            self.insert_exact::<STATS>(
                base,
                diagonal,
                known.saturating_add(already_scanned).saturating_add(got),
            );
        }
        got
    }

    fn insert_exact<const STATS: bool>(&mut self, base: usize, diagonal: usize, lcp: usize) {
        if lcp < self.config.min_lcp {
            return;
        }
        let end = base.saturating_add(lcp);
        let key = (diagonal, end);
        let index = if STATS {
            let mut steps = 0u64;
            let index = self.entries.partition_point(|entry| {
                steps += 1;
                (entry.diagonal, entry.end) < key
            });
            self.stats.insert_steps = self.stats.insert_steps.saturating_add(steps);
            index
        } else {
            self.entries
                .partition_point(|entry| (entry.diagonal, entry.end) < key)
        };
        if let Some(entry) = self
            .entries
            .get_mut(index)
            .filter(|entry| (entry.diagonal, entry.end) == key)
        {
            if base < entry.start {
                entry.start = base;
                if STATS {
                    self.stats.extensions = self.stats.extensions.saturating_add(1);
                }
            }
            return;
        }
        if self.entries.len() >= self.config.capacity {
            if STATS {
                self.stats.capacity_rejects = self.stats.capacity_rejects.saturating_add(1);
            }
            return;
        }
        if STATS {
            self.stats.insert_shifts = self
                .stats
                .insert_shifts
                .saturating_add((self.entries.len() - index) as u64);
        }
        self.entries.insert(
            index,
            MemoEntry {
                diagonal,
                end,
                start: base,
            },
        );
        if STATS {
            self.stats.inserts = self.stats.inserts.saturating_add(1);
            self.stats.max_entries = self.stats.max_entries.max(self.entries.len() as u64);
        }
    }

    fn extend_start<const STATS: bool>(&mut self, index: usize, start: usize) {
        let Some(entry) = self.entries.get_mut(index) else {
            return;
        };
        if start < entry.start {
            entry.start = start;
            if STATS {
                self.stats.extensions = self.stats.extensions.saturating_add(1);
            }
        }
    }

    fn add_scanned<const STATS: bool>(&mut self, value: usize) {
        if STATS {
            self.stats.scanned_matches = self.stats.scanned_matches.saturating_add(value as u64);
        }
    }

    fn add_skipped<const STATS: bool>(&mut self, value: usize) {
        if STATS {
            self.stats.skipped_matches = self.stats.skipped_matches.saturating_add(value as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_start(memo: &GeometricMemo, diagonal: usize, end: usize) -> Option<usize> {
        memo.entries
            .iter()
            .find(|entry| entry.diagonal == diagonal && entry.end == end)
            .map(|entry| entry.start)
    }

    fn config() -> MemoConfig {
        MemoConfig {
            probe: 64,
            min_lcp: 128,
            capacity: 64,
            activate_entries: 1,
        }
    }

    fn naive_lcp<S: Symbol>(text: &[S], p: usize, q: usize, cap: usize) -> usize {
        let lim = text
            .len()
            .saturating_sub(p)
            .min(text.len().saturating_sub(q))
            .min(cap);
        (0..lim).take_while(|&i| text[p + i] == text[q + i]).count()
    }

    fn repeated_blocks() -> Vec<u8> {
        let mut text = vec![b'A'; 6_100];
        text[2_000] = b'C';
        text[5_000] = b'G';
        text
    }

    #[test]
    fn direct_hit_returns_exact_subsumed_lcp() {
        let text = repeated_blocks();
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(config());
        assert_eq!(
            memo.lcp_profiled(&text, dispatch, 0, 3_000, 0, 2_500),
            2_000
        );
        assert_eq!(
            memo.lcp_profiled(&text, dispatch, 500, 3_500, 0, 2_000),
            1_500
        );
        let stats = memo.finish();
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.direct_hits, 1);
        assert!(stats.skipped_matches >= 1_400);
    }

    #[test]
    fn gap_hit_extends_existing_endpoint() {
        let text = repeated_blocks();
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(config());
        assert_eq!(
            memo.lcp_profiled(&text, dispatch, 500, 3_500, 0, 2_000),
            1_500
        );
        assert_eq!(
            memo.lcp_profiled(&text, dispatch, 0, 3_000, 0, 2_500),
            2_000
        );
        assert_eq!(entry_start(&memo, 3_000, 2_000), Some(0));
        let stats = memo.finish();
        assert_eq!(stats.gap_hits, 1);
        assert_eq!(stats.extensions, 1);
    }

    #[test]
    fn capped_match_is_not_admitted_as_exact() {
        let text = repeated_blocks();
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(config());
        assert_eq!(memo.lcp(&text, dispatch, 0, 3_000, 0, 512), 512);
        assert!(memo.entries.is_empty());
    }

    #[test]
    fn known_prefix_is_included_in_admitted_interval() {
        let text = repeated_blocks();
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(config());
        assert_eq!(memo.lcp(&text, dispatch, 0, 3_000, 1_000, 1_500), 1_000);
        assert_eq!(entry_start(&memo, 3_000, 2_000), Some(0));
    }

    #[test]
    fn randomized_queries_match_naive_for_u8_and_i8() {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let text_u8: Vec<u8> = (0..16_384).map(|_| (next() % 5) as u8).collect();
        let text_i8: Vec<i8> = text_u8.iter().map(|&x| x as i8 - 2).collect();
        check_randomized(&text_u8, &mut next);
        check_randomized(&text_i8, &mut next);
    }

    fn check_randomized<S: Symbol>(text: &[S], next: &mut impl FnMut() -> u64) {
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(MemoConfig {
            probe: 4,
            min_lcp: 8,
            capacity: 1_024,
            activate_entries: 1,
        });
        for _ in 0..20_000 {
            let p = next() as usize % (text.len() - 1);
            let mut q = next() as usize % (text.len() - 1);
            if q == p {
                q = (q + 1) % (text.len() - 1);
            }
            let full = naive_lcp(text, p, q, usize::MAX);
            let known = if full == 0 {
                0
            } else {
                next() as usize % (full + 1)
            };
            let cap = 1 + next() as usize % 64;
            let want = naive_lcp(text, p + known, q + known, cap);
            let got = memo.lcp(text, dispatch, p, q, known, cap);
            assert_eq!(got, want, "p={p} q={q} known={known} cap={cap}");
        }
    }

    #[test]
    fn capacity_is_strict_but_existing_endpoint_can_extend() {
        let text = repeated_blocks();
        let dispatch = LcpDispatch::detect();
        let mut memo = GeometricMemo::new(MemoConfig {
            capacity: 1,
            ..config()
        });
        assert_eq!(memo.lcp(&text, dispatch, 500, 3_500, 0, 2_000), 1_500);
        // A different diagonal cannot add a second entry.
        let _ = memo.lcp(&text, dispatch, 0, 3_001, 0, 2_500);
        // The existing endpoint is still extendable while at capacity.
        assert_eq!(memo.lcp(&text, dispatch, 0, 3_000, 0, 2_500), 2_000);
        assert_eq!(memo.entries.len(), 1);
        assert_eq!(entry_start(&memo, 3_000, 2_000), Some(0));
    }
}
