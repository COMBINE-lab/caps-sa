//! Per-partition geometric memoization of exact LCP intervals.
//!
//! An entry `(delta, end) -> start` proves that positions on one text
//! diagonal match on `[start, end)` and mismatch at `end`.  The mismatch is
//! essential: comparisons stopped only by a context or segment cap are lower
//! bounds and are never admitted.

use crate::lcp::{LcpDispatch, Symbol};

const DEFAULT_PROBE: usize = 256;
const DEFAULT_MIN_LCP: usize = 1_024;
const DEFAULT_CAPACITY: usize = 4_096;
const DEFAULT_ACTIVATE_ENTRIES: usize = 64;

/// Runtime controls for the opt-in prototype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MemoConfig {
    pub(crate) probe: usize,
    pub(crate) min_lcp: usize,
    pub(crate) capacity: usize,
    pub(crate) activate_entries: usize,
    pub(crate) collect_stats: bool,
}

impl MemoConfig {
    /// Return a configuration only when geometric memoization is explicitly
    /// enabled.  Environment controls keep the prototype out of the public API
    /// while allowing threshold sweeps with one compiled binary.
    pub(crate) fn from_env() -> Option<Self> {
        let enabled = std::env::var_os("CAPS_SA_GEOMETRIC_MEMO")?;
        if matches!(enabled.to_str(), Some("0" | "false" | "off")) {
            return None;
        }
        Some(Self {
            probe: env_usize("CAPS_SA_MEMO_PROBE", DEFAULT_PROBE).max(1),
            min_lcp: env_usize("CAPS_SA_MEMO_MIN_LCP", DEFAULT_MIN_LCP).max(1),
            capacity: env_usize("CAPS_SA_MEMO_CAPACITY", DEFAULT_CAPACITY).max(1),
            activate_entries: env_usize("CAPS_SA_MEMO_ACTIVATE_ENTRIES", DEFAULT_ACTIVATE_ENTRIES)
                .max(1),
            collect_stats: std::env::var_os("CAPS_SA_MEMO_STATS").is_some(),
        })
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Counters accumulated without synchronization inside one partition.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoStats {
    pub(crate) tables: u64,
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
}

impl MemoStats {
    pub(crate) fn add_assign(&mut self, other: Self) {
        self.tables = self.tables.saturating_add(other.tables);
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
    // `(diagonal, mismatch endpoint) -> earliest proved matching start`.
    //
    // Partition tables are small in practice, so keep them in one sorted,
    // contiguous allocation. This avoids one heap allocation and pointer
    // chase per tree node while retaining logarithmic successor lookup.
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
        if self.config.collect_stats {
            self.stats.tables = 1;
            self.stats.final_entries = self.entries.len() as u64;
        }
        self.stats
    }

    /// Extend the LCP of `text[p..]` and `text[q..]` after a prefix of
    /// `known` symbols has already been proved equal by the merge invariant.
    /// The returned extension is bounded by `max_ext`.
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
        if self.config.collect_stats {
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

        let (base, other) = if p < q { (p, q) } else { (q, p) };
        let diagonal = other - base;
        let scan_p = p.saturating_add(known);
        let scan_q = q.saturating_add(known);

        // Until this partition has enough useful intervals, preserve the
        // original one-call SIMD path and use its exact result to seed the
        // table instead of splitting the scan at the probe boundary.
        if self.entries.len() < self.config.activate_entries {
            if self.config.collect_stats {
                self.stats.cold_direct = self.stats.cold_direct.saturating_add(1);
            }
            let got = dispatch.lcp(text, scan_p, scan_q, max_ext);
            self.add_scanned(got);
            if got < max_ext {
                self.insert_exact(base, diagonal, known.saturating_add(got));
            }
            return got;
        }

        // Resolve ordinary short comparisons before paying for a table query.
        let probe = max_ext.min(self.config.probe);
        let got = dispatch.lcp(text, scan_p, scan_q, probe);
        self.add_scanned(got);
        if got < probe {
            if self.config.collect_stats {
                self.stats.probe_resolved = self.stats.probe_resolved.saturating_add(1);
            }
            self.insert_exact(base, diagonal, known.saturating_add(got));
            return got;
        }
        if probe == max_ext {
            if self.config.collect_stats {
                self.stats.probe_resolved = self.stats.probe_resolved.saturating_add(1);
            }
            return got;
        }

        let query = base.saturating_add(known).saturating_add(probe);
        let remaining = max_ext - probe;

        if self.config.collect_stats {
            self.stats.lookups = self.stats.lookups.saturating_add(1);
        }
        let successor_index = self
            .entries
            .partition_point(|entry| (entry.diagonal, entry.end) < (diagonal, query));
        let successor = self.entries.get(successor_index).and_then(|entry| {
            (entry.diagonal == diagonal).then_some((successor_index, entry.end, entry.start))
        });

        let Some((successor_index, end, start)) = successor else {
            if self.config.collect_stats {
                self.stats.misses = self.stats.misses.saturating_add(1);
            }
            return probe
                + self.scan_tail_and_insert(
                    text, dispatch, p, q, base, diagonal, known, probe, remaining,
                );
        };

        if start <= query {
            // The query lies inside a proved interval.  Its endpoint is an
            // observed mismatch, unless the caller's cap stops us first.
            let available = end - query;
            let skipped = available.min(remaining);
            if self.config.collect_stats {
                self.stats.direct_hits = self.stats.direct_hits.saturating_add(1);
            }
            self.add_skipped(skipped);
            self.extend_start(successor_index, base);
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
        self.add_scanned(gap_lcp);
        if gap_lcp < gap_cap {
            self.insert_exact(
                base,
                diagonal,
                known.saturating_add(probe).saturating_add(gap_lcp),
            );
            return probe + gap_lcp;
        }
        if gap_cap == remaining {
            return max_ext;
        }

        let interval_len = end - start;
        let skipped = interval_len.min(remaining - gap);
        if self.config.collect_stats {
            self.stats.gap_hits = self.stats.gap_hits.saturating_add(1);
        }
        self.add_skipped(skipped);
        self.extend_start(successor_index, base);
        probe + gap + skipped
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_tail_and_insert<S: Symbol>(
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
        self.add_scanned(got);
        if got < remaining {
            self.insert_exact(
                base,
                diagonal,
                known.saturating_add(already_scanned).saturating_add(got),
            );
        }
        got
    }

    fn insert_exact(&mut self, base: usize, diagonal: usize, lcp: usize) {
        if lcp < self.config.min_lcp {
            return;
        }
        let end = base.saturating_add(lcp);
        let key = (diagonal, end);
        let index = self
            .entries
            .partition_point(|entry| (entry.diagonal, entry.end) < key);
        if let Some(entry) = self
            .entries
            .get_mut(index)
            .filter(|entry| (entry.diagonal, entry.end) == key)
        {
            if base < entry.start {
                entry.start = base;
                if self.config.collect_stats {
                    self.stats.extensions = self.stats.extensions.saturating_add(1);
                }
            }
            return;
        }
        if self.entries.len() >= self.config.capacity {
            if self.config.collect_stats {
                self.stats.capacity_rejects = self.stats.capacity_rejects.saturating_add(1);
            }
            return;
        }
        self.entries.insert(
            index,
            MemoEntry {
                diagonal,
                end,
                start: base,
            },
        );
        if self.config.collect_stats {
            self.stats.inserts = self.stats.inserts.saturating_add(1);
            self.stats.max_entries = self.stats.max_entries.max(self.entries.len() as u64);
        }
    }

    fn extend_start(&mut self, index: usize, start: usize) {
        let Some(entry) = self.entries.get_mut(index) else {
            return;
        };
        if start < entry.start {
            entry.start = start;
            if self.config.collect_stats {
                self.stats.extensions = self.stats.extensions.saturating_add(1);
            }
        }
    }

    fn add_scanned(&mut self, value: usize) {
        if self.config.collect_stats {
            self.stats.scanned_matches = self.stats.scanned_matches.saturating_add(value as u64);
        }
    }

    fn add_skipped(&mut self, value: usize) {
        if self.config.collect_stats {
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
            collect_stats: true,
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
        assert_eq!(memo.lcp(&text, dispatch, 0, 3_000, 0, 2_500), 2_000);
        assert_eq!(memo.lcp(&text, dispatch, 500, 3_500, 0, 2_000), 1_500);
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
        assert_eq!(memo.lcp(&text, dispatch, 500, 3_500, 0, 2_000), 1_500);
        assert_eq!(memo.lcp(&text, dispatch, 0, 3_000, 0, 2_500), 2_000);
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
            collect_stats: false,
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
