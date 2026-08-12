# Geometric LCP memoization design

This document turns the endpoint construction in `LCP-memoization.pdf` into
a prototype that can be evaluated in the production external-memory path.
The prototype is deliberately phase-4-only and opt-in while its thresholds
and value are measured.

## Exact interval invariant

Canonicalize a compared pair so `a < b` and let its diagonal be `d = b - a`.
If a comparison observes exactly `l` matching symbols followed by an actual
text mismatch, store

```text
(diagonal=d, endpoint=a+l) -> start=a
```

The entry proves that the two diagonal-aligned text ranges match on
`[start, endpoint)` and mismatch at `endpoint`. Consequently, for any query
position `x` on the same diagonal with `start <= x <= endpoint`, the exact
unbounded LCP is `endpoint - x`.

Comparisons that merely exhaust `max_context` or a `LimitProvider` boundary
do **not** prove a mismatch and must not create an entry. Reusing an exact raw
text interval under a shorter caller cap is safe: the answer is simply
`min(endpoint - x, cap)`.

For one diagonal, overlapping exact intervals necessarily share an endpoint:
a different endpoint would claim both equality and inequality at the earlier
endpoint. The table therefore keeps one entry per `(diagonal, endpoint)` and
extends its `start` leftward when a later comparison proves a longer prefix.
This removes redundant/subsumed entries without an overlap-maintenance pass.

## Query algorithm

The merge already knows a prefix of length `m`. Starting at `a+m, b+m`:

1. Run the ordinary SIMD LCP for a short probe. A mismatch or caller cap ends
   the query without touching the table.
2. Find the nearest stored endpoint at or to the right on the same diagonal.
3. If the probed position is inside its interval, return/skip directly to the
   endpoint (or the caller cap).
4. If the interval begins later, scan only the gap. A mismatch in the gap is
   the exact result; if the gap matches, skip the stored interval and extend
   its start back through the newly proved prefix.
5. With no useful successor, scan normally. Insert only an exact mismatch
   whose complete LCP (including `m`) meets the admission threshold.

The initial probe is useful work, not a discarded pre-check. It both resolves
the overwhelmingly common short comparisons and advances a long comparison
toward the memoized interval.

## Ownership and concurrency

Use one table per phase-4 partition cascade.

- A partition cascade is sequential in current `main`, so the table requires
  no locks, atomics, or shared ownership.
- Reuse across the cascade levels captures the redundancy the note targets.
- Tables disappear when their partition result is emitted, naturally bounding
  lifetime and avoiding stale global state.
- Up to `4 * threads` partitions are in flight, so a strict per-table entry
  cap gives a predictable global memory bound.
- Phase 1 remains on the direct LCP path. Its recursive Rayon work can migrate
  between workers; sharing a table there would introduce contention, while
  thread-local tables would lose task-level locality. Phase 4 is the cleaner
  and more important first experiment.

The first implementation uses a sorted `Vec<(diagonal, endpoint, start)>`.
Successor lookup is a binary search. Tables observed on the chromosome-21
ruSTAR fixture are small (roughly 150--250 entries each), so one contiguous
allocation is cheaper than allocating and chasing tree nodes. The strict
per-table cap bounds insertion shifts and total memory on adversarial input.

## Initial policy and tunables

The opt-in prototype is enabled with `CAPS_SA_GEOMETRIC_MEMO=1`.

| Parameter | Initial value | Environment override |
|---|---:|---|
| Ordinary probe before lookup | 256 symbols | `CAPS_SA_MEMO_PROBE` |
| Minimum exact LCP admitted | 1,024 symbols | `CAPS_SA_MEMO_MIN_LCP` |
| Entries per partition | 4,096 | `CAPS_SA_MEMO_CAPACITY` |
| Entries learned before lookup | 64 | `CAPS_SA_MEMO_ACTIVATE_ENTRIES` |

Per-call instrumentation is enabled separately with `CAPS_SA_MEMO_STATS=1`;
keeping it off prevents profiling counters from contaminating timing runs.

When full, the first prototype still extends an existing endpoint but rejects
new endpoints. Profiling records capacity rejections. If saturation is common,
replacement should prefer longer intervals (the empirical note indicates they
subsume most work), but that complexity is deferred until the data requires it.

## Prototype evidence (2026-08-12)

The production-shaped check used the deterministic chromosome-21 backbone
plus every distinct splice-junction flank generated from GENCODE Human v50.
It retained 359,616,038 ACGT-starting positions and used 32 physical threads.
Every candidate run streamed all positions against the saved post-PR-#12
output and was byte-identical.

| Configuration | Build time | Peak RSS | Result |
|---|---:|---:|---|
| Direct post-#12 path | 12.619 s | 1,665,044 KiB | Reference |
| Geometric memo, defaults | 12.758 s | 1,663,508 KiB | 1.10% slower |

The instrumented candidate built 801,092 final entries across 5,488 partition
tables (maximum 252 in one table). It made 1,889,927 post-probe lookups and
skipped 16,950,304,464 known matching symbols, but phase-4 merge CPU rose from
about 164.5 to 171.0 seconds. Sweeps at 2,048, 4,096, 8,192, 16,384, and
65,536-symbol admission thresholds did not produce a gain. A `BTreeMap`
prototype was worse (about 13.02 seconds), which motivated the flat table.

Periods 1, 2, 61, 64, 65, and 171 all produced identical hashes with and
without memoization, but no stable timing benefit. For period 65 at one thread,
the direct and memoized times were 4.035 and 4.045 seconds respectively. This
is consistent with the existing periodic run-skipping optimization already
capturing the dense special case cheaply.

The focused result does not justify the roughly five-minute complete-GRCh38
A/B: the prototype remains opt-in and is not proposed for default enablement.
The interval invariant and table are nevertheless retained on this feature
branch as a measured starting point for a future integration with phase 1 or a
cheaper per-diagonal lookup scheme.

## Required evidence before default enablement

1. Differential equality with direct LCP on randomized, generic-symbol,
   finite-context, segmented, and ruSTAR-shaped fixtures.
2. Counters for calls, post-probe lookups, direct and gap hits, exact inserts,
   endpoint extensions, capacity rejects, peak entries, scanned matching
   symbols, and memoized matching symbols skipped.
3. Threshold sweeps on the focused annotated fixture.
4. Only if focused results are positive, an A/B on the complete annotated
   GRCh38 fixture against the post-#12 baseline.
5. A clear gain above noise without disproportionate RSS growth.
