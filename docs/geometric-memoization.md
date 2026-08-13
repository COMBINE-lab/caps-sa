# Geometric LCP memoization design

This document turns the endpoint construction in `LCP-memoization.pdf` into
an opt-in policy for the production external-memory path. The implementation
is deliberately phase-4-only: each partition cascade owns an independent,
bounded table and the ordinary direct kernel remains available without table
allocation or per-comparison policy dispatch.

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

## Public policy and tunables

Memoization is selected per construction through `ExtMemOpts` and remains
disabled by default:

```rust
let opts = caps_sa::ExtMemOpts::default()
    .lcp_memoization(caps_sa::LcpMemoizationPolicy::geometric());
```

The configuration is opaque and non-exhaustive. Its getters expose symbol
probe, exact-LCP admission, lazy-activation, and per-partition entry limits;
`with_*` methods take `NonZeroUsize` values and tune them without tying callers
to the struct layout or admitting zero-valued, ineffective configurations. The
measured defaults deliberately keep short comparisons on the direct path:

| Parameter | Default | `ExtMemOpts::from_env()` override |
|---|---:|---|
| Ordinary probe before lookup | 256 symbols | `CAPS_SA_MEMO_PROBE` |
| Minimum exact LCP admitted | 1,024 symbols | `CAPS_SA_MEMO_MIN_LCP` |
| Entries per partition | 4,096 | `CAPS_SA_MEMO_CAPACITY` |
| Entries learned before lookup | 64 | `CAPS_SA_MEMO_ACTIVATE_ENTRIES` |

`ExtMemOpts::from_env()` also recognizes `CAPS_SA_GEOMETRIC_MEMO=1`. Environment
variables are parsed only by that explicit constructor; `ExtMemOpts::default()`
and the build itself never consult them for memoization policy. Invalid and
zero-valued numeric overrides retain the defaults.

Detailed per-call instrumentation is an intentionally unstable diagnostic,
enabled with `CAPS_SA_MEMO_STATS=1` through `ExtMemOpts::from_env()`. Counters
are collected only when phase profiling is also enabled, preventing diagnostic
branches from contaminating ordinary memoized runs.

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

At this stage, the focused result did not justify the roughly five-minute
complete-GRCh38 A/B. The revised integration below supersedes that conclusion.

## Performance analysis and revised integration (2026-08-12)

Hardware counters showed that the original loss was not primarily the table.
On the same 32-core annotated fixture, the original memoized kernel retired
2.716 trillion instructions and 346.6 billion branches versus 2.675 trillion
and 335.0 billion for the direct kernel: +1.54% instructions and +3.45%
branches. Branch misses rose only 0.8%, and L1 data misses decreased. An empty
table that admitted no intervals incurred essentially the same cost, proving
that per-comparison control flow and changed code generation dominated binary
search and insertion.

The revised integration makes two changes:

1. A partition trains with the direct one-call SIMD kernel and chooses the
   direct or active-memo kernel once per merge pair. This removes table-state
   checks from ordinary comparisons.
2. In an active table, the 256-symbol probe is inlined into the merge kernel.
   Only a probe that fully matches calls the out-of-line geometric lookup.
   Short comparisons therefore pay the original scan plus one predictable
   condition, without constructing diagonal keys or touching the table.

The revised default skipped 15.4 billion matching symbols. Instrumentation
reported 1.72 million table lookups, 14.2 million lookup comparison steps,
10.9 million insertion comparison steps, and 13.7 million shifted entries.
The 751,622 final entries covered 437,937 diagonals; 232,055 diagonals had one
entry and no diagonal had more than 14. Thus table operations are tiny beside
the 1.215 billion merge comparisons.

Three interleaved exact-output runs measured:

| Path | Times (s) | Median |
|---|---|---:|
| Direct | 12.805, 12.778, 12.835 | 12.805 |
| Revised memo | 12.717, 12.712, 12.710 | 12.712 |

The median improvement is 0.72%. A counter pair measured 2.628 trillion
instructions for revised memoization versus 2.675 trillion direct (-1.77%),
while branch misses rose from 8.55 to 8.61 billion. This is a small positive
result on the focused fixture.

The complete GRCh38 plus GENCODE-v50 fixture amplified the benefit. It retained
6,176,694,310 positions, used `u64` indices and 8,192 partitions, and ran on
the same 32 pinned physical cores:

| Path | Build time | User CPU | Peak RSS | Output hash |
|---|---:|---:|---:|---|
| Direct | 295.048 s | 8,139.91 s | 10,590,292 KiB | `e81c...2000` |
| Revised memo | 270.063 s | 7,697.18 s | 9,879,268 KiB | `e81c...2000` |

That is 24.985 seconds or 8.47% less wall time, 5.44% less user CPU, and 6.7%
lower observed peak RSS. Temporary-file output was unchanged. The identical
count and 128-bit streaming hash strongly corroborate output equality for the
complete fixture; the focused fixture was compared position-for-position.

An instrumented full run found all 8,192 tables activated. It performed 78.45
million successor lookups, retained 14.24 million intervals (maximum 3,196 in
one table), and skipped 25.80 trillion known matching symbols while scanning
4.45 trillion. Thus approximately 85% of the represented long-prefix symbol
work was skipped. The larger fixture exposes many more long shared contexts
and also benefits from reducing straggler partitions, explaining why its wall
gain is substantially larger than chromosome 21's.

### Table alternatives

A synthetic operation benchmark matched the measured table shape: 137
intervals across 80 diagonals and 313 successor queries per table. Median
costs were:

| Representation | ns/operation |
|---|---:|
| Flat sorted vector | 29.1 |
| `BTreeMap<(diagonal, endpoint), start>` | 36.6 |
| Hash diagonal to sorted `Vec` | 14.8 |
| Hash diagonal with 1--2 endpoints inline | 11.2 |

The hash layouts won the microbenchmark but lost end-to-end: the plain hash
ran as high as 13.166 seconds versus 12.737 seconds for the flat vector in a
paired run; the inline version alternated between 12.754 versus 12.709 and
12.759 versus 12.802. Its hardware-counter run retired 2.643 trillion
instructions, about 15 billion more than the flat vector. Hashing and many
small allocations outweigh the few saved comparisons.

Complete-GRCh38 runs reached the same conclusion at the larger table size. The
inline hash took 269.224 and 270.377 seconds versus 270.063 seconds for the
flat vector, i.e. no repeatable timing difference. Its measured peak RSS was
10,542,468 KiB versus 9,879,268 KiB for the flat vector: a 648 MiB (6.7%)
penalty. The flat vector remains the best production representation measured
here.

The next promising work is therefore not another general map. It is either a
fixed-capacity, allocation-free per-diagonal cache (which must beat the flat
vector end-to-end), or moving reuse into phase 1 where most LCP work occurs.
The latter requires a task-local ownership design because Rayon recursion can
migrate between workers.

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
