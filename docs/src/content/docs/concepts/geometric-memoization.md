---
title: Geometric LCP memoization
description: When and how to reuse exact long common-prefix intervals in external-memory construction.
---

Geometric LCP memoization is an optional optimization for inputs in which many
suffix comparisons revisit the same long matching text intervals. It applies
to the final partition merges of the external-memory and in-memory sample-sort
builders. The ordinary direct merge remains the default.

## Enable the measured policy

```rust
use caps_sa::{ExtMemOpts, LcpMemoizationPolicy};

let opts = ExtMemOpts::default()
    .lcp_memoization(LcpMemoizationPolicy::geometric());
```

`LcpMemoizationPolicy::geometric()` selects the defaults measured on the
complete ruSTAR-shaped GRCh38 + GENCODE Human v50 construction. It is the
recommended entry point; tune the individual thresholds only after measuring
your own workload.

The equivalent environment-controlled form is useful for experiments:

```bash
CAPS_SA_GEOMETRIC_MEMO=1 your-program
```

Your program must construct options with `ExtMemOpts::from_env()` for this to
take effect. `ExtMemOpts::default()` and the builders do not implicitly read
environment variables.

## When it helps

The policy targets **repeated long contexts**, not repetitions in the abstract.
Most suffix comparisons are short and should finish in the SIMD direct path
without touching a table. Memoization helps when many comparisons survive the
ordinary probe and overlap exact long-LCP intervals learned earlier in the
same partition cascade.

On the complete production-shaped fixture used for ruSTAR integration:

- GENCODE Human v50 GRCh38 primary assembly plus comprehensive annotation;
- 6,557,611,930 text symbols and 6,176,694,310 retained ACGT-starting suffixes;
- 1,397,582 segments with STAR boundary ordering;
- external-memory `u64`, 8,192 partitions, and 32 physical cores.

The isolated comparison measured:

| Policy | Wall | User CPU | Peak RSS | Output hash |
| --- | ---: | ---: | ---: | --- |
| Disabled | 291.710 s | 8,029.81 s | 10,672,468 KiB | `e81c...2000` |
| Geometric defaults | 267.128 s | 7,607.24 s | 9,696,720 KiB | `e81c...2000` |

That is **8.43% less wall time** and **5.26% less user CPU**, with identical
output. A focused chromosome-21-plus-junction fixture showed only a small gain,
and synthetic periodic inputs were neutral once other specialized behavior was
accounted for. The policy therefore remains opt-in.

:::tip[Good candidate]
Enable it for a measured workload with many reused, long matching contexts,
such as ruSTAR's repeated genome-plus-splice-junction layout.
:::

:::caution[Do not infer benefit from long `N` runs alone]
ruSTAR parses away FASTA wrapping and filters out `N`-starting suffixes. The
pathological raw-FASTA population of millions of suffixes beginning inside one
long `N` region is not sorted in that workflow. Use an A/B on the actual
filtered and segmented input rather than enabling memoization solely because
the source genome contains `N`s.
:::

## Why it stays cheap when inactive

Each phase-4 partition cascade owns its table. There are no shared writes,
locks, or atomics, and the table disappears when the partition is emitted.
Before lookup begins, the direct kernel trains the table. After activation, a
comparison still scans a short prefix normally and touches the table only if
all probe symbols match.

The defaults are deliberately conservative:

| Setting | Default | Purpose |
| --- | ---: | --- |
| Probe | 256 symbols | Keep short comparisons on the direct SIMD path. |
| Minimum exact LCP | 1,024 symbols | Avoid storing intervals too short to repay lookup cost. |
| Activation | 64 entries | Avoid lookup in partitions that learn little reusable structure. |
| Capacity | 4,096 entries | Bound memory and insertion work per partition. |

Tables use a compact sorted vector. Hash-based layouts were timing-neutral on
the complete workload and added about 648 MiB peak RSS, so they were not
retained.

## Tune only with measurements

```rust
use caps_sa::{ExtMemOpts, GeometricMemoizationConfig};
use std::num::NonZeroUsize;

let memo = GeometricMemoizationConfig::default()
    .with_probe_symbols(NonZeroUsize::new(512).unwrap())
    .with_min_lcp_symbols(NonZeroUsize::new(2048).unwrap())
    .with_activate_after_entries(NonZeroUsize::new(128).unwrap())
    .with_max_entries_per_partition(NonZeroUsize::new(8192).unwrap());

let opts = ExtMemOpts::default().lcp_memoization(memo);
```

Sweeps around the defaults did not produce a repeatable improvement on the
ruSTAR fixture. Treat the controls as workload-specific experimental knobs,
not values that should routinely be raised.

For the exact interval invariant, implementation design, instrumentation, and
full experiment history, see the
[design record on GitHub](https://github.com/COMBINE-lab/caps-sa/blob/main/docs/geometric-memoization.md).
