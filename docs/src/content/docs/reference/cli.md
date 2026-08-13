---
title: CLI parameters
description: Every flag accepted by the caps_sa command-line example.
---

The `caps_sa` example reads a file as raw bytes, builds its suffix array, and writes the result to disk. Build it with `cargo build --release --example caps_sa`.

## Usage

```text
caps_sa <input> <output> [--ext-mem | --in-mem-ss] [--subproblem-count N] [--threads N]
```

Two positional arguments are required.

## Positional arguments

| Argument | Description |
| --- | --- |
| `<input>` | Path to the input file, read as a raw byte string (each byte is one `u8` symbol). |
| `<output>` | Path the suffix array is written to, as a packed little-endian integer array. |

The output is `u32[]` when the in-memory path runs on an input ≤ 4 GiB, and `u64[]` for the sample-sort paths (`--ext-mem` / `--in-mem-ss`), which always emit `u64`.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--ext-mem` | off | Use the **external-memory** sample-sort: working buckets spill to disk, the SA streams to `<output>`, peak RAM stays bounded. Best for genome-scale inputs. |
| `--in-mem-ss` | off | Use the **in-memory sample-sort**: same streaming shape as `--ext-mem`, but buckets stay in RAM. Useful when disk is the bottleneck and you have the memory. |
| `--subproblem-count N` | `0` (auto) | Number of subproblems `p` for the sample-sort paths. `0` targets ~65,536 positions per subarray, bounded by the worker count and 8,192. Only meaningful with `--ext-mem` / `--in-mem-ss`. |
| `--threads N` | all cores | Size of the Rayon worker pool. Omit to use every logical CPU. |
| `-h`, `--help` | — | Print usage and exit. |

`--ext-mem` and `--in-mem-ss` are mutually exclusive; with neither, the plain in-memory parallel merge-sort runs.

## Examples

```bash
# In-memory, all cores (default)
caps_sa hg38.bin hg38.sa

# External-memory, 32 threads, explicit subproblem count
caps_sa hg38.bin hg38.sa --ext-mem --threads 32 --subproblem-count 128

# In-memory sample-sort on a RAM-rich host
caps_sa reads.bin reads.sa --in-mem-ss --threads 16
```

## Output on stderr

The build prints timing (excluding I/O) so the tool can be used as a benchmark harness:

```text
read: n=3117275501 bytes in 1.842s
build: mode=ext-mem(u64,stream) n=3099750911 entries in 628.301s
write: 0.000s
```

For programmatic use, the library entry points behind these flags are documented in the [Library API](/caps-sa/reference/api/).
