//! Minimal CLI for benchmarking `caps-sa` against the upstream C++
//! reference. Matches a subset of `caps_sa <input> <output> [flags]`:
//!
//! ```text
//! caps_sa <input> <output> [--ext-mem] [--subproblem-count N] [--threads N]
//! ```
//!
//! `<input>` is read as a raw byte file. The suffix array is written to
//! `<output>` as a packed little-endian `u64[]`. Timing of the SA build
//! (excluding I/O) is printed to stderr.
//!
//! Build with `cargo build --release --example caps_sa`.

use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use caps_sa::{ExtMemOpts, build_ext_mem, build_in_memory, build_in_memory_sample_sort};

struct Args {
    input: PathBuf,
    output: PathBuf,
    ext_mem: bool,
    in_mem_ss: bool,
    subproblem_count: usize,
    threads: Option<usize>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut ext_mem = false;
    let mut in_mem_ss = false;
    let mut subproblem_count: usize = 0;
    let mut threads: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--ext-mem" => {
                ext_mem = true;
                i += 1;
            }
            "--in-mem-ss" => {
                in_mem_ss = true;
                i += 1;
            }
            "--subproblem-count" => {
                subproblem_count = argv[i + 1]
                    .parse()
                    .expect("--subproblem-count expects a positive integer");
                i += 2;
            }
            "--threads" => {
                threads = Some(
                    argv[i + 1]
                        .parse()
                        .expect("--threads expects a positive integer"),
                );
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: caps_sa <input> <output> [--ext-mem | --in-mem-ss] \
                     [--subproblem-count N] [--threads N]"
                );
                process::exit(0);
            }
            _ => {
                positional.push(argv[i].clone());
                i += 1;
            }
        }
    }
    if positional.len() != 2 {
        eprintln!(
            "error: expected 2 positional args (input, output), got {}",
            positional.len()
        );
        process::exit(2);
    }
    Args {
        input: PathBuf::from(&positional[0]),
        output: PathBuf::from(&positional[1]),
        ext_mem,
        in_mem_ss,
        subproblem_count,
        threads,
    }
}

fn main() -> std::io::Result<()> {
    let args = parse_args();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("failed to configure rayon thread pool");
    }

    let read_start = Instant::now();
    let text = fs::read(&args.input)?;
    let read_elapsed = read_start.elapsed();
    eprintln!(
        "read: n={} bytes in {:.3}s",
        text.len(),
        read_elapsed.as_secs_f64()
    );

    // Pick the narrowest index type that can address `text`. The
    // sample-sort paths (ext-mem and in-mem-ss) handle the u32/u64
    // dispatch internally and always emit u64; only the plain in-mem
    // merge-sort path needs to be widened/narrowed at this CLI layer.
    let use_u32 = text.len() <= (u32::MAX as usize) + 1 && !args.ext_mem && !args.in_mem_ss;
    let n_entries: usize;
    let build_elapsed;
    let write_elapsed;

    if use_u32 {
        let build_start = Instant::now();
        let sa: Vec<u32> = build_in_memory(&text);
        build_elapsed = build_start.elapsed();
        n_entries = sa.len();
        eprintln!(
            "build: mode=in-mem(u32) n={n_entries} entries in {:.3}s",
            build_elapsed.as_secs_f64()
        );
        let write_start = Instant::now();
        let mut writer = BufWriter::new(fs::File::create(&args.output)?);
        let mut buf = [0u8; 4];
        for v in &sa {
            buf.copy_from_slice(&v.to_le_bytes());
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        write_elapsed = write_start.elapsed();
    } else if args.ext_mem || args.in_mem_ss {
        // Stream the SA straight from caps-sa's emit closure to the
        // output file. Both sample-sort paths share this streaming
        // shape; the only difference is whether the working buckets
        // are disk-backed (ext-mem) or RAM-only (in-mem-ss).
        let opts = ExtMemOpts {
            subproblem_count: args.subproblem_count,
            ..ExtMemOpts::default()
        };
        let writer = std::cell::RefCell::new(BufWriter::new(fs::File::create(&args.output)?));
        let mut count = 0usize;
        let mode_label = if args.ext_mem { "ext-mem" } else { "in-mem-ss" };
        let build_start = Instant::now();
        if args.ext_mem {
            build_ext_mem(&text, &opts, |pos| {
                count += 1;
                writer.borrow_mut().write_all(&pos.to_le_bytes())?;
                Ok(())
            })?;
        } else {
            build_in_memory_sample_sort(&text, &opts, |pos| {
                count += 1;
                writer.borrow_mut().write_all(&pos.to_le_bytes())?;
                Ok(())
            })?;
        }
        build_elapsed = build_start.elapsed();
        writer.borrow_mut().flush()?;
        n_entries = count;
        eprintln!(
            "build: mode={mode_label}(u64,stream) n={n_entries} entries in {:.3}s",
            build_elapsed.as_secs_f64()
        );
        write_elapsed = std::time::Duration::ZERO;
    } else {
        let _ = args.subproblem_count;
        let build_start = Instant::now();
        let sa: Vec<u64> = build_in_memory(&text);
        build_elapsed = build_start.elapsed();
        n_entries = sa.len();
        eprintln!(
            "build: mode=in-mem(u64) n={n_entries} entries in {:.3}s",
            build_elapsed.as_secs_f64()
        );
        let write_start = Instant::now();
        let mut writer = BufWriter::new(fs::File::create(&args.output)?);
        let mut buf = [0u8; 8];
        for v in &sa {
            buf.copy_from_slice(&v.to_le_bytes());
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        write_elapsed = write_start.elapsed();
    }

    eprintln!("write: {:.3}s", write_elapsed.as_secs_f64());
    Ok(())
}
