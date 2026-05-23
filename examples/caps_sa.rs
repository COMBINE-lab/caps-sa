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

use caps_sa::{ExtMemOpts, build_ext_mem, build_in_memory};

struct Args {
    input: PathBuf,
    output: PathBuf,
    ext_mem: bool,
    subproblem_count: usize,
    threads: Option<usize>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut ext_mem = false;
    let mut subproblem_count: usize = 0;
    let mut threads: Option<usize> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--ext-mem" => {
                ext_mem = true;
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
                    "usage: caps_sa <input> <output> [--ext-mem] \
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

    // Pick the narrowest index type that can address `text`. u32 (and
    // upstream's `i32` SA) lets the whole working set live in half the
    // cache footprint when `n < 2^31`, which is a big win on
    // moderately-large inputs.
    let use_u32 = text.len() < (1usize << 31) && !args.ext_mem;
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
    } else {
        let build_start = Instant::now();
        let sa: Vec<u64> = if args.ext_mem {
            let opts = ExtMemOpts {
                subproblem_count: args.subproblem_count,
                ..ExtMemOpts::default()
            };
            let mut out: Vec<u64> = Vec::with_capacity(text.len());
            build_ext_mem(&text, &opts, |pos| {
                out.push(pos);
                Ok(())
            })?;
            out
        } else {
            let _ = args.subproblem_count;
            build_in_memory(&text)
        };
        build_elapsed = build_start.elapsed();
        n_entries = sa.len();
        eprintln!(
            "build: mode={}(u64) n={n_entries} entries in {:.3}s",
            if args.ext_mem { "ext-mem" } else { "in-mem" },
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
