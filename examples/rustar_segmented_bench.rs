//! Replay of a **rustar-aligner shaped** segmented build, standalone.
//!
//! rustar-aligner's `sa_build` builds a generalized suffix array over
//! `T = forward || revcomp` of a spacer-padded, splice-junction-extended
//! genome. It never widens the alphabet: the text stays `u8` (bases
//! `0..=3`, `N = 4`, spacer `= 5`), and the segment structure is handed
//! to caps-sa as a [`SegmentedText`] whose `boundary_order` is flipped
//! to STAR's `spacer-as-largest` convention. Only ACGT positions are
//! sorted, through the streaming filter API.
//!
//! This example replays exactly that call shape from a dumped fixture,
//! so caps-sa-side changes can be measured on the real workload
//! without a full `genomeGenerate` run around them.
//!
//! Fixture layout (produced by rustar-aligner, see `bench/README.md`):
//!
//! - `text.bin`: the `2 * n_genome` byte text, verbatim.
//! - `ends.u64`: the cumulative segment ends, little-endian `u64[]`,
//!   last entry equal to the text length.
//!
//! ```text
//! cargo run --release --example rustar_segmented_bench -- FIXTURE_DIR \
//!     [--threads N] [--repeat N] [--in-mem] [--work-dir DIR]
//! ```
//!
//! Reports the wall time of the caps-sa build alone, plus the emitted
//! entry count and an order-sensitive checksum of the emitted position
//! stream. Matching counts and checksums are a strong regression signal,
//! not a proof of equality; use an exact stream comparison when validating
//! a change before release.

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use caps_sa::{
    ExtMemOpts, LimitProvider, Opts, PackedPrefixSeedPolicy, SegmentedText,
    build_ext_mem_for_filter_with, build_in_memory_for_positions_with,
};

/// rustar-aligner's `StarSegmentedText`: `SegmentedText` limits with
/// STAR's boundary convention (longer remaining segment sorts first,
/// ascending position on ties).
struct StarSegmentedText {
    inner: SegmentedText,
}

impl LimitProvider for StarSegmentedText {
    #[inline]
    fn lim_at(&self, p: usize) -> usize {
        self.inner.lim_at(p)
    }

    #[inline]
    fn boundary_order(&self, p_a: usize, lim_a: usize, p_b: usize, lim_b: usize) -> Ordering {
        lim_b.cmp(&lim_a).then(p_a.cmp(&p_b))
    }

    /// Declare that this comparator's boundary convention is representable by
    /// packed keys. Activation remains a separate `ExtMemOpts` policy below.
    /// Set `CAPS_SA_BENCH_NO_RANK=1` to measure semantic ineligibility.
    #[inline]
    fn boundary_rank(&self) -> Option<caps_sa::BoundaryRank> {
        if std::env::var_os("CAPS_SA_BENCH_NO_RANK").is_some() {
            None
        } else {
            Some(caps_sa::BoundaryRank::LongerFirst)
        }
    }
}

struct Args {
    fixture: PathBuf,
    threads: Option<usize>,
    repeat: usize,
    in_mem: bool,
    work_dir: Option<PathBuf>,
}

const USAGE: &str = "usage: rustar_segmented_bench FIXTURE_DIR [--threads N] \
                     [--repeat N] [--in-mem] [--work-dir DIR]";

fn usage_error(message: &str) -> ! {
    eprintln!("error: {message}\n{USAGE}");
    process::exit(2);
}

fn option_value<'a>(argv: &'a [String], i: usize, option: &str) -> &'a str {
    argv.get(i + 1)
        .map(String::as_str)
        .unwrap_or_else(|| usage_error(&format!("{option} requires a value")))
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut threads = None;
    let mut repeat = 1usize;
    let mut in_mem = false;
    let mut work_dir = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--threads" => {
                let value = option_value(&argv, i, "--threads")
                    .parse::<usize>()
                    .unwrap_or_else(|_| usage_error("--threads expects a positive integer"));
                if value == 0 {
                    usage_error("--threads expects a positive integer");
                }
                threads = Some(value);
                i += 2;
            }
            "--repeat" => {
                repeat = option_value(&argv, i, "--repeat")
                    .parse::<usize>()
                    .unwrap_or_else(|_| usage_error("--repeat expects a positive integer"));
                if repeat == 0 {
                    usage_error("--repeat expects a positive integer");
                }
                i += 2;
            }
            "--in-mem" => {
                in_mem = true;
                i += 1;
            }
            "--work-dir" => {
                work_dir = Some(PathBuf::from(option_value(&argv, i, "--work-dir")));
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("{USAGE}");
                process::exit(0);
            }
            option if option.starts_with('-') => {
                usage_error(&format!("unknown option: {option}"));
            }
            _ => {
                positional.push(argv[i].clone());
                i += 1;
            }
        }
    }
    if positional.len() != 1 {
        usage_error("expected exactly one fixture directory");
    }
    Args {
        fixture: PathBuf::from(&positional[0]),
        threads,
        repeat,
        in_mem,
        work_dir,
    }
}

fn read_ends(path: &Path) -> Vec<u64> {
    let raw = fs::read(path).expect("read ends.u64");
    assert!(
        raw.len().is_multiple_of(8),
        "ends.u64 is not a whole u64 array"
    );
    raw.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let args = parse_args();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("build rayon pool");
    }

    let text = fs::read(args.fixture.join("text.bin")).expect("read text.bin");
    let ends = read_ends(&args.fixture.join("ends.u64"));
    let n = text.len();
    assert!(n > 0, "text.bin must not be empty");
    assert!(
        text.iter().all(|&symbol| symbol <= 5),
        "text.bin contains a symbol outside ruSTAR's encoded alphabet 0..=5"
    );
    assert_eq!(
        ends.last().copied(),
        Some(n as u64),
        "ends.u64 must close at the text length"
    );

    let n_kept = text.iter().filter(|&&b| b < 4).count();
    println!(
        "fixture: text={n} bytes, {} segments, ACGT kept={n_kept} ({:.1}%), \
         path={}, threads={}",
        ends.len(),
        100.0 * n_kept as f64 / n as f64,
        if args.in_mem { "in-memory" } else { "ext-mem" },
        rayon::current_num_threads(),
    );

    let lp = StarSegmentedText {
        inner: SegmentedText::from_ends(n, ends),
    };

    for round in 0..args.repeat {
        // A wide, order-sensitive checksum is a convenient regression signal.
        // It is not a proof of equality; release validation should compare the
        // emitted position streams directly.
        let mut count: u64 = 0;
        let mut checksum = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58du128;
        let mut emit = |p: u64| -> std::io::Result<()> {
            count += 1;
            checksum ^= p as u128;
            checksum = checksum.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
            checksum ^= (p as u128) << 64;
            Ok(())
        };

        let t0 = Instant::now();
        if args.in_mem {
            let positions: Vec<u64> = (0..n as u64).filter(|&p| text[p as usize] < 4).collect();
            let sa = build_in_memory_for_positions_with(&text, positions, &lp, &Opts::default());
            for &p in &sa {
                emit(p).expect("checksum sink never fails");
            }
        } else {
            let mut opts = ExtMemOpts::from_env()
                .packed_prefix_seed(PackedPrefixSeedPolicy::DenseAlphabetOnly);
            if let Some(dir) = &args.work_dir {
                opts = opts.work_dir(dir);
            }
            let text_ref: &[u8] = &text;
            build_ext_mem_for_filter_with(
                &text,
                |p| text_ref[p as usize] < 4,
                &lp,
                &opts,
                &mut emit,
            )
            .expect("caps-sa external-memory build");
        }
        let elapsed = t0.elapsed();

        println!(
            "round {round}: {:.3} s  entries={count}  checksum=0x{checksum:032x}",
            elapsed.as_secs_f64()
        );
    }
}
