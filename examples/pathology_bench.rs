//! Empirical check of the spacer-padding pathology that motivated
//! `build_*_for_positions`. Synthesises the kind of transformed text
//! rustar-aligner's `sa_build` produces for a small-chromosome /
//! large-`genomeChrBinNbits` test fixture, then times the OLD shape
//! (`build_ext_mem(text)` — sort everything, filter at emit) against
//! the NEW shape (`build_ext_mem_for_positions(text, positions)` —
//! never queue the spacer-starting suffixes) on the same `t_prime`.
//!
//! Run with: `cargo run --release --example pathology_bench`.

use std::time::Instant;

use caps_sa::{ExtMemOpts, build_ext_mem, build_ext_mem_for_positions};

/// Synthesise the post-sentinel-transform text for a `chr_len`-byte
/// chromosome rounded up to `padded_chr_len` (matches what
/// `Genome::from_fasta` produces with a given `--genomeChrBinNbits`),
/// forward + reverse-complement laid out as `[F | RC]`, with each
/// spacer run replaced by a distinct sentinel and a terminal sentinel
/// appended. Two segments → two spacer runs → sentinels 5 and 6;
/// terminal sentinel = 7.
fn pathology_text(chr_len: usize, padded_chr_len: usize) -> Vec<u8> {
    assert!(padded_chr_len > chr_len, "pad_len must be positive");
    let pad_len = padded_chr_len - chr_len;
    let mut t = Vec::with_capacity(2 * padded_chr_len + 1);
    // Forward chromosome: cyclic 0..4 (ACGT) so suffixes diverge fast.
    for i in 0..chr_len {
        t.push((i % 4) as u8);
    }
    // Spacer run 0 → sentinel 5.
    t.resize(t.len() + pad_len, 5);
    // RC chromosome: shifted cyclic so it's distinguishable from forward.
    for i in 0..chr_len {
        t.push(((i + 1) % 4) as u8);
    }
    // Spacer run 1 → sentinel 6.
    t.resize(t.len() + pad_len, 6);
    // Terminal sentinel = 7 (larger than every per-run sentinel).
    t.push(7);
    t
}

fn run_one(chr_len: usize, padded_chr_len: usize) {
    let text = pathology_text(chr_len, padded_chr_len);
    let n = text.len();
    let positions: Vec<u64> = (0..n as u64).filter(|&p| text[p as usize] < 4).collect();
    let n_kept = positions.len();
    let n_skipped = n - n_kept;
    println!(
        "fixture: chr={chr_len} padded_chr={padded_chr_len} t_prime={n} \
         ACGT_kept={n_kept} ({:.1}%) spacer/sentinel_skipped={n_skipped} ({:.1}%)",
        100.0 * n_kept as f64 / n as f64,
        100.0 * n_skipped as f64 / n as f64,
    );

    let opts = ExtMemOpts {
        work_dir: std::env::temp_dir(),
        ..Default::default()
    };

    // OLD API: sort the entire text. caps-sa has no choice — every
    // spacer-starting suffix is in the queue and they all share a
    // near-maximal LCP with each other inside the run.
    let mut count_old = 0usize;
    let t = Instant::now();
    build_ext_mem(&text, &opts, |_| {
        count_old += 1;
        Ok(())
    })
    .unwrap();
    let dt_old = t.elapsed();
    assert_eq!(count_old, n);

    // NEW API: hand caps-sa only the ACGT positions; the
    // spacer/sentinel suffixes are never sorted.
    let mut count_new = 0usize;
    let t = Instant::now();
    build_ext_mem_for_positions(&text, positions, &opts, |_| {
        count_new += 1;
        Ok(())
    })
    .unwrap();
    let dt_new = t.elapsed();
    assert_eq!(count_new, n_kept);

    let speedup = dt_old.as_secs_f64() / dt_new.as_secs_f64();
    println!(
        "  OLD build_ext_mem                  : {:>10.3} ms  ({} positions)",
        dt_old.as_secs_f64() * 1e3,
        count_old
    );
    println!(
        "  NEW build_ext_mem_for_positions    : {:>10.3} ms  ({} positions)",
        dt_new.as_secs_f64() * 1e3,
        count_new
    );
    println!("  speedup                          : {:>10.1}×", speedup);
    println!();
}

fn main() {
    println!("== Synthetic spacer-padding pathology bench ==\n");
    // Matches rustar-aligner's docstring example: "20 kb test fixture
    // rounds to 256 kb of padded text".
    run_one(20_000, 256 * 1024); // bin_nbits = 18
    // A more extreme padded:chr ratio — what the smallest fixtures
    // (sub-1 kb chromosome) would hit at the same bin_nbits.
    run_one(1_000, 256 * 1024);
    // Larger bin_nbits = 20 (1 MiB padding): the regime the docstring
    // implies would have made ext-mem unusable on tiny inputs.
    run_one(10_000, 1024 * 1024);
}
