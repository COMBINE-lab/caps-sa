//! Benchmark the annotation-shaped, splice-junction index build.
//!
//! This is the shape a STAR-style genome index actually constructs, and it is
//! the one none of the packed-key work reached until segmented keys existed:
//!
//! * the text is **segmented** — one segment per chromosome plus one per
//!   splice-junction flank — so LCP comparisons stop at segment boundaries;
//! * the comparator is STAR's **spacer-as-largest** `boundary_order`, in which
//!   the suffix that reaches its boundary first is the *larger* one, with an
//!   ascending-position tie-break;
//! * only **ACGT-starting** positions participate, so no suffix beginning
//!   inside an `N` block enters the sort at all;
//! * construction goes through the **external-memory** path.
//!
//! Usage:
//!
//! ```text
//! gsj_bench <text> <segment-lengths> [--threads N] [--plain] [--verify]
//! ```
//!
//! `<text>` is one byte per symbol with A/C/G/T/N coded `0..=4`.
//! `<segment-lengths>` is a packed little-endian `u64[]` summing to the text
//! length. `bench/gsj_fixture.py` builds both from a FASTA and a GTF.
//!
//! `--plain` swaps the segmented provider for `PlainText`, which is the
//! comparison worth having: it shows what the same positions cost when the
//! segmented comparator is not required.

use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use caps_sa::{
    BoundaryRank, ExtMemOpts, LimitProvider, PlainText, SegmentedText,
    build_ext_mem_for_positions_with,
};

/// STAR's convention: whichever suffix hits its boundary first is larger.
struct StarConvention {
    inner: SegmentedText,
}

impl LimitProvider for StarConvention {
    #[inline]
    fn lim_at(&self, p: usize) -> usize {
        self.inner.lim_at(p)
    }
    #[inline]
    fn boundary_order(&self, p_a: usize, lim_a: usize, p_b: usize, lim_b: usize) -> Ordering {
        lim_b.cmp(&lim_a).then(p_a.cmp(&p_b))
    }
    #[inline]
    fn boundary_rank(&self) -> Option<BoundaryRank> {
        Some(BoundaryRank::LongerFirst)
    }
}

fn main() -> std::io::Result<()> {
    let argv: Vec<String> = env::args().collect();
    let mut positional: Vec<String> = Vec::new();
    let mut threads: Option<usize> = None;
    let mut plain = false;
    let mut verify = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--threads" => {
                threads = Some(argv[i + 1].parse().expect("--threads expects an integer"));
                i += 2;
            }
            "--plain" => {
                plain = true;
                i += 1;
            }
            "--verify" => {
                verify = true;
                i += 1;
            }
            _ => {
                positional.push(argv[i].clone());
                i += 1;
            }
        }
    }
    if positional.len() != 2 {
        eprintln!("usage: gsj_bench <text> <segment-lengths> [--threads N] [--plain] [--verify]");
        process::exit(2);
    }

    let text = fs::read(PathBuf::from(&positional[0]))?;
    let raw = fs::read(PathBuf::from(&positional[1]))?;
    let lengths: Vec<usize> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    assert_eq!(
        lengths.iter().sum::<usize>(),
        text.len(),
        "segment lengths must sum to the text length"
    );

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("failed to configure rayon");
    }

    // Only ACGT starts participate, as in a STAR index.
    let positions: Vec<u64> = (0..text.len() as u64)
        .filter(|&p| text[p as usize] < 4)
        .collect();
    eprintln!(
        "fixture: {} symbols, {} segments, {} ACGT-start positions, mode={}",
        text.len(),
        lengths.len(),
        positions.len(),
        if plain { "plain" } else { "segmented+STAR" },
    );

    let opts = ExtMemOpts::default();
    let mut count = 0usize;
    let mut last = 0u64;
    let mut ordered = true;

    let start = Instant::now();
    if plain {
        let lp = PlainText::new(text.len());
        build_ext_mem_for_positions_with(&text, positions, &lp, &opts, |pos| {
            count += 1;
            ordered &= count == 1 || last <= pos;
            last = pos;
            Ok(())
        })?;
    } else {
        let lp = StarConvention {
            inner: SegmentedText::from_lengths(text.len(), &lengths),
        };
        // Checking order here would need the comparator; `--verify` below does
        // that properly on the collected output instead.
        build_ext_mem_for_positions_with(&text, positions, &lp, &opts, |_pos| {
            count += 1;
            Ok(())
        })?;
    }
    let elapsed = start.elapsed();
    eprintln!("build: {count} positions in {:.3}s", elapsed.as_secs_f64());
    let _ = ordered;

    if verify {
        // Re-run collecting, then check every adjacent pair against the
        // comparator directly. O(n) comparisons, each bounded by a segment.
        let lp = StarConvention {
            inner: SegmentedText::from_lengths(text.len(), &lengths),
        };
        let positions: Vec<u64> = (0..text.len() as u64)
            .filter(|&p| text[p as usize] < 4)
            .collect();
        let mut out: Vec<u64> = Vec::with_capacity(positions.len());
        build_ext_mem_for_positions_with(&text, positions, &lp, &opts, |pos| {
            out.push(pos);
            Ok(())
        })?;
        let t = Instant::now();
        let mut bad = 0usize;
        for w in out.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            let (la, lb) = (lp.lim_at(a), lp.lim_at(b));
            let mut ord = Ordering::Equal;
            for j in 0..la.min(lb) {
                if text[a + j] != text[b + j] {
                    ord = text[a + j].cmp(&text[b + j]);
                    break;
                }
            }
            if ord == Ordering::Equal {
                ord = lp.boundary_order(a, la, b, lb);
            }
            if ord == Ordering::Greater {
                bad += 1;
            }
        }
        if bad == 0 {
            eprintln!("verify: OK in {:.3}s", t.elapsed().as_secs_f64());
        } else {
            eprintln!("verify: FAILED, {bad} adjacent pairs out of order");
            process::exit(1);
        }
    }
    Ok(())
}
