//! Capture-and-replay microbench for the byte-level LCP kernel.
//!
//! `capture` runs the rustar-shaped segmented build with the sampling
//! instrumentation compiled in and writes the sampled `(p, q, max_bytes)`
//! triples to disk. `replay` loads the text and the triples and times the
//! kernel alone over them, so a kernel change can be measured without the
//! surrounding build's run-to-run noise.
//!
//! ```text
//! lcp_replay capture FIXTURE_DIR TRIPLES.bin [--threads N]
//! lcp_replay replay  FIXTURE_DIR TRIPLES.bin [--rounds N]
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use caps_sa::LcpDispatch;

fn main() {
    let argv: Vec<String> = env::args().collect();
    let mode = argv.get(1).map(String::as_str).unwrap_or("");
    let fixture = PathBuf::from(argv.get(2).expect("fixture dir"));
    let triples_path = PathBuf::from(argv.get(3).expect("triples path"));
    let mut rounds = 5usize;
    let mut i = 4;
    while i < argv.len() {
        match argv[i].as_str() {
            "--rounds" => {
                rounds = argv[i + 1].parse().unwrap();
                i += 2;
            }
            other => panic!("unknown flag {other}"),
        }
    }

    let text = fs::read(fixture.join("text.bin")).expect("read text.bin");

    match mode {
        "replay" => {
            let raw = fs::read(&triples_path).expect("read triples");
            let triples: Vec<(usize, usize, usize)> = raw
                .chunks_exact(24)
                .map(|c| {
                    (
                        u64::from_le_bytes(c[0..8].try_into().unwrap()) as usize,
                        u64::from_le_bytes(c[8..16].try_into().unwrap()) as usize,
                        u64::from_le_bytes(c[16..24].try_into().unwrap()) as usize,
                    )
                })
                .collect();
            let dispatch = LcpDispatch::detect();
            eprintln!("replaying {} triples, {rounds} rounds", triples.len());
            for r in 0..rounds {
                let t0 = Instant::now();
                let mut acc = 0usize;
                for &(p, q, m) in &triples {
                    acc = acc.wrapping_add(dispatch.lcp(&text, p, q, m));
                }
                let dt = t0.elapsed();
                println!(
                    "round {r}: {:.4} s  {:.1} ns/call  sum={acc}",
                    dt.as_secs_f64(),
                    dt.as_secs_f64() * 1e9 / triples.len() as f64
                );
            }
        }
        other => panic!("unknown mode {other}"),
    }
}
