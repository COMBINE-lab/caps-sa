//! Benchmark: `LcpDispatch::lcp` (byte-view SIMD) vs `lcp_scalar` across
//! symbol widths. Confirms the Symbol-trait byte-view dispatch picks up
//! the AVX-512 fast path for u16/u32/u64/[u8; 3] in addition to u8.
//!
//! Run with: `cargo run --release --example alphabet_bench`.

use std::hint::black_box;
use std::time::Instant;

use caps_sa::{LcpDispatch, Symbol, lcp_scalar};

const N: usize = 1_000_000;
const ITERS: usize = 1_000;

/// Time a tight loop of LCP calls into `text` between two starting
/// offsets. Both pointers shift by 1 each iter to defeat any caching
/// of the answer; `black_box` defeats LLVM's attempts to hoist the
/// dispatched call out of the loop.
fn bench<S: Symbol>(label: &str, text: &[S]) {
    let dispatch = LcpDispatch::detect();
    let q_base = text.len() / 2;

    let t = Instant::now();
    let mut acc = 0usize;
    for i in 0..ITERS {
        let p = i % (text.len() / 4);
        let q = q_base + (i % (text.len() / 4));
        acc = acc.wrapping_add(dispatch.lcp(black_box(text), p, q, usize::MAX));
    }
    let dt_simd = t.elapsed();

    let t = Instant::now();
    let mut acc2 = 0usize;
    for i in 0..ITERS {
        let p = i % (text.len() / 4);
        let q = q_base + (i % (text.len() / 4));
        acc2 = acc2.wrapping_add(lcp_scalar(black_box(text), p, q, usize::MAX));
    }
    let dt_scalar = t.elapsed();

    assert_eq!(acc, acc2, "SIMD and scalar disagree on {label}");
    let speedup = dt_scalar.as_secs_f64() / dt_simd.as_secs_f64();
    println!(
        "{label:<12}  size {size}B/sym  scalar {scalar:>9.3} ms  SIMD {simd:>9.3} ms  speedup {speedup:>5.1}×",
        size = std::mem::size_of::<S>(),
        scalar = dt_scalar.as_secs_f64() * 1e3,
        simd = dt_simd.as_secs_f64() * 1e3,
    );
}

fn main() {
    // Long-LCP regime: long stretches of identical symbols, then a
    // single difference. Maximises the per-call work and shows the
    // SIMD-vs-scalar ratio. (Random short-LCP regimes resolve in 1-2
    // iterations of either path, so the speedup is much smaller.)
    println!("== alphabet_bench — long-LCP regime (1M symbols, 1k iters) ==\n");

    let mut u8_text = vec![0u8; N];
    u8_text[N - 1] = 1;
    bench("u8", &u8_text);

    let mut u16_text = vec![0u16; N];
    u16_text[N - 1] = 1;
    bench("u16", &u16_text);

    let mut u24_text = vec![[0u8; 3]; N];
    u24_text[N - 1] = [0, 0, 1];
    bench("[u8; 3]", &u24_text);

    let mut u32_text = vec![0u32; N];
    u32_text[N - 1] = 1;
    bench("u32", &u32_text);

    let mut u64_text = vec![0u64; N];
    u64_text[N - 1] = 1;
    bench("u64", &u64_text);
}
