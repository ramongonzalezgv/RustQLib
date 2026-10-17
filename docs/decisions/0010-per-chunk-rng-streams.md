# 0010. Per-chunk RNG streams: Xoshiro256++ seeded by SplitMix64
Status: Accepted (2026-10-16, PR #30, issue #10)

## Context
Parallel MC drew all normals for a chunk on one thread from a single Box–Muller generator; only the path update ran in parallel, capping speedup (Heston 2.4× on 15 cores) and disabling parallelism for BS terminal pricing.

## Decision
- Paths are split into blocks of `MonteCarloConfig::chunk_size` (default 4096). Chunk `i` owns the stream `chunk_rng(seed, i)` = `Xoshiro256PlusPlus::seed_from_u64(splitmix64(seed ^ splitmix64(i)))` (`monte_carlo_engine/core.rs`).
- Normals from `rand_distr::StandardNormal` (ziggurat).
- Chunks run serially or with rayon and are merged in chunk order, so a seeded result depends on `chunk_size` but not on the thread count or `parallel`.

## Alternatives considered
- ChaCha8 with `set_stream`: tried first, 3.9 ns per u64, slower than the old xorshift.
- Xoshiro `jump()` per chunk or counter-based Philox (both suggested in #10): hashing `(seed, chunk)` chosen instead.

## Consequences
- BS terminal European, 1e7 paths, one thread: 130.1 → 80.1 ms (1.63×). Heston 200k × 252 parallel: 694 → 130 ms; speedup 11.5× on 15 cores (0.77 per core).
- Seeded outputs changed once (wasm `NATIVE_REFERENCE` regenerated).
- Test: `seeded_prices_are_bit_identical_for_any_thread_count` (serial and rayon pools of 1, 2, 8 threads, Heston arithmetic Asian with antithetic) in `quantlib_engines/tests/monte_carlo_engine.rs`; bench `quantlib_engines/benches/mc_bench.rs`.
