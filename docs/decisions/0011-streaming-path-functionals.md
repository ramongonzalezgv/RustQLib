# 0011. Streaming path functionals and one step kernel per model
Status: Accepted (2026-10-16, PR #30, issue #10)

## Context
Asian pricing stored every path as `Vec<Vec<f64>>` (1e6 × 252 ≈ 2 GB) although the payoff needs only a running sum. The Heston step was written four times and BS four times.

## Decision
In `quantlib_engines/src/monte_carlo_engine/paths.rs`, one generic driver `simulate<M: PathModel, F: PathFunctional>`:
- `PathModel`: `initial`, `step`, `spot`. `BlackScholesPath` is the exact log-normal step; `HestonPath` wraps the single Heston stepper (0009). BS European and digital use one exact step.
- `PathFunctional`: `start`, `observe`, `finish` → `Sample { y, x }` (payoff, control). `Terminal` and `Average` (running arithmetic sum, log-sum only when the geometric average is needed).
- Each chunk accumulates division-free sums shifted by its first sample; chunks merge with Chan et al.'s formula.
- No path storage. `record_paths` → flat row-major `SimulatedPaths` only for `sample_paths_*` (playground charts).

## Alternatives considered
- Plain `(sum, sum_sq, n)` per chunk (#10's plan): cancels when the mean is large against the spread; shifted sums avoid it.

## Consequences
- Arithmetic Asian 1e6 × 252: peak RSS 2.10 GB → 3.1 MB, wall 2.13 → 0.36 s; 100k × 252 parallel 177.7 → 22.2 ms.
- `paths.rs` is 311 lines vs 434 in `simulators.rs` (issue asked for −50%; the module also carries antithetic pairing and path recording).
- Barriers and lookbacks are new `PathFunctional`s, not new simulators.
- Tests: `tests/mc_memory.rs` (peak heap growth < 1 MB for 1e6 paths), `merged_chunk_moments_match_one_pass` in `monte_carlo_engine/mod.rs`.
