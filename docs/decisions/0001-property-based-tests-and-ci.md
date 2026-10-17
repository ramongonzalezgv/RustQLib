# 0001. Property-based and reference tests; CI on three OSes plus Python
Status: Accepted (2026-10-11, PR #22, issue #5)

## Context
The CSV fixture tests (deleted in 6cbc77c) compared against outputs of the old Python port, so bugs shared by both (e.g. #1) passed. CI pointed at a non-existent `working-directory` and never ran; PyO3 0.23 did not build on Python 3.14.

## Decision
- Tests check mathematics, not recorded outputs: no-arbitrage invariants (parity, bounds, monotonicity, convexity in K), known limits (σ → 0, Heston σ_v → 0 vs BS with integrated variance), cross-engine agreement, and published values inline with a citation.
- `proptest` (512 cases) for analytical invariants; Monte Carlo checked against `k · stderr`, never fixed tolerances.
- CI: `fmt --check`, `clippy -D warnings`, `test --workspace`, `bench --no-run` on Ubuntu, macOS, Windows (Windows from #23); a Python job builds the wheel with maturin and runs pytest on 3.12 and 3.14.
- PyO3/numpy 0.29; `quantlib_py/build.rs` adds the extension-module link args so `cargo test --workspace` needs no `--exclude`.

## Alternatives considered
- Regenerating fixtures from this library: tests would detect change, not error.
- `--exclude quantlib_py` or `default-members` (issue #5's plan): replaced by the `build.rs` link args.

## Consequences
- The binomial cache returned the European price for the matching American option; found by the new tests (fixed here, cache removed in 0008).
- Tests: `quantlib_engines/tests/{european_analytical_black_scholes,european_analytical_black76,binomial_engine,monte_carlo_engine}.rs`, shared helpers in `tests/common/mod.rs`.
- 95 tests at merge; CI is the gate for every later PR.
