# 0008. Remove the pricing cache instead of fixing it
Status: Accepted (2026-10-11, PR #28, issue #2)

## Context
`OptionValuationContext` and the private `BinomialEngine` cache returned stale prices (keys omitted valuation date and calendar), stored only a 64-bit hash, grew without bound behind a global `Mutex`, and built string keys. Issue #2 proposed typed keys and a bounded concurrent cache.

## Decision
Delete the cache. Engines are pure and do not memoise. Removed: `OptionValuationContext`, `make_cache_key`, `SimpleLruCache` (`quantlib_core/src/valuation.rs`), `FinancialProduct::get_parameters`, the `StochasticModel` trait. `BinomialEngine` is `Clone + Copy`; `price_and_levels_*` kept, uncached. Batch pricing goes through `calculate_prices_parallel`.

## Alternatives considered
- Typed full keys + `moka`/`quick_cache` (the issue's plan): a BS cache hit took 1,320 ns vs 24 ns to recompute (~55×), building a key 1,176 ns; COS Heston is 11 µs, binomial 200 steps 8.9 µs. Usual workloads do not repeat inputs (calibration, bumped Greeks, smiles).

## Consequences
- Breaking: `OptionValuationContext`, `make_cache_key`, `StochasticModel`, `FinancialProduct::get_parameters` leave the public API. Callers that reprice identical inputs memoise on keys they control.
- `quantlib_core` drops `rayon`.
- Test: `one_engine_reprices_options_that_differ_only_by_valuation_date_or_calendar` in `quantlib_engines/tests/binomial_engine.rs` (fails on the cached code).
