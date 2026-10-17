# 0012. One Monte Carlo engine
Status: Accepted (2026-10-16, PR #30, issue #10)

## Context
`MonteCarloEngine` copied every `MonteCarloConfig` field and converted back and forth, next to `MonteCarloEuropeanEngine`, `MonteCarloAsianEngine` and `MonteCarloDigitalEngine` that repeated the same setup.

## Decision
`MonteCarloEngine { config: MonteCarloConfig }` is the only MC engine (`quantlib_engines/src/monte_carlo_engine/mod.rs`), with `with_*` setters. Model specifics live behind a private `McModel` trait (BS, Heston); three generic `MonteCarloPricing` impls cover European, digital and Asian options. `PricingEngine::calculate_price` returns `price_with_stats(..).price`.

## Alternatives considered
- Keep the specialised engines as type aliases: they had no users outside the crate.
- One impl per product × model pair: six impls, collapsed to three in the review round.

## Consequences
- Breaking: the three specialised engines are removed; fields moved into `.config`; `sample_paths_*` return `SimulatedPaths`; `MonteCarloEngine` no longer implements `PricingEngine<AmericanOption, _>` (those stubs only returned errors).
- `chunk_size = 0` is rejected in `MonteCarloConfig::validate` and in the Python constructor.
