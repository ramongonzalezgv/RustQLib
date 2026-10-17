# 0007. Heston calibration at exact maturities, one FFT strip per maturity
Status: Accepted (2026-10-11, PR #27, issue #4)

## Context
The objective built synthetic options on a hardcoded 2026-01-01 valuation date and rounded maturities to days (T = 1/12 priced at 0.08219, T < 1/365 clamped up). It ran one full FFT per quote although quotes at one maturity share a grid.

## Decision
In `quantlib_utils/src/calibration.rs`:
- `group_heston_strips` groups quotes once, before optimisation, by bit-exact `(maturity, underlying)`; the underlying is in the key so different spots at one maturity stay correct.
- The objective calls `FFTEngine::price_strip` once per group at the exact float maturity and scatters prices back into quote order.
- `build_heston_options` and `maturity_to_days` are deleted; `quantlib_utils` no longer depends on `quantlib_products`.
- DE population-level parallelism kept; no nested parallelism inside one objective.

## Alternatives considered
- COS for calibration (proposed in #8): not done; FFT strips met the 15× target.

## Consequences
- 5 × 20 surface objective 32.49 → 1.61 ms (20.2×, on top of 0005); the existing recovery test 0.61 → 0.11–0.14 s.
- Tests: `heston_calibration_prices_quotes_at_their_exact_maturity` (T = 1/12 and T = 0.001 within 1e-10; old code 7.0e-3), `heston_calibration_keeps_quote_order_across_maturities_and_spots`, `calibrates_heston_surface_from_fft_prices`; bench `quantlib_utils/benches/heston_calibration_bench.rs`.
