# 0020. Calibration: fixed rates, no flat-vol target, unsorted DE, measured FFT default
Status: Accepted (2026-10-17, issue #52)

## Context
- The `Volatility` target for Black and BS compared the constant `sigma` with every quoted vol, so it fitted a weighted flat vol and r, q never entered the objective.
- r and q were calibrated like model parameters, though they are market data (curves, #17).
- DE sorted the three mutation indices, so the base vector was always the smallest draw, not uniform as in DE/rand/1 (Storn & Price 1997).
- `black_price` and `black_scholes_price` duplicated the `AnalyticalEngine` closed forms.
- The Heston FFT default (1024, 180) was coarser than `FFTEngine::default()` and its error was never measured.

## Decision
- r and q are inputs. Black calibrates σ with r fixed, BS σ with r, q fixed, Heston (κ, θ, σ, ρ, v₀) with r, q taken from `initial_model`. Non-finite r or q is a `Validation` error.
- `CalibrationTarget::Volatility` returns `Validation` for Black and BS until an implied-vol solver exists (#19). The `target` argument and enum stay, so #19 does not break the API again.
- DE draws a, b, c uniformly in draw order, no sort.
- `quantlib_engines::{black_price, black_scholes_price}(underlying, strike, ttm, option_type, model)` is the single closed form, called by the `EuropeanOption` engine path and by calibration.
- `HestonFftConfig::default()` is the cheapest power-of-two n at `b = FFTEngine::DEFAULT_B` whose max absolute price error (S = 100) is ≤ 1e-2 against a COS reference on the grid in [calibration.md](../engines/calibration.md): n = 2048.

## Alternatives rejected
- Remove `CalibrationTarget`: #19 would break the API a second time.
- Keep r, q calibratable: they are weakly identified and not model parameters; pinning them with equal bounds was the workaround in every test.
- Keep the flat-vol fit: it is a mean of the quotes, not a calibration.
- n = 1024 with b = 140 (error 5.5e-3): it passes only in a narrow b window (b = 120, 160 and 180 exceed 1e-2), so it is not robust.

## Consequences
- Rust and Python API break: Black/BS/Heston take fixed r (q), bounds arrays shrink (1, 1, 5), `*_PARAMETER_NAMES` lose r and q, Python drops `r_bounds`/`q_bounds` and renames `initial_r`/`initial_q` to `r`/`q`.
- Seeded DE and hybrid outputs change.
- The default Heston calibration grid is 2× larger (n = 2048), so objective evaluations cost more.
- Tests: `quantlib_utils/tests/calibration_cleanup.rs` (A1–A5), `de_base_index_is_uniform` in `calibration.rs`.
