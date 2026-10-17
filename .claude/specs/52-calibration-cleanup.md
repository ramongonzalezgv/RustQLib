# #52 Calibration: Volatility target is a flat-vol fit, DE index sort, duplicated closed forms, coarse FFT grid

All code in `quantlib_utils/src/calibration.rs` unless stated. Python bindings in `quantlib_py/src/functions.rs` (`calibrate_black`, `calibrate_black_scholes`, `calibrate_heston`) follow every Rust API change.

## Maths

No new maths.

1. **Volatility target.** For Black-76 and Black–Scholes the model implied vol is the constant σ for every quote, so `CalibrationTarget::Volatility` fits a weighted flat vol and r, q never enter the objective. Implied-vol calibration needs an implied-vol solver (#19, open). Until then the `Volatility` target is rejected for these constant-vol models.
2. **Rates are inputs, not calibrated.** r and q are market data (#17), not model parameters to fit. Black calibrates σ only, with r fixed. Black–Scholes calibrates σ only, with r and q fixed. Heston calibrates (κ, θ, σ, ρ, v₀), with r and q taken unchanged from `initial_model`.
3. **DE/rand/1** (Storn & Price 1997): mutant `v = x_a + F (x_b − x_c)`, where a, b, c are distinct, differ from the target index, and are drawn uniformly **in draw order**. Delete `indices.sort_unstable()` (it makes `a` always the smallest of the three indices). `random_three_distinct_indices` is already uniform. Seeded runs stay deterministic, but their outputs may change.
4. **One closed form per model.** `black_price` and `black_scholes_price` in `calibration.rs` duplicate `AnalyticalEngine::price_black` and `price_black_scholes` (`quantlib_engines/src/analytical_engine.rs`). Calibration quotes carry a float maturity, not dates. Keep a single implementation in `quantlib_engines` that works on `(underlying, strike, ttm, option_type, model)` and call it from both the `EuropeanOption` engine path and calibration. Delete the calibration copies and the calibration `intrinsic` helper if it becomes unused.
5. **FFT default by measured error.** The `HestonFftConfig::default()` grid (n, b) is the cheapest power-of-two n, with b in the same family as `FFTEngine::default()` (`DEFAULT_N` = 4096, `DEFAULT_B` = 200), such that the max absolute call/put price error stays at or below **1e-4 per unit spot (S = 100, so ≤ 1e-2)**. The error is measured against a reference on the domain below:
   - Reference: `CosEngine` with a large N (or FFT with n = 2^16), checked against FFT at that size to agree within 1e-7 before use.
   - Domain: strikes K/S ∈ {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3}, maturities T ∈ {0.1, 0.5, 1, 2}, r = 0.02, q = 0.01. Three Heston parameter sets: (κ, θ, σ, ρ, v₀) = (2, 0.04, 0.5, −0.7, 0.04), (0.5, 0.09, 1.0, −0.9, 0.09) and (5, 0.02, 0.3, 0.0, 0.01).
   - If no grid cheaper than `FFTEngine::default()` meets the bound, make the default equal `FFTEngine::default()` by reusing `DEFAULT_N` and `DEFAULT_B`, not by copying the literals.

## API

```rust
pub const BLACK_PARAMETER_NAMES: [&str; 1] = ["sigma"];
pub const BLACK_SCHOLES_PARAMETER_NAMES: [&str; 1] = ["sigma"];
pub const HESTON_PARAMETER_NAMES: [&str; 5] = ["kappa", "theta", "sigma", "rho", "v0"];
pub fn default_black_bounds() -> [(f64, f64); 1];          // [(1e-4, 3.0)]
pub fn default_black_scholes_bounds() -> [(f64, f64); 1];  // [(1e-4, 3.0)]
pub fn default_heston_bounds() -> [(f64, f64); 5];         // existing first five entries

pub fn calibrate_black_model(quotes, target, initial_sigma: f64, r: f64,
    bounds: Option<[(f64, f64); 1]>, config) -> Result<(BlackModel, CalibrationResult), QuantLibError>;
pub fn calibrate_black_scholes_model(quotes, target, initial_sigma: f64, r: f64, q: f64,
    bounds: Option<[(f64, f64); 1]>, config) -> Result<(BlackScholesModel, CalibrationResult), QuantLibError>;
pub fn calibrate_heston_model(quotes, initial_model: HestonModel,
    bounds: Option<[(f64, f64); 5]>, config, fft_config: HestonFftConfig) -> ...; // r, q from initial_model, fixed
```

Keep the `target` parameter and `CalibrationTarget` so that #19 can enable `Volatility` without another API break. `CalibrationResult` parameters list only the calibrated names.

Python: `calibrate_black(..., initial_sigma, r, sigma_bounds, ...)` (no `r_bounds`). `calibrate_black_scholes(..., initial_sigma, r, q, sigma_bounds, ...)` (no `r_bounds` or `q_bounds`). `calibrate_heston(..., r, q, ...)` replaces `initial_r` / `initial_q` and drops `r_bounds` / `q_bounds`. Update `quantlib_py/README.md`, the walkthrough notebook, `quantlib_utils/README.md` and `quantlib_utils/benches/heston_calibration_bench.rs`.

## Validation

- `calibrate_black_model` and `calibrate_black_scholes_model` with `CalibrationTarget::Volatility` return `QuantLibError::Validation` before optimising, with a message naming implied-vol calibration (#19).
- A non-finite `r` or `q` returns `QuantLibError::Validation` (model constructors may already do this; validate once).

## Acceptance criteria

- **A1.** `Volatility` target is rejected for Black and BS with `QuantLibError::Validation`. Oracle: error variant match.
- **A2.** r and q are fixed inputs. A BS calibration to BS prices generated with σ = 0.25, r = 0.03, q = 0.01, passing r = 0.03 and q = 0.01, returns σ within 1e-4. The returned model's r and q equal the inputs exactly, and `CalibrationResult` lists only `sigma`. The same holds for Black (σ, r). For Heston, the returned model's r and q equal `initial_model`'s exactly. Oracle: closed form round trip.
- **A3.** DE base index is uniform. Over N ≥ 1e5 draws from the mutation-index selection for population P = 10 and a fixed target index, each allowed index appears as the base `a` with frequency within 4·√(p(1−p)/N) of p = 1/(P−1). Oracle: binomial bound. The test fails on the sorted version, where index 0 dominates. Expose the selection only as a private helper testable from the in-file `tests` module.
- **A4.** One closed form per model: `calibration.rs` contains no Black or BS pricing formula (no `norm_cdf` call), and calibration prices equal `AnalyticalEngine` prices for the same option to 1e-12. Oracle: equality with the engine.
- **A5.** The `HestonFftConfig::default()` max abs price error on the domain in Maths item 5 is ≤ 1e-2 (S = 100). A test asserts this bound, and the measured max error per grid tried is recorded in `docs/engines/calibration.md`. Oracle: reference prices.
- **A6.** Existing calibration tests (Rust and `pytest -q quantlib_py/tests`) pass after being updated to the new signatures. Seeded expectations may move because of item 3. Loosen a tolerance only with a stated reason.

## Docs

- Decision record `docs/decisions/0020-calibration-fixed-rates.md`: r and q are inputs, the `Volatility` target is rejected until #19, DE/rand/1 uses unsorted indices, and the FFT default is chosen by the measured bound. Give alternatives rejected (removing `CalibrationTarget`, keeping r and q calibratable) and consequences (Rust and Python API break, seeded outputs change).
- `docs/engines/calibration.md`: update the signature table, bounds, the `HestonFftConfig` default with the measured errors, and the tests table. Remove the Known-limitations bullets that cite #52.
- `docs/models/black-scholes.md`, `docs/models/heston.md`, `docs/models/sabr.md`, `docs/engines/fourier.md`: fix any calibration signature or #52 references.
- `CHANGELOG.md` Unreleased: a `Changed` line (breaking: r/q fixed inputs, `Volatility` target rejected, new FFT default) and a `Fixed` line (DE/rand/1), each with #52.
