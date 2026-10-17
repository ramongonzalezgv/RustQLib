# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No release has been tagged yet; the workspace version is 0.1.0.

## [Unreleased]

### Added
- Baseline: core types, European/American/digital/Asian products, BS, Black-76, Heston, Bates, Kou and SABR models, analytical/FFT/binomial/MC engines, Python bindings, calibration with DE, Nelder–Mead and hybrid optimisers (initial commits).
- Property-based (`proptest`) and reference tests replacing the CSV fixtures; CI on Linux, macOS, Windows and Python 3.12/3.14 (#22).
- `quantlib_core::math`: `norm_cdf`, `norm_pdf`, `inv_norm_cdf` (#23).
- Web pricing playground (`quantlib_wasm`, `web/`), deployed to GitHub Pages (#24).
- `MonteCarloEngine::sample_paths_black_scholes` / `sample_paths_heston` (#24).
- `FourierModel` trait for BS, Black-76, Heston, Bates and Kou; `FFTEngine::price_strip`, `DEFAULT_N`, `DEFAULT_B` (#26).
- `CosEngine` (Rust and Python) (#26).
- Heston calibration benchmark (#27).
- `HestonScheme::{QuadraticExponential, FullTruncation}`; Python `heston_scheme` (#29).
- `price_with_stats` → `McEstimate { price, std_error, n_paths }`; Python `*_with_stats` and `McEstimate` (#30).
- Antithetic and control variates (`antithetic`, `control_variate`), off by default (#30).
- `geometric_asian_black_scholes`; `AnalyticalEngine` prices continuously monitored geometric Asians (#30).
- `HolidayCalendar`, `WeekendsOnly`, `HolidayList`, `BusinessDayConvention`, `Schedule` (#47).
- Docs: decision records, model and engine pages, AGENTS.md documentation rules; rustdoc warnings fail CI (#53).

### Changed
- PyO3 and numpy 0.23 → 0.29 (Python 3.14 support) (#22).
- FFT engine: one generic impl, plan built once, local 8-point Lagrange interpolation, faster Heston CF (default Heston bench 1.89×); `FFTEngine` is `Clone`, not `Copy`, fields read via `n()`, `b()`, `interp()` (#26).
- Heston calibration prices quotes at exact maturities with one FFT strip per maturity (20× faster objective) (#27).
- `BinomialEngine` is `Clone + Copy` and uncached (#28).
- Heston MC defaults to the QE scheme; Heston MC prices change (#29).
- MC: per-chunk Xoshiro256++ streams (thread-count independent), streaming payoffs, `MonteCarloEngine { config }`; seeded MC outputs change; `sample_paths_*` return `SimulatedPaths` (#30).
- Playground MC standard error comes from one run (#30).

### Removed
- `BinomialLevels::level1` / `level2`, which nothing read (#51).
- CSV fixture tests and `tests/scratchpad.rs` (#22).
- System libm FFI (`unsafe extern "C" fn erf`) and duplicate `norm_cdf` copies (#23).
- `OptionValuationContext`, `make_cache_key`, `SimpleLruCache`, `StochasticModel`, `FinancialProduct::get_parameters` (#28).
- Heston variance-absorption Euler scheme (#29).
- `MonteCarloEuropeanEngine`, `MonteCarloAsianEngine`, `MonteCarloDigitalEngine`; `PricingEngine<AmericanOption, _>` stubs on `MonteCarloEngine` (#30).

### Fixed
- Binomial cache returned the European price for the matching American option (#22).
- CI `working-directory` pointed to a non-existent folder (#22).
- `norm_cdf` left-tail precision (Φ(−9) was 0) and Windows/WASM linking (#23).
- SABR implied vols now follow Hagan (2002) eq. (2.17a) for general, lognormal and normal variants; stable `x(z)`; |ρ| = 1 rejected (#25).
- Heston calibration: hardcoded valuation date and day-rounded maturities (#27).
- Stale cached prices for options differing only by valuation date or calendar (#28).
- Heston MC bias when the Feller condition fails (#29).
- Biased Heston control variate under full truncation; it is now applied only with QE (#30).
- Heston control variate was ignored under full truncation, whose spot step is an exact martingale; it now applies under both schemes (#50).
- Shifted SABR is now priced on `F + s`, `K + s`; it was priced on unshifted `F`, `K`, giving wrong prices or NaN (#48).
- Heston CF with σ = 0: κ → 0 limit returned θT instead of v0·T integrated variance; κ = 0 now matches Black–Scholes with vol √v0 (#49).
- Binomial: `qty` is applied after the tree, so short American options no longer pick the wrong exercise branch; `BinomialEngine::default()` uses 100 steps (was 0, silently 1) and `n_steps = 0` is rejected (#51).
