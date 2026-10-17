# Calibration

Bounded, derivative-free fitting of Black-76, Black–Scholes, SABR and Heston parameters to market quotes (models: [black-scholes.md](../models/black-scholes.md), [sabr.md](../models/sabr.md), [heston.md](../models/heston.md)). Code: `quantlib_utils/src/calibration.rs`.

## Entry points

| Function | Quotes | Model value per quote | Parameters (bounds default) |
|---|---|---|---|
| `calibrate_black_model(quotes, target, σ₀, r, bounds, config)` | `VanillaOptionCalibrationQuote` (underlying = forward) | `Price`: Black-76 price (`quantlib_engines::black_price`). `Volatility`: rejected (#19) | σ [1e-4, 3]; `r` fixed |
| `calibrate_black_scholes_model(quotes, target, σ₀, r, q, bounds, config)` | same (underlying = spot) | `Price`: BS price (`quantlib_engines::black_scholes_price`). `Volatility`: rejected (#19) | σ [1e-4, 3]; `r`, `q` fixed |
| `calibrate_sabr_model(quotes, α₀, β₀, ρ₀, ν₀, r, bounds, config)` | `SabrCalibrationQuote { forward, strike, maturity, market_volatility, weight }` | Hagan lognormal implied vol of `SabrModel::general` | α [1e-4, 3], β [0.01, 0.99], ρ [−0.999, 0.999], ν [1e-4, 3]; `r` fixed |
| `calibrate_heston_model(quotes, initial_model, bounds, config, fft_config)` | `VanillaOptionCalibrationQuote`, `Price` only | FFT price at the quote's exact float maturity | κ [1e-4, 20], θ [1e-5, 2], σ [1e-4, 5], ρ [−0.999, 0.999], v₀ [1e-5, 2]; `r`, `q` fixed from `initial_model` |

`Volatility` returns `Validation` before optimising: the constant-vol models would fit a flat vol and r, q would not enter the objective. #19 (implied-vol solver) will enable it; `target` and `CalibrationTarget` stay for that. r and q are market data (#17), so they are never calibrated, and a non-finite r or q is a `Validation` error ([0020](../decisions/0020-calibration-fixed-rates.md)). `CalibrationResult.calibrated_params` lists only the fitted names.

A bound with `lower == upper` pins that parameter. Bounds must be finite with lower ≤ upper, and the initial guess must lie inside them.

Each function returns `(Model, CalibrationResult)`. `CalibrationResult` (`quantlib_core`) holds:
- `success`, `calibrated_params`, `objective_value`, `iterations`, `message`;
- per-quote `errors`, `market_values`, `model_values`;
- unweighted `rmse`, `mape` (over quotes with |market| > 1e-10), `max_error`.

## Quotes

`VanillaOptionCalibrationQuote { underlying, strike, maturity, option_type, market_value, market_value_kind, weight }`, built with `::price(…)` or `::implied_volatility(…)`. Validation:
- underlying, strike, maturity and weight finite and > 0;
- a price ≥ 0, an implied vol > 0;
- every quote's `market_value_kind` matches the `CalibrationTarget` (`Price` ↔ `Price`, `Volatility` ↔ `ImpliedVolatility`).

Heston accepts `Price` only.

## Objective

With $e_i = \text{model}_i - \text{market}_i$ and weights $w_i$ (`ObjectiveKind`):

| Kind | $f$ |
|---|---|
| `LeastSquares` (default) | $\sum_i w_i e_i^2$ |
| `RelativeLeastSquares` | $\sum_i w_i (e_i / \text{market}_i)^2$; plain $e_i$ when $\lvert\text{market}_i\rvert \le 10^{-10}$ |
| `Huber { delta }` | $\sum_i w_i\,\ell_\delta(e_i)$, with $\ell_\delta(e) = \tfrac12e^2$ for $\lvert e\rvert \le \delta$, else $\delta(\lvert e\rvert - \tfrac12\delta)$; $\delta \ge 10^{-8}$ |

Any error inside the objective (model construction or pricing) and any non-finite value scores `1e300` (penalty), so the optimisers never see an `Err`.

**Heston objective.** Quotes are grouped once, before optimisation, by bit-exact `(maturity, underlying)` (`group_heston_strips`). Each evaluation then calls `FFTEngine::price_strip` once per group and scatters the prices back into quote order (see [fourier.md](fourier.md)).

## Optimisers

**Differential evolution** (`DifferentialEvolutionConfig`), DE/rand/1/bin (Storn & Price 1997):
- Population size is `max(population_size, 8, 6·dim)`, drawn uniformly in the bounds from `ChaCha8Rng(seed)`. Member 0 is the initial guess.
- Mutation: $x_a + F(x_b - x_c)$ with three distinct random indices ≠ the target, used in draw order (so $a$ is uniform over the other $P-1$ members).
- Binomial crossover with probability CR, forcing one random dimension. Mutants are clamped to the bounds.
- Greedy one-to-one selection, synchronous per generation.
- Trial vectors are generated serially from the seeded RNG; only objective evaluation is parallel (`parallel_objective`), so results do not depend on the thread count.
- Stops when $\mathrm{sd}(f) \le \text{tol}\,(1 + \lvert\bar f\rvert)$ over the population, or after `max_generations`.
- $F$ is clamped to [0.1, 1.9] and CR to [0.05, 1].

**Nelder–Mead** (`NelderMeadConfig`):
- Standard coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5. Every trial point is projected onto the bounds.
- Initial simplex: the guess plus one vertex per dimension, offset by `initial_simplex_scale · max(width, 1)`, with the sign flipped if clamping cancels the step.
- Stops when the value spread is ≤ tol·(1 + |f_best|) **and** the largest relative vertex distance is ≤ tol.

**`HybridGlobalLocal`** (default): DE, then Nelder–Mead from the DE best point. Iteration and evaluation counts are summed. The result counts as converged if either stage converged, and `message` says which.

## Configuration

| Struct | Field | Default | Meaning |
|---|---|---|---|
| `CalibrationConfig` | `method` | `HybridGlobalLocal` | `DifferentialEvolution`, `NelderMead` or `HybridGlobalLocal`. |
| | `objective` | `LeastSquares` | See above. |
| `DifferentialEvolutionConfig` | `max_generations` | 350 | ≥ 1. |
| | `population_size` | 48 | Raised to `max(8, 6·dim)`. |
| | `differential_weight` | 0.8 | $F$. |
| | `crossover_probability` | 0.9 | CR. |
| | `tolerance` | 1e-6 | Population objective spread. |
| | `seed` | 42 | ChaCha8 seed. |
| | `parallel_objective` | `true` | Rayon over the population. |
| `NelderMeadConfig` | `max_iterations` | 300 | |
| | `tolerance` | 1e-7 | Value and simplex size. |
| | `initial_simplex_scale` | 0.08 | Fraction of the bound width (≥ 1e-4). |
| `HestonFftConfig` | `n`, `b`, `interp` | 2048, `FFTEngine::DEFAULT_B` (200), `Cubic` | FFT grid used for Heston quotes; max error 4.7e-4 (below). |

## Heston FFT grid

The default is the cheapest power-of-two n at b = `FFTEngine::DEFAULT_B` with max absolute price error ≤ 1e-2 (S = 100, i.e. 1e-4 per unit spot). Reference: `CosEngine(2^13, L = 16)`, which agrees with FFT(2^16, 1600) to 2e-9 (`a5_reference_is_validated_against_large_fft`). Domain: K/S ∈ {0.7, …, 1.3} (step 0.1), T ∈ {0.1, 0.5, 1, 2}, r = 0.02, q = 0.01, calls and puts, (κ, θ, σ, ρ, v₀) ∈ {(2, 0.04, 0.5, −0.7, 0.04), (0.5, 0.09, 1, −0.9, 0.09), (5, 0.02, 0.3, 0, 0.01)}. Max error per grid (cubic):

| n | b | max abs error |
|---|---|---|
| 256 | 100 / 200 | 1.3 / 6.2 |
| 512 | 100 / 150 / 200 | 6.9e-2 / 0.35 / 1.3 |
| 1024 | 100 / 120 / 140 / 150 / 160 / 180 / 200 / 250 | 4.7e-2 / 1.7e-2 / 5.5e-3 / 9.5e-3 / 1.0e-2 / 1.3e-2 / 2.5e-2 / 0.12 |
| **2048** | 200 | **4.7e-4** |
| 2048 | 100 / 150 / 180 / 250 | 4.7e-2 / 7.9e-3 / 4.3e-3 / 9.6e-4 |
| 4096 | 200 | 4.7e-4 |

The previous default (1024, 180) had error 1.3e-2, above the bound. (1024, 140) passes but its neighbours do not, so it is not a robust default. Test: `a5_default_heston_fft_grid_meets_price_error_bound`.

## Performance

From PR #27: `heston_calibration_bench` (`quantlib_utils/benches/heston_calibration_bench.rs`), 5 maturities × 20 strikes, fixed evaluation count.

| | Before | After |
|---|---|---|
| `heston_objective_5x20_surface` | 32.49 ms | 1.61 ms (20.2×) |
| One objective's pricing (100 `calculate_price` vs 5 `price_strip`) | 3.79 ms | 0.181 ms |

## Design choices

- **Exact maturities.** Quotes carry float maturities. There is no day rounding and no synthetic valuation date. The old path rounded T = 1/12 and was off by 7e-3 relative.
- **One FFT strip per (maturity, underlying).** This gives the 20× speedup above. Grouping on the underlying too keeps quotes with different spots at one maturity correct.
- **No nested parallelism.** DE parallelises over the population; one objective evaluation is serial.
- **Penalty instead of errors** keeps the optimisers simple. Invalid parameter regions score 1e300.

## Validation

`quantlib_utils/src/calibration.rs` tests:

| Test | Checks |
|---|---|
| `calibrates_black_scholes_from_price_quotes`, `calibrates_black_from_price_quotes` | Recover synthetic σ with r (q) fixed, to 1e-2 / 1.5e-2. |
| `de_base_index_is_uniform` | DE base index uniform over the P−1 other members, 4·√(p(1−p)/N) over 1e5 seeded runs. |
| `calibrates_sabr_surface` | 3 maturities × 5 strikes, β pinned at 0.6, relative least squares, hybrid: α ± 3e-2, ρ ± 3e-2, ν ± 4e-2. |
| `calibrates_heston_surface_from_fft_prices` | 2 maturities × 3 strikes × call/put from FFT(512, 120), κ, θ pinned, Nelder–Mead: σ ± 0.1, ρ ± 0.08, v₀ ± 0.02. |
| `heston_calibration_prices_quotes_at_their_exact_maturity` | T = 1/12 (call, put) and 0.001 priced at exactly that T (1e-10 relative). |
| `heston_calibration_keeps_quote_order_across_maturities_and_spots` | Interleaved maturities, two spots at one maturity. |
| `supports_robust_huber_objective` | Huber objective runs and is finite (two quotes, Nelder–Mead). |
| `rejects_quote_kind_target_mismatch` | Kind/target validation. |

`quantlib_utils/tests/calibration_cleanup.rs`:

| Test | Checks |
|---|---|
| `a1_volatility_target_rejected_for_black_and_black_scholes`, `non_finite_rates_are_validation_errors` | `Validation` errors (message names #19). |
| `a2_*` | Round trips return σ within 1e-4 with r, q exactly as passed, and only `sigma` in `calibrated_params`; Heston keeps `initial_model`'s r, q. |
| `api_names_and_default_bounds` | Parameter names and default bounds. |
| `a4_*` | Calibration model values equal `AnalyticalEngine` prices to 1e-12; `calibration.rs` has no pricing formula. |
| `a5_*` | FFT reference check and the default-grid error bound. |

Python: `quantlib_py/tests/test_calibration_bindings.py` (Black, BS, SABR, partial Heston).

## Limitations and open issues

- Derivative-free only. DE needs thousands of evaluations, and Nelder–Mead stalls on the κ–θ valley. Levenberg–Marquardt with analytic Heston gradients (Cui et al. 2017) and a DE→LM hybrid: #9. Parameter standard errors are also part of #9.
- SABR calibrates only `SabrModel::general` (β strictly inside (0, 1)). Normal, lognormal and shifted SABR cannot be calibrated, and `r` is fixed.
- `Volatility` target for Black and BS is rejected until the implied-vol solver (#19).
- Heston calibrates five parameters (r, q fixed). On small surfaces κ and θ are weakly identified; the Heston test pins them.
- Heston pricing uses the FFT, not the faster COS engine.
- Bates and Kou have no calibration. No calibration to American quotes (#13).
