# Analytical engine

Closed-form prices for European vanillas (Black–Scholes, Black-76, SABR), cash/asset digitals (Black–Scholes, Black-76) and geometric Asians (Black–Scholes). Models: [black-scholes.md](../models/black-scholes.md), [sabr.md](../models/sabr.md). Code: `quantlib_engines/src/analytical_engine.rs`, `quantlib_engines/src/batch.rs`.

## Formulas

All prices are per unit and then multiplied by `qty`. If $T \le 0$ the engine returns the undiscounted payoff on today's spot (forward for Black-76 and SABR).

**Black–Scholes** (spot-quoted, continuous dividend yield $q$):

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad d_2 = d_1 - \sigma\sqrt T$$

$$C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2),\qquad P = Ke^{-rT}N(-d_2) - Se^{-qT}N(-d_1)$$

**Black-76** (forward-quoted): $d_{1,2} = \dfrac{\ln(F/K) \pm \tfrac12\sigma^2T}{\sigma\sqrt T}$, $C = e^{-rT}\big(FN(d_1) - KN(d_2)\big)$, $P = e^{-rT}\big(KN(-d_2) - FN(-d_1)\big)$.

**Digitals** (payout $A$, $\omega = +1$ call, $-1$ put):

| Payoff | Black–Scholes | Black-76 |
|---|---|---|
| Cash-or-nothing | $A e^{-rT} N(\omega d_2)$ | $A e^{-rT} N(\omega d_2)$ |
| Asset-or-nothing | $A S e^{-qT} N(\omega d_1)$ | $A e^{-rT} F N(\omega d_1)$ |

**SABR** (Hagan et al. 2002). The model's implied volatility $\sigma_{\text{impl}}(F, K, T)$ is plugged into a closed form:
- $\beta = 0$ (`|β| < 1e-12`): Hagan normal vol into Bachelier, $d = (F-K)/(\sigma_N\sqrt T)$, $C = e^{-rT}\big((F-K)N(d) + \sigma_N\sqrt T\,n(d)\big)$, $P = e^{-rT}\big((K-F)N(-d) + \sigma_N\sqrt T\,n(d)\big)$.
- otherwise: Hagan lognormal vol (eq. 2.17a) into Black-76.

The SABR implied-vol formulas are in [sabr.md](../models/sabr.md).

**Geometric Asian** (Kemna & Vorst 1990), `geometric_asian_black_scholes(spot, strike, ttm, model, option_type, n_steps)`. With `n_steps = Some(n)` the average runs over the $n+1$ equally spaced fixings $S_0, S_{T/n}, \dots, S_T$, the same monitoring as the Monte Carlo engine. With `None` it is continuous. $\ln G$ is normal:

$$\mu_G = \ln S_0 + (r - q - \tfrac12\sigma^2)\tfrac T2,\qquad \sigma_G^2 = \sigma^2T\,\frac{2n+1}{6(n+1)} \;\xrightarrow{n\to\infty}\; \frac{\sigma^2T}{3}$$

The price is Black-76 on $F_G = e^{\mu_G + \sigma_G^2/2}$ with total variance $\sigma_G^2$, discounted at $e^{-rT}$.

## Implementation map

| Item | Role |
|---|---|
| `AnalyticalEngine` | Zero-sized engine; one `PricingEngine<P, M>` impl per product/model pair. |
| `price_black_scholes`, `price_black` | Vanilla closed forms; spot from `require_spot()`, forward from `require_forward()`. |
| `price_digital_black_scholes`, `price_digital_black` | Digital closed forms above. |
| `price_sabr` → `price_sabr_normal` / `price_sabr_lognormal` | Branch on `model.beta()`. |
| `geometric_asian_black_scholes` (pub) | Discrete or continuous geometric Asian, unit quantity. Also the Monte Carlo control-variate mean. |
| `PricingEngine<AsianOption, BlackScholesModel>` | Geometric averaging only, **continuous** monitoring (`None`). Arithmetic averaging returns an error that points to `MonteCarloEngine`. |
| `batch::calculate_prices_parallel` | Prices `&[P]` under one model with Rayon `par_iter`. Every product is priced; the first error in input order is returned. |
| `batch::calculate_prices_parallel_per_model` | Same, zipping products with models; errors if the lengths differ. |

Supported pairs:

| Product \ Model | `BlackScholesModel` | `BlackModel` | `SabrModel` |
|---|---|---|---|
| `EuropeanOption` | yes | yes (forward) | yes (forward) |
| `DigitalOption` | yes | yes (forward) | error |
| `AsianOption` | geometric, continuous | error | error |
| `AmericanOption` | error | error | error |

## Configuration

None. `AnalyticalEngine` has no fields.

## Design choices

- Black-76 and SABR take forward-quoted products, BS takes spot-quoted ones. A mismatch is a validation error (`require_forward` / `require_spot`). The engine never converts between spot and forward.
- The geometric Asian closed form takes the fixing count, so it matches the MC engine's discrete monitoring exactly and can serve as its control variate (see [monte-carlo.md](monte-carlo.md)). The engine impl uses the continuous limit because `AsianOption` has no fixing schedule.
- Unsupported pairs return `QuantLibError::Engine` instead of being left unimplemented, so generic callers (Python, wasm) get a uniform error.

## Validation

| Test | File | Checks |
|---|---|---|
| `matches_published_reference_values` | `tests/european_analytical_black_scholes.rs` | S=K=100, T=1, r=5%, σ=20%: call 10.450583572, put 5.573526022 (1e-8). |
| `zero_vol_limit_is_discounted_forward_intrinsic`, `deep_otm_put_is_positive_and_accurate` | same | σ→0 limit (1e-8); K/S = 0.2 put = 2.1604e-31 to 1e-10 relative (`norm_cdf` tail). |
| `put_call_parity`, `no_arbitrage_bounds`, `monotone_in_spot_strike_and_vol`, `convex_in_strike` | same (proptest) | Arbitrage identities on random inputs. |
| `matches_published_reference_value` | `tests/european_analytical_black76.rs` | Hull Black-76 put, F=K=20, r=9%, σ=25%, T=122/365: 1.118075498 (1e-9). |
| `put_call_parity`, `no_arbitrage_bounds`, `equals_black_scholes_on_forward` | same (proptest) | Parity, bounds; Black-76 on $F = Se^{(r-q)T}$ = BS (1e-9·S). |
| `put_call_parity`, `calls_are_decreasing_and_convex_in_strike` | `tests/european_analytical_sabr.rs` | General, lognormal and normal SABR, F = 100, 50 strikes in [60, 158]. |
| `analytical_prices_cash_digital_bs`, `analytical_prices_asset_digital_black` | `src/analytical_engine.rs` | Cash call (BS) and asset put (Black-76) with `qty` scaling (1e-10). |
| `analytical_black_requires_forward`, `analytical_engine_rejects_american_option`, `analytical_engine_rejects_arithmetic_asian_option` | same | Error paths. |
| `geometric_asian_closed_form_matches_monte_carlo` | `tests/monte_carlo_engine.rs` | Discrete closed form vs 1e6-path MC within 3·stderr, call/put × n ∈ {12, 52, 252}. |
| `analytical_engine_prices_continuous_geometric_asians_only` | same | Engine = continuous form; n = 100 000 within 1e-4 of it. |
| `parallel_batch_matches_individual_pricing` | `src/batch.rs` | Batch = per-option prices (1e-12). |

## Limitations and open issues

- No American closed form or approximation: #13 (Andersen–Lake–Offengenden). Use [binomial.md](binomial.md).
- Hot path: `d2()` recomputes `d1()`, every call re-runs `validate()`, and the batch path is per-option structs rather than a SIMD kernel: #14.
- No Greeks: #18.
- Shifted SABR prices Black-76 on the unshifted $F, K$ with the vol implied on $F+s, K+s$. That is only correct for a normal (β = 0) base model, and gives NaN for $F \le 0$ or $K \le 0$: #48.
- The geometric Asian ignores past fixings: at $T \le 0$ it returns the intrinsic value of the spot, not of an average.
