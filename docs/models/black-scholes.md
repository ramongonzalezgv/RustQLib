# Black–Scholes and Black-76

Code: `quantlib_models/src/black_scholes_model.rs` (`BlackScholesModel`), `quantlib_models/src/black_model.rs` (`BlackModel`), `FourierModel` impls in `quantlib_models/src/fourier_model.rs`.

## Dynamics

Black–Scholes (spot-quoted), risk-neutral:

```math
\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\,dW_t
```

Black-76 (forward-quoted), the forward is a driftless martingale and payoffs are discounted at `r`:

```math
\frac{dF_t}{F_t} = \sigma\,dW_t
```

| Model | Field | Symbol | Meaning | `validate()` |
|---|---|---|---|---|
| both | `sigma` | $\sigma$ | volatility | finite, `> 0` |
| both | `r` | $r$ | flat continuously compounded rate | finite |
| BS | `q` | $q$ | flat continuous dividend yield | finite |

## Closed form

`d1`/`d2` are methods on the models:

```math
d_1^{BS} = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad
d_1^{B76} = \frac{\ln(F/K) + \tfrac12\sigma^2 T}{\sigma\sqrt T},\qquad d_2 = d_1 - \sigma\sqrt T
```

```math
C^{BS} = S e^{-qT}N(d_1) - K e^{-rT}N(d_2),\qquad
C^{B76} = e^{-rT}\big(F N(d_1) - K N(d_2)\big)
```

Puts use $N(-d_2)$, $N(-d_1)$. Digitals (cash- and asset-or-nothing) use the same `d1`/`d2`.

Characteristic function of $X = \ln(S_T/S_0)$, or $\ln(F_T/F_0)$ for Black-76 (`black_scholes_cf`), with drift $\mu = r - q$ for BS and $\mu = 0$ for Black-76:

```math
\varphi(u) = \exp\!\big(iu(\mu - \tfrac12\sigma^2)T - \tfrac12 u^2\sigma^2 T\big)
```

Cumulants are closed form: $c_1 = (\mu - \tfrac12\sigma^2)T$, $c_2 = \sigma^2 T$, $c_4 = 0$.

Black-76 implements `FourierModel` with `FORWARD_QUOTED = true` and `dividend_yield() = r`. Fourier engines then use the forward as the "spot" with $q = r$, so there is no separate code path.

## Engines

| Engine | BS | Black-76 | Notes |
|---|---|---|---|
| [`AnalyticalEngine`](../engines/analytical.md) | European, digital, geometric Asian (continuous) | European, digital | Asian: Kemna–Vorst; arithmetic Asian and American return an error |
| [`FFTEngine` / `CosEngine`](../engines/fourier.md) | European | European | through `FourierModel` |
| [`MonteCarloEngine`](../engines/monte-carlo.md) | European, digital, Asian | — | exact log-normal step $S \leftarrow S e^{(r-q-\sigma^2/2)\Delta + \sigma\sqrt\Delta z}$; European and digital use one step |
| [`BinomialEngine`](../engines/binomial.md) | European, American | — | CRR: $u = e^{\sigma\sqrt{\Delta}}$, $d = 1/u$, $p = (e^{(r-q)\Delta} - d)/(u - d)$ |

The spot-quoted engines require `UnderlyingQuote::Spot`, and Black-76 requires `UnderlyingQuote::Forward`. A mismatch is an error (`fft_black_scholes_rejects_forward_quoted_option`, `analytical_black_requires_forward`).

## Calibration

`quantlib_utils/src/calibration.rs` (see [calibration.md](../engines/calibration.md)):

- `calibrate_black_model` fits $\sigma$ with $r$ fixed; `calibrate_black_scholes_model` fits $\sigma$ with $r, q$ fixed (rates are market data, [0020](../decisions/0020-calibration-fixed-rates.md)).
- The quote's `underlying` is the forward (Black) or the spot (BS).
- `CalibrationTarget::Price`: the model value is the closed-form price from `quantlib_engines::{black_price, black_scholes_price}`, the same functions `AnalyticalEngine` uses.
- `CalibrationTarget::Volatility` is rejected (`Validation`) until implied-vol calibration (#19): a constant-vol model would only fit a flat vol.
- Objective: weighted least squares, relative least squares or Huber.
- Optimiser: DE, Nelder–Mead, or DE→Nelder–Mead (default).
- Default bounds: $\sigma \in [10^{-4}, 3]$.

## Design choices

- Black-76 is BS on the forward with $q = r$ (#7). It reuses the Fourier engines without a forward-specific path.
- The model carries `r` and `q` as flat scalars. This is the current design; #17 moves them to curves.

## Validation

| Test | File | Reference |
|---|---|---|
| `matches_published_reference_values` | `quantlib_engines/tests/european_analytical_black_scholes.rs` | Hull: S=K=100, T=1, r=5%, σ=20% → call 10.450583572185565, put 5.573526022256971 |
| `deep_otm_put_is_positive_and_accurate` | same | mpmath, 2.1603814591587665e-31, 1e-10 rel |
| `zero_vol_limit_is_discounted_forward_intrinsic` | same | σ = 1e-8 (validate rejects 0): call = $(Se^{-qT} - Ke^{-rT})^+$, 1e-8 |
| proptests `put_call_parity`, `no_arbitrage_bounds`, `monotone_in_spot_strike_and_vol`, `convex_in_strike` | same | no-arbitrage identities |
| `matches_published_reference_value` | `quantlib_engines/tests/european_analytical_black76.rs` | Hull Black-76 put example (F=K=20), exact value 1.118075498308455 at T=122/365 |
| proptests `put_call_parity`, `no_arbitrage_bounds`, `equals_black_scholes_on_forward` | same | identities; Black-76 on $F = Se^{(r-q)T}$ equals BS |
| `fft_black_scholes_matches_analytic_on_strike_maturity_grid`, `fft_black_matches_analytic_on_strike_maturity_grid` | `quantlib_engines/tests/fft_black_and_bs_engine.rs` | analytic, ≤ 1e-6 |
| `cos_black_scholes_matches_analytic_with_128_terms`, `cos_black_prices_forward_quoted_options` | `quantlib_engines/tests/cos_engine.rs` | analytic, < 1e-10 |
| `numerical_cumulants_match_black_scholes_closed_form` | `quantlib_models/src/fourier_model.rs` | closed-form cumulants |
| `european_converges_to_black_scholes`, `american_put_matches_published_reference` | `quantlib_engines/tests/binomial_engine.rs` | analytic BS; Hull American put 4.2842 |
| `black_scholes_mc_within_three_stderr_of_analytic`, `standard_error_has_nominal_coverage` | `quantlib_engines/tests/monte_carlo_engine.rs` | analytic, 3·stderr / 95% coverage |
| `calibrates_black_scholes_from_price_quotes`, `calibrates_black_from_price_quotes`, `calibrates_black_from_implied_vol_quotes` | `quantlib_utils/src/calibration.rs` | parameter recovery on synthetic quotes |
| `rejects_nan_inputs` | both model files | validation |

## Known limitations

- `r`, `q` and $\sigma$ are flat; there is no term structure (#17).
- No implied-vol inversion, so the vol-target calibration only fits a flat $\sigma$ (#19).
- No Greeks (#18).
- American options only through the CRR tree; ALO is pending (#13).
