# SABR

Code: `quantlib_models/src/sabr_model.rs` (`SabrModel` with variants `General`, `Normal`, `Lognormal`, `Shifted`). Pricing: `quantlib_engines/src/analytical_engine.rs`.

## Dynamics

Hagan, Kumar, Lesniewski & Woodward (2002), "Managing Smile Risk", forward measure:

```math
dF_t = \alpha_t F_t^{\beta}\,dW_t,\qquad d\alpha_t = \nu\,\alpha_t\,dZ_t,\qquad d\langle W, Z\rangle_t = \rho\,dt,\qquad \alpha_0 = \alpha
```

Variants:

- **General**: $\beta \in (0, 1)$.
- **Normal**: $\beta = 0$.
- **Lognormal**: $\beta = 1$.
- **Shifted**: any base variant applied to $F + s$ and $K + s$.

| Field | Symbol | Meaning | Validation (`new` and `SabrModel::validate`) |
|---|---|---|---|
| `alpha` | $\alpha$ | initial vol level | finite, `> 0` |
| `beta` (General only) | $\beta$ | CEV exponent | finite, strictly in `(0, 1)`; Normal fixes 0, Lognormal fixes 1 |
| `rho` | $\rho$ | forward/vol correlation | finite, strictly in `(-1, 1)` ($x(z)$ divides by $1 \mp \rho$) |
| `nu` | $\nu$ | vol of vol | finite, `> 0` |
| `r` | $r$ | flat discount rate | finite |
| `f0` | $F_0$ | optional stored forward | finite if present; not used in pricing |
| `shift` (Shifted) | $s$ | displacement | finite, `> 0`; base model must validate |

`implied_volatility(F, K, T)` also requires $F, K, T$ finite and $F, K, T > 0$. For Shifted, $F+s$ and $K+s$ must be $> 0$.

## Implemented formulas

**General and Lognormal** (`hagan_lognormal_vol`): Black implied vol, Hagan (2002) eq. (2.17a). Lognormal passes $\beta = 1$.

```math
\sigma_B(K) = \frac{\alpha}{(FK)^{\frac{1-\beta}{2}}\Big[1 + \frac{(1-\beta)^2}{24}\ln^2\frac FK + \frac{(1-\beta)^4}{1920}\ln^4\frac FK\Big]}
\cdot\frac{z}{x(z)}\cdot
\Big[1 + \Big(\frac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{\frac{1-\beta}{2}}} + \frac{2-3\rho^2}{24}\nu^2\Big)T\Big]
```

```math
z = \frac{\nu}{\alpha}(FK)^{\frac{1-\beta}{2}}\ln\frac FK,\qquad
x(z) = \ln\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}
```

At $K = F$ this reduces to the ATM formula (2.18) without a separate branch.

**Normal** (`hagan_normal_vol`): normal (Bachelier) vol from Hagan's normal expansion at $\beta = 0$. The $-\beta(2-\beta)\alpha^2$ and $\rho\beta\nu\alpha/4$ terms vanish:

```math
\sigma_N(K) = \alpha\,\frac{z}{x(z)}\Big[1 + \frac{2-3\rho^2}{24}\nu^2T\Big],\qquad z = \frac{\nu}{\alpha}(F - K)
```

**$z/x(z)$** (`z_over_x`):

- For $|z| < 10^{-6}$ it uses the Taylor series $1 - \tfrac12\rho z + (2 - 3\rho^2)z^2/12$.
- Otherwise it evaluates $x(z)$ with `ln_1p` on a cancellation-free argument, using $\sqrt D - 1 = (z^2 - 2\rho z)/(\sqrt D + 1)$.
- It has a separate branch for $z < 0$ via $(\sqrt D + z - \rho)(\sqrt D - z + \rho) = 1 - \rho^2$.
- So the smile is one continuous formula across the money.

**Shifted**: `implied_volatility(F, K, T)` returns the base model's vol at $(F + s, K + s, T)$.

## Engines

[`AnalyticalEngine`](../engines/analytical.md), European options only. It needs a forward-quoted product, and digital, American and Asian options return an error.

1. `price_sabr` sets $\sigma$ to `implied_volatility(F, K, T)` and discounts with $e^{-rT}$.
2. If `beta()` is below `1e-12` (Normal, or Shifted over Normal), the price is Bachelier: $C = e^{-rT}\big((F-K)N(d) + \sigma\sqrt T\,n(d)\big)$, with $d = (F-K)/(\sigma\sqrt T)$.
3. Otherwise the price is Black-76 with $\sigma$. For a Shifted model both steps use $F+s, K+s$ (`shift()`, summed over nested shifts; 0 otherwise), since the vol is a shifted-Black vol (#48). Bachelier is shift-invariant, so the shift only matters there for conditioning.

There are no FFT, COS, MC or binomial engines for SABR.

## Calibration

`calibrate_sabr_model` (`quantlib_utils/src/calibration.rs`; see [calibration.md](../engines/calibration.md)):

- Fits the General variant only. Parameters are $(\alpha, \beta, \rho, \nu)$, and `r` is fixed.
- The objective compares model Hagan vols (`implied_volatility`) with `SabrCalibrationQuote::market_volatility`: weighted least squares, relative least squares or Huber.
- Optimiser: DE→Nelder–Mead by default.
- Default bounds: $\alpha \in [10^{-4}, 3]$, $\beta \in [0.01, 0.99]$, $\rho \in [-0.999, 0.999]$, $\nu \in [10^{-4}, 3]$.
- Normal, Lognormal and Shifted cannot be calibrated.

## Design choices (#1, #25)

- Fixed to eq. (2.17a) exactly. The log-moneyness series is a divisor, not part of the time correction.
- General and Lognormal share one function, with $\beta = 1$ for Lognormal.
- The spurious $\rho\nu\alpha/4$ term was removed from the normal vol, because it carries a $\beta$ factor.
- `z_over_x` replaces a `numerator <= 0 → |z|` fallback and absolute ATM thresholds.
- $|\rho| = 1$ is rejected; the calibration bounds were already ±0.999.
- Tests use independent references (QuantLib, analytic limits) instead of parity with the original Python port, which shared the old bugs.

## Validation

| Test | File | Reference |
|---|---|---|
| `matches_quantlib_reference_values` | `quantlib_models/tests/sabr_hagan.rs` | 13 vols from QuantLib 1.43 `sabrVolatility` (eq. 2.17a), β ∈ {0.3, 0.5, 0.9, 1}, < 1e-12 |
| `lognormal_smile_is_flat_without_vol_of_vol`, `normal_smile_is_flat_without_vol_of_vol` | same | ν → 0 limits |
| `normal_atm_vol_has_only_the_vol_of_vol_correction` | same | $\alpha(1 + (2-3\rho^2)\nu^2T/24)$, 1e-14 |
| `general_sabr_is_continuous_at_beta_one` | same | Lognormal at β = 1 − 1e-9, 1e-7 |
| `smile_is_continuous_across_the_money` | same | local linear trend at $F(1\pm\varepsilon)$ |
| `rejects_degenerate_correlation`, proptest `negative_rho_gives_downward_skew_and_positive_vols` | same | validation; skew sign, positivity on K/F ∈ [0.2, 5] |
| `z_over_x_is_finite_and_positive_over_wide_range`, `z_over_x_series_matches_closed_form_at_the_switch` | `quantlib_models/src/sabr_model.rs` | z ∈ [−50, 50]; series vs closed form at 1e-6 |
| `shifted_sabr_delegates_to_base_with_shifted_strikes`, `general_sabr_validation_rejects_beta_bounds`, `normal_sabr_implied_vol_is_positive`, `rejects_nan_parameters` | same | delegation; validation |
| `put_call_parity`, `calls_are_decreasing_and_convex_in_strike` | `quantlib_engines/tests/european_analytical_sabr.rs` | General, Lognormal, Normal on 50 strikes; parity to 1e-12 |
| `analytical_sabr_prices_european_option` | `quantlib_engines/src/analytical_engine.rs` | smoke test |
| `calibrates_sabr_surface` | `quantlib_utils/src/calibration.rs` | recovery of $(\alpha, \rho, \nu)$ on a synthetic smile, β fixed at 0.6 by its bounds, 3e-2 |
| `test_calibrate_sabr_surface` | `quantlib_py/tests/test_calibration_bindings.py` | Python binding |

`shifted_sabr_prices_black_on_shifted_forward_and_strike`, `shifted_sabr_prices_zero_forward_and_strike_with_put_call_parity` (same file) check Shifted pricing against Black-76 on $(F+s, K+s)$ to 1e-12, and parity at $F=0$ or $K=0$.

## Known limitations

- Hagan 2002 asymptotics: negative density at low strikes or large $\nu^2T$, and reduced accuracy for β < 1 away from ATM. The Obłój (2008) correction, arbitrage-free PDE SABR and density diagnostics are pending (#20).
- `EuropeanOption` rejects negative $F$ and $K$, so Shifted SABR through the engine only reaches $F, K \ge 0$.
- `implied_volatility` requires $F, K > 0$ even for Normal SABR, so negative rates need the Shifted wrapper.
- No implied-vol inversion: market normal vols cannot be checked against prices (#19).
- Flat `r` (#17). No Greeks (#18).
- `f0` is stored and validated but unused.
