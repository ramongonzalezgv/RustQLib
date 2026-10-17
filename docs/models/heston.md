# Heston

Code: `quantlib_models/src/heston_model.rs` (`HestonModel`). MC discretisation: `quantlib_engines/src/monte_carlo_engine/heston.rs`.

## Dynamics

Heston (1993), risk-neutral:

```math
\begin{aligned}
\frac{dS_t}{S_t} &= (r - q)\,dt + \sqrt{v_t}\,dW^S_t \\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^v_t,\qquad d\langle W^S, W^v\rangle_t = \rho\,dt
\end{aligned}
```

| Field | Symbol | Meaning | `validate()` |
|---|---|---|---|
| `kappa` | $\kappa$ | mean-reversion speed | finite, `>= 0` |
| `theta` | $\theta$ | long-run variance | finite, `>= 0` |
| `sigma` | $\sigma$ | vol of variance | finite, `>= 0` |
| `rho` | $\rho$ | spot/variance correlation | finite, in `[-1, 1]` |
| `v0` | $v_0$ | initial variance | finite, `>= 0` |
| `r` | $r$ | flat rate | finite |
| `q` | $q$ | flat dividend yield | finite |

The Feller condition $2\kappa\theta \ge \sigma^2$ is not enforced.

## Characteristic function

`characteristic_function(u, T)` returns $\varphi(u) = E[e^{iu\ln(S_T/S_0)}]$. It uses the formulation with $g = (b-d)/(b+d)$ and $e^{-dT}$. This is the second, continuous form in Albrecher, Mayer, Schoutens & Tistaert (2007), "The little Heston trap" (equation number not cited here):

```math
\begin{aligned}
b &= \kappa - \rho\sigma iu,\qquad d = \sqrt{b^2 + \sigma^2(iu + u^2)},\qquad g = \frac{b - d}{b + d} \\
C &= (r - q)\,iuT + \frac{\kappa\theta}{\sigma^2}\Big[(b - d)T - 2\ln\frac{1 - g e^{-dT}}{1 - g}\Big] \\
D &= \frac{b - d}{\sigma^2}\,\frac{1 - e^{-dT}}{1 - g e^{-dT}},\qquad \varphi(u) = e^{C + D v_0}
\end{aligned}
```

Implementation details:

- $g$ is never formed. The log argument is computed as $\big((b+d) - (b-d)e^{-dT}\big)/(2d)$ and $D$ as $(b-d)(1-e^{-dT})(b+d)/\big(\sigma^2((b+d)-(b-d)e^{-dT})\big)$, which saves two complex divisions (#26).
- `principal_sqrt` is an algebraic principal square root (branch cut on the negative real axis, $\operatorname{Re} d \ge 0$); it avoids `hypot` and `atan2`/`sin_cos`. `fast_ln` is the principal log $\tfrac12\ln|z|^2 + i\arg z$: it avoids `hypot` but still calls `atan2` (`arg`).
- If $T \le 0$, $\varphi = 1$.
- If $\sigma^2 \le 10^{-16}$, the code uses the deterministic-variance limit (wrong for $\kappa = 0$, #49):

  ```math
  \varphi(u) = \exp\!\big(iu(r-q)T - \tfrac12(u^2 + iu)\bar V\big),\quad
  \bar V = \theta T + (v_0 - \theta)\frac{1 - e^{-\kappa T}}{\max(\kappa, 10^{-16})}
  ```

- Cumulants for the COS range come from the default `FourierModel::cumulants`. These are numerical central differences of $\ln\operatorname{Re}\varphi(-it)$ at $t = 0$, with $h = 10^{-3}$ for $c_1, c_2$ and $h = 5\cdot10^{-2}$ for $c_4$. There are no closed-form Heston cumulants.

## Engines

- **FFT** (Lewis 2001) and **COS** (Fang & Oosterlee 2008) price Europeans through `FourierModel`; see [fourier.md](../engines/fourier.md). COS uses the numerical $c_1, c_2, c_4$ above. The $c_4$ term widens the interval for the fat left tail, which is why the Fang–Oosterlee case needs N ≈ 256 (#26).
- **Monte Carlo** ([monte-carlo.md](../engines/monte-carlo.md)) prices European, digital and Asian options with `n_steps` steps. Each step uses two independent normals, $z_s$ (spot) and $z_v$ (variance). The scheme is set by `MonteCarloConfig::heston_scheme` and defaults to QE:
  - **`QuadraticExponential`** (Andersen 2008), $\gamma_1 = \gamma_2 = \tfrac12$, $\psi_c = 1.5$:
    - It matches the exact conditional mean $m = \theta + (v - \theta)e^{-\kappa\Delta}$ and variance $s^2 = v\sigma^2 e^{-\kappa\Delta}\frac{1-e^{-\kappa\Delta}}{\kappa} + \tfrac12\theta\sigma^2\frac{(1-e^{-\kappa\Delta})^2}{\kappa}$.
    - For $\psi = s^2/m^2 \le 1.5$ it uses the quadratic branch $v' = a(b + z_v)^2$.
    - Otherwise it uses the exponential branch with point mass $p = (\psi-1)/(\psi+1)$ at zero, sampled from $U = \Phi(z_v)$.
    - Log-spot step: $\ln S \mathrel{+}= (r-q)\Delta + K_0^* + K_1 v + K_2 v' + \sqrt{K_3 v + K_4 v'}\,z_s$.
    - $K_0^*$ is martingale-corrected. Where the variance MGF does not exist, the code falls back to Andersen's $K_0$, which breaks the martingale for that step.
    - If $\sigma \le 10^{-8}$ or $m \le 0$ or $s^2 \le 0$, the variance is deterministic and its integral uses the trapezoidal rule.
  - **`FullTruncation`** (Lord, Koekkoek & van Dijk 2010): Euler where only $v^+ = \max(v, 0)$ enters drift and diffusion, and $v$ itself is never floored.
  - Why: the old Euler scheme floored $v$ at every step ("absorption"). It had the largest bias of the Euler fixes and was badly biased when Feller fails, which is typical for calibrated equity parameters (#6). In the #29 bias table (Feller-violating case, reference 6.8400), the bias was:

    | Steps | Absorption | Full truncation | QE |
    |---|---|---|---|
    | 52 | +1.431 | +0.115 | −0.005 |

    QE at 32 steps beats any Euler variant at 252 steps.
  - **Control variate**: European and digital options use $S_T$ with $E = S_0 e^{(r-q)T}$, and Asians use the arithmetic average. The control applies under both schemes: each discretised spot is an exact martingale, except QE steps that fall back to $K_0$ ([0018](../decisions/0018-control-variate-both-heston-schemes.md)).
- **Analytical / binomial**: not supported.

## Calibration

`calibrate_heston_model` (`quantlib_utils/src/calibration.rs`; see [calibration.md](../engines/calibration.md)):

- Fits all 7 fields, including `r` and `q`. To fix a parameter, set its lower bound equal to its upper bound.
- Default bounds: $\kappa \in [10^{-4}, 20]$, $\theta \in [10^{-5}, 2]$, $\sigma \in [10^{-4}, 5]$, $\rho \in [-0.999, 0.999]$, $v_0 \in [10^{-5}, 2]$, $r, q \in [-0.2, 0.2]$.
- Price quotes only.
- Model prices come from `FFTEngine::price_strip` with `HestonFftConfig` (default n = 1024, b = 180, cubic).
- Quotes are grouped once by bit-exact `(maturity, underlying)`. Each group is one FFT strip, priced at its exact float maturity (#4/#27, 20× faster objective).
- Objective: weighted least squares, relative least squares or Huber. Optimiser: DE→Nelder–Mead by default.
- A parameter set that fails `validate()` or gives a non-finite objective scores `1e300`.

## Design choices

- Little-trap CF form with the principal branch: no branch-cut discontinuity. It was rewritten for speed with the same formulation and branch (#26).
- Generic `FourierModel` trait: one FFT/COS code path for all CF models (#7).
- QE is the default MC scheme. Full truncation is kept as the simple, low-bias Euler baseline, and absorption was removed (#6/#29).
- Exact-maturity strips in calibration (#4).

## Validation

| Test | File | Reference |
|---|---|---|
| `characteristic_function_is_one_at_origin`, `characteristic_function_respects_first_moment` | `quantlib_models/src/heston_model.rs` | $\varphi(0) = 1$; $\varphi(-i) = e^{(r-q)T}$ |
| `principal_sqrt_matches_num_complex` | same | `num_complex::Complex64::sqrt` |
| `validates_rho_range`, `rejects_nan_inputs` | same | validation |
| `cos_heston_matches_fang_oosterlee_reference` | `quantlib_engines/tests/cos_engine.rs` | Fang & Oosterlee (2008) Heston case, 5.785155450, < 1e-7 |
| `cos_matches_fft_for_heston_bates_and_kou` | same | 2^15-point FFT, < 1e-6 |
| `fft_heston_matches_reference_lewis_integral` | `quantlib_engines/tests/fft_heston_engine.rs` | direct Simpson Lewis integral (same CF, so it tests the quadrature only) |
| `fft_heston_zero_vol_of_vol_matches_black_scholes_with_integrated_variance` | same | BS with time-averaged variance, σ ∈ {0, 1e-4}, κ = 1.5, 5e-3 abs |
| `fft_heston_put_call_parity_holds`, `fft_handles_near_degenerate_heston_parameters` | same | parity; σ = 0, κ = 1e-8, $v_0 = \theta$ vs BS (cannot detect #49, since $v_0 = \theta$) |
| `heston_mc_within_three_stderr_of_fourier` | `quantlib_engines/tests/monte_carlo_engine.rs` | FFT, 3·stderr |
| `heston_qe_is_unbiased_with_coarse_steps_when_feller_fails` | same | COS (N=1024, L=16), 32 steps, 3·stderr |
| `heston_full_truncation_bias_shrinks_with_steps_when_feller_fails` | same | COS; bias monotone over {12, 52, 252} |
| `heston_terminal_spot_is_a_martingale` | same | $S_0e^{-qT} - Ke^{-rT}$, QE and FT |
| `european_control_variate_reduces_heston_error` | same | FFT, QE and FT; SE cut |
| `mc_heston_european_deterministic_zero_variance` | `quantlib_engines/src/monte_carlo_engine/mod.rs` | deterministic forward, 1e-10 |
| `test_qe_matches_fourier_when_feller_fails` | `quantlib_py/tests/test_heston_mc_scheme.py` | COS, delta 0.08 |
| `calibrates_heston_surface_from_fft_prices`, `heston_calibration_prices_quotes_at_their_exact_maturity`, `heston_calibration_keeps_quote_order_across_maturities_and_spots` | `quantlib_utils/src/calibration.rs` | parameter recovery; direct price at T = 1/12 to 1e-10 rel |

## Known limitations

- Flat `r`, `q` (#17).
- Price-space calibration only (#19).
- Derivative-free optimisers only; LM with analytic gradients (Cui et al. 2017) is pending (#9).
- No Greeks (#18).
- Bug #49: the deterministic-variance limit uses `kappa.max(1e-16)`. With $\kappa = 0$ and $\sigma = 0$ the numerator $1 - e^{-\kappa T}$ is 0, so it returns $\bar V = \theta T$ instead of $v_0 T$ (wrong whenever $v_0 \ne \theta$). Bates inherits it.
- MC: the QE martingale correction is skipped on steps where the MGF does not exist.
