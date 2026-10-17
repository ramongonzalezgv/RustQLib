# Bates

Code: `quantlib_models/src/bates_model.rs` (`BatesModel`).

## Dynamics

Bates (1996): Heston plus compound-Poisson log-normal jumps in the spot, risk-neutral:

```math
\begin{aligned}
\frac{dS_t}{S_{t^-}} &= (r - q - \lambda\bar k)\,dt + \sqrt{v_t}\,dW^S_t + (e^{J} - 1)\,dN_t,\qquad J \sim \mathcal N(\mu_J, \sigma_J^2) \\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^v_t,\qquad d\langle W^S, W^v\rangle_t = \rho\,dt,\qquad \bar k = e^{\mu_J + \sigma_J^2/2} - 1
\end{aligned}
```

$N$ is a Poisson process with intensity $\lambda$, independent of $W$.

| Field | Symbol | Meaning | `validate()` |
|---|---|---|---|
| `kappa`, `theta`, `sigma`, `rho`, `v0` | $\kappa, \theta, \sigma, \rho, v_0$ | Heston part, see [heston.md](heston.md) | as Heston: finite, $\kappa, \theta, \sigma, v_0 \ge 0$, $\rho \in [-1, 1]$ |
| `r`, `q` | $r, q$ | flat rate, dividend yield | finite |
| `lambda_jump` | $\lambda$ | jump intensity (per year) | finite, `>= 0` |
| `mu_j` | $\mu_J$ | mean of the log jump size | finite |
| `sigma_j` | $\sigma_J$ | std. dev. of the log jump size | finite, `>= 0` |

## Characteristic function

`characteristic_function` builds a `HestonModel` from the Heston fields and multiplies its CF (little-trap form, see [heston.md](heston.md)) by the compensated jump part:

```math
\varphi_{Bates}(u) = \varphi_{Heston}(u)\,\exp\!\Big(\lambda T\big(e^{iu\mu_J - \frac12\sigma_J^2u^2} - 1 - iu\bar k\big)\Big)
```

If $T \le 0$, $\varphi = 1$. Cumulants for the COS range are numerical (the default `FourierModel::cumulants`).

## Engines

Europeans through `FFTEngine` and `CosEngine` only (`FourierModel` impl); see [fourier.md](../engines/fourier.md). There is no Monte Carlo, analytical or binomial support.

## Calibration

None. `quantlib_utils/src/calibration.rs` has no Bates calibrator (see [calibration.md](../engines/calibration.md)).

## Design choices

- Reuses the Heston CF instead of duplicating it. With $\lambda = 0$ the result is the Heston CF to 1e-12.
- Engine access goes through the generic `FourierModel` trait (#7).

## Validation

| Test | File | Reference |
|---|---|---|
| `characteristic_function_is_one_at_origin`, `characteristic_function_respects_first_moment` | `quantlib_models/src/bates_model.rs` | $\varphi(0)=1$; $\varphi(-i) = e^{(r-q)T}$ (checks the compensator) |
| `reduces_to_heston_when_jump_intensity_is_zero` | same | Heston CF, 1e-12 |
| `validates_rho_range`, `rejects_negative_jump_intensity`, `rejects_nan_inputs` | same | validation |
| `fft_bates_matches_reference_lewis_integral` | `quantlib_engines/tests/fft_bates_engine.rs` | direct Simpson Lewis integral (same CF, so it tests the quadrature only) |
| `fft_bates_put_call_parity_holds`, `fft_bates_reduces_to_heston_when_jump_intensity_is_zero` | same | parity; Heston FFT price |
| `cos_matches_fft_for_heston_bates_and_kou` | `quantlib_engines/tests/cos_engine.rs` | 2^15-point FFT, < 1e-6 |

No test compares prices against an independent published Bates value.

## Known limitations

- Flat `r`, `q` (#17).
- No calibration, and no MC (jump simulation).
- No Greeks (#18).
