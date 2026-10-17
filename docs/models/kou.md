# Kou

Code: `quantlib_models/src/kou_model.rs` (`KouModel`); closed-form cumulants in `quantlib_models/src/fourier_model.rs`.

## Dynamics

Kou (2002), double-exponential jump diffusion, risk-neutral:

```math
\frac{dS_t}{S_{t^-}} = (r - q - \lambda\zeta)\,dt + \sigma\,dW_t + (e^{Y} - 1)\,dN_t,\qquad
f_Y(y) = p\,\eta_1 e^{-\eta_1 y}\mathbf 1_{y\ge0} + (1-p)\,\eta_2 e^{\eta_2 y}\mathbf 1_{y<0}
```

```math
\zeta = E[e^Y] - 1 = \frac{p\,\eta_1}{\eta_1 - 1} + \frac{(1-p)\,\eta_2}{\eta_2 + 1} - 1
```

| Field | Symbol | Meaning | `validate()` |
|---|---|---|---|
| `sigma` | $\sigma$ | diffusion volatility | finite, `>= 0` |
| `r` | $r$ | flat rate | finite |
| `q` | $q$ | flat dividend yield | finite |
| `lambda_jump` | $\lambda$ | jump intensity | finite, `>= 0` |
| `p_up` | $p$ | probability of an up jump | finite, in `[0, 1]` |
| `eta1` | $\eta_1$ | rate of up jumps (mean size $1/\eta_1$) | finite, `> 1` (finite $E[e^Y]$) |
| `eta2` | $\eta_2$ | rate of down jumps (mean size $1/\eta_2$) | finite, `> 0` |

## Characteristic function

```math
\varphi(u) = \exp\!\Big(iu\big(r - q - \lambda\zeta - \tfrac12\sigma^2\big)T - \tfrac12\sigma^2u^2T
+ \lambda T\Big(\frac{p\,\eta_1}{\eta_1 - iu} + \frac{(1-p)\,\eta_2}{\eta_2 + iu} - 1\Big)\Big)
```

If $T \le 0$, $\varphi = 1$. The jump term is analytic for $-\eta_1 < \operatorname{Im} u < \eta_2$; $\eta_1 > 1$ puts $u = -i$ inside this strip, so $E[e^Y]$ and $\zeta$ are finite.

Cumulants are closed form, overriding the numerical default. For a compound Poisson process $\kappa_n = \lambda T\,E[Y^n]$, and $E[Y^n] = n!\,(p/\eta_1^n + (-1)^n(1-p)/\eta_2^n)$:

```math
c_1 = (r - q - \lambda\zeta - \tfrac12\sigma^2)T + \lambda T\,E[Y],\quad
c_2 = \sigma^2 T + \lambda T\,E[Y^2],\quad
c_4 = \lambda T\,E[Y^4]
```

## Engines

Europeans through `FFTEngine` and `CosEngine` only; see [fourier.md](../engines/fourier.md). COS uses the closed-form cumulants above for its truncation range. There is no Monte Carlo, analytical (Kou 2002 closed form) or binomial support.

## Calibration

None.

## Design choices

- Closed-form cumulants because they are exact and cheap. The numerical default is kept for models without them.
- Engine access goes through the generic `FourierModel` trait (#7).

## Validation

| Test | File | Reference |
|---|---|---|
| `characteristic_function_is_one_at_origin`, `characteristic_function_respects_first_moment` | `quantlib_models/src/kou_model.rs` | $\varphi(0)=1$; $\varphi(-i) = e^{(r-q)T}$ (checks $\zeta$) |
| `reduces_to_black_scholes_when_jump_intensity_is_zero` | same | BS CF, 1e-12 |
| `validates_probability_bounds`, `validates_eta_constraints`, `rejects_nan_inputs` | same | validation |
| `kou_closed_form_cumulants_match_numerical` | `quantlib_models/src/fourier_model.rs` | numerical cumulants of the CF |
| `fft_kou_matches_reference_lewis_integral` | `quantlib_engines/tests/fft_kou_engine.rs` | direct Simpson Lewis integral (same CF, so it tests the quadrature only) |
| `fft_kou_put_call_parity_holds`, `fft_kou_reduces_to_black_scholes_when_jump_intensity_is_zero` | same | parity; BS |
| `cos_matches_fft_for_heston_bates_and_kou` | `quantlib_engines/tests/cos_engine.rs` | 2^15-point FFT, < 1e-6 |

No test compares prices against Kou's (2002) closed-form price or a published value.

## Known limitations

- Flat `r`, `q` (#17).
- No calibration, MC or closed-form engine.
- No Greeks (#18).
- With $\sigma = 0$ and $\lambda = 0$, $c_2 = 0$ and COS returns a "degenerate truncation range" error.
