# Fourier engines (FFT, COS)

European vanilla prices from the characteristic function of the log-return. Both engines work with any `FourierModel`: Black–Scholes, Black-76, Heston, Bates and Kou (CFs in [black-scholes.md](../models/black-scholes.md), [heston.md](../models/heston.md), [bates.md](../models/bates.md), [kou.md](../models/kou.md)). Each one prices a whole strike strip at one maturity for about the cost of one strike. Code: `quantlib_engines/src/fourier.rs`, `fft_engine.rs`, `cos_engine.rs`, `quantlib_models/src/fourier_model.rs`.

## `FourierModel` trait

| Item | Meaning |
|---|---|
| `log_return_cf(u, T)` | $\varphi_T(u) = \mathbb E[e^{iuX_T}]$, $X_T = \ln(S_T/S_0)$ under $\mathbb Q$, including the $(r-q)T$ drift. |
| `rate()`, `dividend_yield()` | $r$, $q$. |
| `FORWARD_QUOTED` | `true` only for Black-76. It describes $\ln(F_T/F_0)$ (zero drift) with $q = r$, so engines treat the forward as the spot and need no separate code path. |
| `cumulants(T)` | $c_1, c_2, c_4$ of $X_T$. Closed form for BS, Black-76 and Kou ($\lambda T\,\mathbb E[J^n]$ for the jump part). Heston and Bates use the default `numerical_cumulants`: central differences of $K(t) = \ln\varphi(-it)$ at $t = 0$, with $h = 10^{-3}$ for $c_1, c_2$ and $h = 0.05$ (five-point stencil) for $c_4$. |

## Shared plumbing (`fourier.rs`)

`price_european` validates the product and model, then:
- $T \le 0$: undiscounted intrinsic value on the spot (forward for Black-76).
- $K \le 0$: call $= Se^{-qT}$, put $= 0$.
- otherwise it prices a one-strike strip.

`validate_strip` requires a finite, positive spot, $T$ and strikes, and equal strike and type lengths. A non-finite price is an `Engine` error.

## FFT engine: Lewis formula on an FFT grid

The code uses the Lewis (2001) single-integral formula, not the damped Carr–Madan call transform, so there is **no damping parameter α**. With $k = \ln(S/K)$:

$$C(K) = Se^{-qT} - \frac{\sqrt{SK}\,e^{-rT}}{\pi}\, I(k),\qquad I(k) = \int_0^\infty \mathrm{Re}\!\left[e^{iuk}\,\varphi_T\!\left(u - \tfrac i2\right)\right]\frac{du}{u^2 + \frac14}$$

Puts follow from parity: $P = C + Ke^{-rT} - Se^{-qT}$.

**Discretisation.** $N$ nodes $u_j = j\,\Delta u$ with $\Delta u = B/N$, so the integral is truncated at $u < B$. The log-strike grid is $k_m = -\tfrac{N}{2}\Delta k + m\,\Delta k$ with $\Delta k = 2\pi/B$, which gives $\Delta u\,\Delta k = 2\pi/N$. Then $u_j k_m = -\pi j + 2\pi jm/N$, and the whole grid is one unnormalised inverse FFT:

$$I(k_m) \approx \mathrm{Re}\sum_{j=0}^{N-1} e^{2\pi i jm/N}\;(-1)^j\,\frac{w_j\,\Delta u}{3\,(u_j^2 + \frac14)}\;\varphi_T\!\left(u_j - \tfrac i2\right),\qquad w_0 = 1,\; w_{\text{odd}} = 4,\; w_{\text{even}>0} = 2$$

These are the Carr & Madan (1999) Simpson weights $3 - (-1)^j - \delta_{j0}$ (0-indexed); there is no end-point correction at $j = N-1$. $I$ is then interpolated at each strike's $k$:
- `Cubic` (default): local 8-point Lagrange on the uniform grid, centred on the bracketing interval and clamped at the edges. O(1) and allocation-free. Despite the name it is degree 7.
- `Linear`.

A strike whose $k$ falls outside $[-\pi N/B,\; \pi N/B - 2\pi/B]$ is an `Engine` error. The defaults give $|k| \lesssim 64$.

**Cost.** One sweep of $N$ CF evaluations and one FFT per maturity, plus O(1) per strike. The FFT plan, nodes $u_j - i/2$ and weight factors are built once in `FFTEngine::new` and shared by clones through an `Arc`.

## COS engine (Fang & Oosterlee 2008)

Let $x = \ln(S/K)$ and $y = \ln(S_T/K) = x + X_T$. The truncation range is F–O eq. (49), shifted by $x$:

$$[a, b] = \left[x + c_1 - L\sqrt{c_2 + \sqrt{c_4}},\; x + c_1 + L\sqrt{c_2 + \sqrt{c_4}}\right]$$

with $c_4$ clamped at 0 and $\omega_k = k\pi/(b-a)$. The engine prices the **put**, whose payoff $K(1 - e^y)^+$ is bounded:

$$P \approx Ke^{-rT}\sum_{k=0}^{N-1}{}' \,\mathrm{Re}\!\left[\varphi_T(\omega_k)\,e^{i\omega_k(x-a)}\right] U_k,\qquad U_k = \frac{2}{b-a}\big(\psi_k(a,0) - \chi_k(a,0)\big)$$

$$\chi_k(a,0) = \frac{\cos(\omega_k a) - e^{a} - \omega_k\sin(\omega_k a)}{1 + \omega_k^2},\qquad \psi_k(a,0) = \begin{cases} -a & k = 0\\ -\sin(\omega_k a)/\omega_k & k \ge 1\end{cases}$$

These are F–O eqs. (22), (23) and (29). $\Sigma'$ halves the $k = 0$ term. The call is $C = P + Se^{-qT} - Ke^{-rT}$. Both prices are floored at 0 against round-off.

- The width $b - a$ and the phase argument $x - a = L\sqrt{c_2+\sqrt{c_4}} - c_1$ do not depend on the strike. A strip therefore evaluates the CF $N$ times once and only rebuilds $U_k$ per strike, at O(N) per strike. The trig terms come from rotating $e^{-i\pi a/(b-a)}$ rather than per-term calls.
- If $a \ge 0$ the put is 0. If $b \le 0$ the put is the exact short forward $Ke^{-rT} - Se^{-qT}$, which keeps truncated tail mass (for example rare jumps at short maturities) out of the price.

## Implementation map

| Item | Role |
|---|---|
| `FourierStrip` (crate-private) | `strip(model, spot, T, strikes, types)`: unit prices for one maturity. Implemented by both engines. |
| `FFTEngine::new(n, b, interp)`, `Default` | Builds `FftGrid` (plan, nodes, factors, $k_0$, $\Delta k$). `n()`, `b()`, `interp()` read the settings. |
| `FFTEngine::price_strip` / `CosEngine::price_strip` | Public strip pricing with validation. For Black-76, `spot` is the forward. |
| `FFTEngine::integral_grid`, `interpolate` | CF sweep plus FFT; O(1) interpolation. |
| `cos_engine::put_series` | $\Sigma' A_k U_k$ for the put payoff. |
| `impl<M: FourierModel> PricingEngine<EuropeanOption, M>` | One generic impl per engine, through `price_european`. |

## Configuration

| Engine | Field | Default | Meaning |
|---|---|---|---|
| `FFTEngine` | `n` | `DEFAULT_N = 4096` (≥ 8) | Quadrature nodes = FFT size. |
| | `b` | `DEFAULT_B = 200.0` (> 0) | Integration cutoff in $u$. $\Delta u = b/n \approx 0.049$, $\Delta k = 2\pi/b \approx 0.031$. |
| | `interp` | `Cubic` | `Cubic` (8-point Lagrange) or `Linear`; parsed from `"cubic"` / `"linear"`. |
| `CosEngine` | `n` | `256` (≥ 2) | Cosine terms. |
| | `l` | `12.0` (> 0) | Truncation multiplier $L$. |

**Why these defaults.**
- COS $N = 256$, $L = 12$: the Fang–Oosterlee Heston case has a fat left tail. It needs the $c_4$-widened interval, and that interval needs $N \approx 256$ to reach 1e-7. No $L$ reaches 1e-7 at $N = 160$ (PR #26).
- FFT 4096 / 200 were kept unchanged in #26. For BS and Black-76 they are accurate to ≤ 1e-6 for $T \ge 0.5$, but at short maturities the price is interpolation-limited by $\Delta k$ (see Limitations).

Python: `CosEngine(n=256, l=12.0)`.

## Accuracy and performance

Measured in PR #26 (criterion, `benches/fft_bench.rs`, Apple Silicon):

| Benchmark | Time |
|---|---|
| Heston single price, FFT 4096 (cubic) | 149.8 µs |
| Heston single price, COS 256 | 10.9 µs (13.7× faster) |
| Heston strip of 50 strikes, FFT 4096 | 151.6 µs |
| Heston strip of 50 strikes, COS 256 | 32.1 µs (4.7× faster) |

An FFT price is about 4096 CF evaluations. The 1.89× gain over the previous FFT came from the plan/array caching and the Heston CF rewrite.

Accuracy:
- FFT-BS and FFT-Black-76 vs analytic: ≤ 1e-6 on K ∈ [50, 200], T ∈ {0.1, 0.5, 1, 2, 5}. T = 0.1 needs n = 8192, b = 400; the default grid gives 2.7e-5 there.
- 8-point Lagrange vs the old global spline, Heston T = 1: 5.6e-6 vs 3.4e-5.
- COS-BS, N = 128: < 1e-10 on K ∈ [50, 200], T ∈ [0.05, 5].
- COS Heston (F–O case), default: 4.5e-8 from 5.785155450. The reference itself is about 1.6e-8 above the converged value.
- COS vs a fine FFT ($2^{15}$, b = 1200) for Heston, Bates and Kou, 10 strikes × 10 maturities: < 1e-6.

## Design choices

- **Lewis instead of damped Carr–Madan.** No α to tune per model or maturity. The integrand is regular at $u = 0$, and one grid gives every strike.
- **COS prices puts, calls by parity.** The put payoff is bounded, so truncating the density costs little. A call's payoff grows like $e^{b}$.
- **One generic impl per engine over `FourierModel`.** It replaced five model-specific FFT impls; adding a model only needs its CF.
- **Precomputed, shared FFT grid.** A price makes at most 3 allocations whatever the strike count, and no planner runs in `calculate_price`.

## Validation

| Test | File | Checks |
|---|---|---|
| `fft_black_scholes_matches_analytic_on_strike_maturity_grid`, `fft_black_matches_analytic_on_strike_maturity_grid` | `tests/fft_black_and_bs_engine.rs` | ≤ 1e-6 vs closed form (default grid; n = 8192, b = 400 at T = 0.1). |
| `fft_single_option_price_matches_strip`, `fft_black_scholes_rejects_forward_quoted_option`, `fft_black_rejects_spot_quoted_option` | same | Single price = strip; quote-type errors. |
| `fft_heston_matches_reference_lewis_integral`, `fft_heston_put_call_parity_holds`, `fft_heston_zero_vol_of_vol_matches_black_scholes_with_integrated_variance`, `fft_handles_near_degenerate_heston_parameters`, `fft_returns_intrinsic_when_ttm_is_zero`, `fft_returns_error_for_log_moneyness_outside_domain` | `tests/fft_heston_engine.rs` | Direct quadrature reference, parity, limits, domain error. |
| `fft_bates_*`, `fft_kou_*` | `tests/fft_bates_engine.rs`, `tests/fft_kou_engine.rs` | Reference integral, parity, reduction to Heston / BS at zero jump intensity. |
| `cos_heston_matches_fang_oosterlee_reference` | `tests/cos_engine.rs` | < 1e-7 vs 5.785155450. |
| `cos_black_scholes_matches_analytic_with_128_terms`, `cos_black_prices_forward_quoted_options` | same | < 1e-10. |
| `cos_matches_fft_for_heston_bates_and_kou` | same | < 1e-6 vs FFT $2^{15}$/1200, 10 maturities in [0.1, 5]. |
| `cos_short_maturity_and_deep_otm_are_finite_non_negative_and_satisfy_parity` | same | Heston, Bates, Kou; T ∈ {1/365, 1}, K/S ∈ {0.5, 0.8, 1, 1.25, 2}: finite, non-negative, parity < 1e-10. |
| `cos_single_option_matches_strip_and_scales_with_quantity`, `cos_rejects_invalid_settings_and_inputs` | same | `qty`, validation. |
| `fft_price_allocates_a_constant_handful_of_buffers`, `cos_price_allocates_a_constant_handful_of_buffers` | `tests/fourier_allocations.rs` | ≤ 3 (FFT) and ≤ 2 (COS) allocations per price; FFT strip of 50 allocates as much as one price. |
| `numerical_cumulants_match_black_scholes_closed_form`, `kou_closed_form_cumulants_match_numerical` | `quantlib_models/src/fourier_model.rs` | Cumulants. |
| `test_cos_engine.py` | `quantlib_py/tests` | Python defaults, FFT agreement, batch, Black-76. |

## Limitations and open issues

- European vanillas only.
- The default FFT grid is interpolation-limited at short maturities ($\Delta k \approx 0.031$ against $\sigma\sqrt T \approx 0.06$ at T = 0.1). Raise `n` and `b` together, keeping $\Delta u$ fixed.
- $N$, $L$, $n$, $b$ are fixed by the user. There is no adaptive choice by maturity or model.
- Heston and Bates cumulants come from finite differences. $c_4$ is noisy, so it is clamped at 0.
- Heston calibration prices with the FFT (`HestonFftConfig`, see [calibration.md](calibration.md)), not COS.
- No analytic CF gradients for calibration (#9). No Greeks (#18).
