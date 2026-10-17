# Monte Carlo engine

One engine, `MonteCarloEngine { config: MonteCarloConfig }`. It prices European, digital and Asian options under Black–Scholes and Heston ([black-scholes.md](../models/black-scholes.md), [heston.md](../models/heston.md)) and returns the price with its standard error. Code: `quantlib_engines/src/monte_carlo_engine/{mod.rs, core.rs, paths.rs, heston.rs}`.

## Driver

`paths::simulate` is the generic driver over two traits:

- `PathModel`: a state, one time step and the number of normals per step (BS 1, Heston 2).
- `PathFunctional`: reduces a path to a `Sample { y, x }` while it is simulated (`y` is the payoff, `x` the control), so no path is stored. The two functionals are `Terminal` (payoff of $S_T$, $x = S_T$) and `Average` (running sum and, only when needed, log-sum over $S_0, S_\Delta, \dots, S_T$).

**Chunking and RNG.** The `n_samples` samples are cut into chunks of `chunk_size` samples. Chunk $i$ draws from its own stream, `Xoshiro256PlusPlus::seed_from_u64(splitmix64(seed ^ splitmix64(i)))`.

Normals come from `rand_distr::StandardNormal` (ziggurat). Chunks run under Rayon when `parallel = true`. Each chunk returns its moments, and they are merged **in chunk order**. Seeded results are therefore bit-identical serially, in parallel and for any thread count. They do depend on `chunk_size`.

**Statistics.** Within a chunk, `ChunkSums` accumulates division-free sums of $(y, x)$ shifted by the chunk's first sample, which keeps the sums of squares well conditioned. Chunks are merged with the pairwise update of Chan, Golub & LeVeque. With $n = n_a + n_b$ and $\delta = \bar y_b - \bar y_a$:

$$\bar y = \bar y_a + \delta\,\frac{n_b}{n},\qquad M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{n_a n_b}{n}$$

The same update applies to $x$ and to the co-moment $C_{xy}$. The estimate is

$$\hat V = e^{-rT}\,\bar y,\qquad \text{SE} = e^{-rT}\sqrt{\frac{M_2}{n(n-1)}}$$

`std_error` is NaN when $n < 2$.

## Path models

**Black–Scholes, exact step** (`BlackScholesPath`): $S \leftarrow S\exp\big((r - q - \tfrac12\sigma^2)\Delta + \sigma\sqrt\Delta\,z\big)$. European and digital options take **one** step to $T$ whatever `n_steps` is. Asians take `n_steps` steps.

**Heston** (`HestonPath` → `HestonStepper`). Each step draws two independent normals: $z_s$ for the stock and $z_v$ for the variance. European, digital and Asian options all take `n_steps` steps.

### Quadratic-exponential (default; Andersen 2008)

The exact conditional moments of $v_{n+1}$, with $e = e^{-\kappa\Delta}$:

$$m = \theta + (v - \theta)e,\qquad s^2 = \frac{v\sigma^2 e(1-e)}{\kappa} + \frac{\theta\sigma^2(1-e)^2}{2\kappa},\qquad \psi = s^2/m^2$$

$(1-e)/\kappa$ is replaced by $\Delta$ when $\kappa\Delta < 10^{-12}$.

- $\psi \le \psi_c = 1.5$ (quadratic branch): $b^2 = 2/\psi - 1 + \sqrt{2/\psi}\sqrt{2/\psi - 1}$, $a = m/(1+b^2)$, and $v_{n+1} = a(b + z_v)^2$.
- $\psi > 1.5$ (exponential branch): $p = \frac{\psi-1}{\psi+1}$, $\beta = (1-p)/m$, $U = \Phi(z_v)$. Then $v_{n+1} = 0$ if $U \le p$, else $\beta^{-1}\ln\frac{1-p}{1-U}$.

The log-spot uses $\gamma_1 = \gamma_2 = \tfrac12$:

$$\ln S_{n+1} = \ln S_n + (r-q)\Delta + K_0^* + K_1 v_n + K_2 v_{n+1} + \sqrt{K_3 v_n + K_4 v_{n+1}}\;z_s$$

$$K_0 = -\frac{\rho\kappa\theta\Delta}{\sigma},\quad K_{1,2} = \gamma_{1,2}\Delta\Big(\frac{\kappa\rho}{\sigma} - \frac12\Big) \mp \frac{\rho}{\sigma},\quad K_{3,4} = \gamma_{1,2}\Delta(1-\rho^2)$$

**Martingale correction.** $K_0^* = -\ln \mathbb E[e^{A v_{n+1}} \mid v_n] - (K_1 + \tfrac12K_3)v_n$ with $A = K_2 + \tfrac12K_4$. This gives $\mathbb E[S_{n+1} \mid S_n] = S_n e^{(r-q)\Delta}$. The MGF terms are:
- quadratic branch: $\frac{Ab^2a}{1-2Aa} - \tfrac12\ln(1-2Aa)$, which needs $A < 1/(2a)$;
- exponential branch: $\ln\big(p + \frac{\beta(1-p)}{\beta - A}\big)$, which needs $A < \beta$.

Where the MGF does not exist, that step keeps $K_0$ and is not exactly a martingale.

**Deterministic fallback.** If $\sigma \le 10^{-8}$, $m \le 0$ or $s^2 \le 0$, the step sets $v_{n+1} = \max(m, 0)$ and integrates the variance with the trapezoidal rule.

### Full truncation (Lord, Koekkoek & van Dijk 2010)

With $v^+ = \max(v, 0)$:

$$S \leftarrow S\exp\big((r-q)\Delta - \tfrac12 v^+\Delta + \sqrt{v^+}\sqrt\Delta\,z_s\big),\qquad v \leftarrow v + \kappa(\theta - v^+)\Delta + \sigma\sqrt{v^+}\sqrt\Delta\big(\rho z_s + \sqrt{1-\rho^2}\,z_v\big)$$

The variance itself is never floored.

## Variance reduction

**Antithetic** (`antithetic = true`). Each normal vector $z$ drives two paths, $z$ and $-z$. Under QE, $-z_v$ maps to $1 - U$. The pair average $\tfrac12(y^+ + y^-)$ is **one sample**, so the standard error is computed over pairs and stays correct. `n_samples = ceil(n_paths / 2)`, and `McEstimate::n_paths` reports `2 * n_samples` simulated paths. At equal path count the SE ratio to plain MC is $\sqrt{1 + \rho}$ with $\rho = \mathrm{corr}(f(Z), f(-Z))$. For the ATM BS call of the test, $\rho = -0.473$ (PR #30), so the expected cut is 27.4% (ratio 0.726).

**Control variate** (`control_variate = true`). $\beta$ is estimated in-sample from the same run:

$$\hat V = e^{-rT}\big(\bar y - \beta(\bar x - \mathbb E[X])\big),\qquad \beta = \frac{C_{xy}}{M_{2,x}},\qquad M_2^{\text{res}} = M_{2,y} - \beta\,C_{xy}$$

If $M_{2,x} = 0$ the control is skipped.

| Product | Model | Control $X$ (undiscounted) | $\mathbb E[X]$ |
|---|---|---|---|
| European, digital | BS, Heston QE | $S_T$ | $S_0e^{(r-q)T}$ |
| Arithmetic Asian | BS | Same payoff on the geometric average (same fixings) | `qty` · `geometric_asian_black_scholes(…, Some(n_steps))` · $e^{rT}$ |
| Arithmetic Asian | Heston QE | Arithmetic average of $S$ | $\frac{S_0}{n+1}\sum_{i=0}^{n} e^{(r-q)i\Delta}$ |
| Geometric Asian | BS, Heston QE | Arithmetic average of $S$ | same |
| any | Heston full truncation | **none**: the flag is ignored (bug, #50) | – |

Every control mean above assumes $\mathbb E[S_t] = S_0e^{(r-q)t}$ for the discretised spot. Under Heston the control is applied only with QE (`McModel::martingale`); with full truncation `control_variate = true` is silently ignored. That gate is a bug (#50): the full-truncation spot step is conditionally log-normal with mean $S_ne^{(r-q)\Delta}$ given $v_n$, so it is an exact discrete martingale and the control means hold. Under QE they hold except on steps that fall back to $K_0$.

## `sample_paths`

`sample_paths_black_scholes(spot, model, ttm)` and `sample_paths_heston` return `SimulatedPaths { n_steps, values }`: `n_paths` rows of `n_steps + 1` points each, row-major, with $S_0$ first. They use the same chunk streams as pricing, without antithetic pairing, and are meant for plotting. Pricing never stores paths.

## Implementation map

| Item | File | Role |
|---|---|---|
| `MonteCarloEngine`, `with_*` builders | `mod.rs` | Engine and config setters. |
| `MonteCarloPricing<P, M>::price_with_stats` | `mod.rs` | Three generic impls (European, Digital, Asian) over `McModel`; `PricingEngine::calculate_price` returns `.price`. |
| `McModel` (private) | `mod.rs` | `rates`, `path`, `terminal_steps` (BS 1, Heston `n_steps`), `martingale(scheme)`, `geometric_asian` (BS only). |
| `estimate`, `price_terminal`, `expected_arithmetic_average` | `mod.rs` | Build the functional and control mean, run, discount. |
| `MonteCarloConfig`, `HestonScheme`, `McEstimate` | `core.rs` | Config, scheme enum (`"qe"`, `"full_truncation"`/`"ft"`), result. |
| `chunk_rng`, `splitmix64` | `core.rs` | Per-chunk streams. |
| `ChunkSums`, `Moments::{merge, estimate}` | `core.rs` | Shifted sums, Chan merge, control variate, SE. |
| `simulate`, `simulate_path`, `simulate_antithetic_pair`, `record_paths` | `paths.rs` | Driver. |
| `BlackScholesPath`, `HestonPath`, `Terminal`, `Average`, `AsianControl` | `paths.rs` | Models and functionals. |
| `HestonStepper::{qe_step, full_truncation_step}` | `heston.rs` | The one Heston step. |

## Configuration (`MonteCarloConfig`)

| Field | Default | Meaning |
|---|---|---|
| `n_paths` | 10 000 | Paths (≥ 1). Rounded up to even with antithetic. |
| `n_steps` | 252 | Time steps to $T$ (≥ 1). Ignored for BS European and digital (exact single step). Also the Asian fixing count: $n + 1$ fixings including $S_0$. |
| `seed` | `None` | `None` seeds from the system clock in nanoseconds, so results are not reproducible. |
| `parallel` | `true` | Run chunks on Rayon. Does not change results. |
| `chunk_size` | 4 096 | Samples per RNG stream and per task (≥ 1). Changes results. |
| `heston_scheme` | `QuadraticExponential` | Or `FullTruncation`. |
| `antithetic` | `false` | Pair $z$ with $-z$. |
| `control_variate` | `false` | Apply the control in the table above. |

`MonteCarloEngine::new(n_paths, n_steps, seed)` validates. Python: `MonteCarloEngine(heston_scheme="qe"|"full_truncation", antithetic=…, control_variate=…)`, `*_with_stats` methods, `McEstimate`.

## Accuracy and performance

PR #30 (criterion `benches/mc_bench.rs`, 15-core Apple Silicon):

| Bench | Time |
|---|---|
| BS terminal European, 1e7 paths, one thread | 80.1 ms (1.63× vs before) |
| Heston European 200k × 252, serial / parallel | 1.501 s / 130 ms (11.5× = 0.77 × cores) |
| Arithmetic Asian BS 100k × 252, parallel | 22.2 ms |
| Arithmetic Asian 1e6 × 252, peak RSS | 3.1 MB (was 2.10 GB) |

Statistics (PR #30, same tests as CI):
- Coverage: 192/200 = 0.96 of runs within ±1.96 SE.
- Antithetic SE ratio 0.727 against a theoretical 0.726.
- Geometric-Asian control on an ATM arithmetic Asian (σ = 0.2, T = 1, 252 fixings, 50k paths): SE 0.0347 → 0.00093 (37×).
- Heston European control (K = 90): SE 0.039 → 0.017.

Heston bias, from PR #29. Feller-violating case κ = 1.5, θ = 0.04, σ = 1, ρ = −0.9, v₀ = 0.04, r = 0.02, S = K = 100, T = 1; reference 6.8400; 500k paths:

| Steps | Full truncation | QE |
|---|---|---|
| 12 | +0.740 (74 SE) | −0.010 (1.3 SE) |
| 32 | +0.222 (25 SE) | −0.007 (1.1 SE) |
| 52 | +0.115 (11 SE) | −0.005 (0.7 SE) |
| 252 | +0.012 (1.1 SE) | −0.005 (0.6 SE) |

QE costs about 8% more per step than full truncation (100k × 252: 360 vs 337 ms). At 32 steps it beats full truncation at 252 steps.

## Design choices

- **Streaming functionals instead of stored paths.** Memory is O(chunks), not O(paths × steps).
- **Per-chunk streams merged in fixed order.** This makes results reproducible across thread counts. ChaCha8 `set_stream` was tried first and was slower (3.9 ns per u64).
- **One Heston step implementation** shared by pricing and `sample_paths`.
- **QE default.** It is unbiased at coarse steps when Feller fails, the usual case for calibrated equity parameters. The old absorption scheme was +1.43 (125 SE) at 52 steps and was removed.
- **Two normals per Heston step** for both schemes (QE's uniform is $\Phi(z_v)$), so the RNG layout does not depend on the scheme.
- **In-sample β.** One pass, no pilot run. The resulting bias is $O(1/n)$.

## Validation

| Test | File | Checks |
|---|---|---|
| `black_scholes_mc_within_three_stderr_of_analytic` | `tests/monte_carlo_engine.rs` | 400k paths, 3 strikes. |
| `heston_mc_within_three_stderr_of_fourier` | same | QE, 160k paths, 200 steps vs FFT. |
| `heston_qe_is_unbiased_with_coarse_steps_when_feller_fails` | same | 32 steps, 500k paths vs COS(1024, 16). |
| `heston_full_truncation_bias_shrinks_with_steps_when_feller_fails` | same | 200k paths; bias decreasing over {12, 52, 252}; < 3 SE at 252. |
| `heston_qe_terminal_spot_is_a_martingale` | same | Feller-violating case, 32 steps, 500k paths: near-zero-strike call = discounted forward within 3 SE. |
| `seeded_prices_are_bit_identical_for_any_thread_count` | same | Arithmetic Asian, BS and Heston, antithetic + control: serial = 1/2/8 threads. |
| `standard_error_has_nominal_coverage` | same | 200 runs × 20k paths, ±1.96 SE coverage in [0.90, 0.99]. |
| `antithetic_variates_cut_the_standard_error` | same | ATM BS call, 200k paths: SE ratio 0.726 ± 0.03. |
| `geometric_asian_closed_form_matches_monte_carlo` | same | 1e6 paths, n ∈ {12, 52, 252}. |
| `geometric_control_variate_cuts_arithmetic_asian_error_tenfold`, `control_variate_matches_large_plain_reference` (`#[ignore]`) | same | 50k paths: > 10× SE cut; 100k controlled paths vs a 1e7-path plain run within 3 SE. |
| `european_control_variate_reduces_heston_error`, `full_truncation_ignores_the_control_variate` | same | Heston QE, K = 90, 100k × 32: SE more than halved, within 3 SE of FFT. FT: the flag changes nothing (pins the #50 bug). |
| `sampled_paths_have_expected_shape_and_terminal_mean` | same | Shape, $S_0$, $\mathbb E[S_T]$. |
| `arithmetic_asian_peak_heap_does_not_grow_with_paths` | `tests/mc_memory.rs` | Peak heap growth < 1 MB for 1e6 paths. |
| `merged_chunk_moments_match_one_pass`, `antithetic_counts_both_legs_of_each_pair`, `mc_heston_european_deterministic_zero_variance`, `mc_asian_with_zero_vol_matches_discounted_deterministic_payoff`, `mc_digital_bs_is_close_to_analytical` | `src/monte_carlo_engine/mod.rs` | Merge, path count, degenerate limits. |
| `test_mc_statistics.py`, `test_heston_mc_scheme.py`, `test_parallelization.py` | `quantlib_py/tests` | Bindings. |

## Limitations and open issues

- No QMC (Sobol and Brownian bridge). It was optional in #11 and is not implemented.
- No American or path-dependent barrier products. `MonteCarloEngine` does not implement `PricingEngine<AmericanOption, _>`.
- The Asian functional has no fixing schedule: fixings are $n_{\text{steps}}+1$ equally spaced points including $S_0$. It does not use the schedules from #47.
- Results depend on `chunk_size` and change whenever the RNG layout changes.
- `control_variate` is ignored under Heston full truncation, although the FT spot step is an exact martingale. Bug: #50.
- No pathwise or likelihood-ratio Greeks: #18.
