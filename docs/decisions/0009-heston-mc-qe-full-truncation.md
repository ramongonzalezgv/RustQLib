# 0009. Heston Monte Carlo: Andersen QE (default) and full truncation, no absorption
Status: Accepted (2026-10-11, PR #29, issue #6)

## Context
The Euler scheme floored the variance every step ("absorption"), adding mass at v = 0 and biasing prices, worst when the Feller condition 2κθ ≥ σ² fails, as it usually does for calibrated equity parameters.

## Decision
`HestonScheme::{QuadraticExponential, FullTruncation}`, default QE; absorption removed. Code: `quantlib_engines/src/monte_carlo_engine/heston.rs` (one `HestonStepper` for all simulators).
- QE (Andersen 2008): quadratic branch for ψ ≤ 1.5, exponential branch with a point mass at 0 above; γ₁ = γ₂ = ½; martingale-corrected K₀*, falling back to K₀ where the MGF does not exist; deterministic fallback for σ ≈ 0.
- Full truncation (Lord, Koekkoek & van Dijk 2010): variance never floored, only v⁺ enters drift and diffusion.
- Two normals per step (QE's uniform is Φ(z_v)). Python: `MonteCarloEngine(heston_scheme="qe" | "full_truncation")`.

## Alternatives considered
- Full truncation only: needs ~252 steps for the bias QE has at 12.
- Exact (Broadie–Kaya) sampling: not attempted; #6 recommended QE.

## Consequences
Bias vs Fourier 6.8400 (κ=1.5, θ=0.04, σ=1, ρ=−0.9, v₀=0.04, S=K=100, T=1; 500k paths, stderr 0.007–0.012):

| Steps | Absorption | Full truncation | QE |
|---|---|---|---|
| 12 | +2.111 | +0.740 | −0.010 |
| 52 | +1.431 (125 se) | +0.115 | −0.005 |
| 252 | +0.999 | +0.012 | −0.005 |

- QE costs ~8% more per step; at 32 steps it beats any Euler variant at 252, ~6.6× faster.
- Heston MC prices change for every user (the fix).
- Tests (`quantlib_engines/tests/monte_carlo_engine.rs`): `heston_qe_is_unbiased_with_coarse_steps_when_feller_fails`, `heston_full_truncation_bias_shrinks_with_steps_when_feller_fails`, `heston_qe_terminal_spot_is_a_martingale`; `quantlib_py/tests/test_heston_mc_scheme.py`.
