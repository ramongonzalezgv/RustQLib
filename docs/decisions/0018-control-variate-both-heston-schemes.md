# 0018. Control variates under both Heston schemes
Status: Accepted (2026-10-17, issue #50); supersedes 0014

## Context
0014 gated the control variate to QE on the premise that the full-truncation spot is not a martingale. That premise was wrong: with $v^+ = \max(v_n, 0)$ known at the start of the step, $S_{n+1} = S_n\exp((r-q)\Delta - \tfrac12 v^+\Delta + \sqrt{v^+}\,\Delta W_1)$ gives $\mathbb E[S_{n+1} \mid \mathcal F_n] = S_n e^{(r-q)\Delta}\,\mathbb E[e^{-\frac12 v^+\Delta + \sqrt{v^+}\Delta W_1}] = S_n e^{(r-q)\Delta}$ exactly. QE with the $K_0^*$ correction is also exact, except on steps that fall back to $K_0$ when the MGF does not exist (`heston.rs`).

## Decision
Apply the control variate under both schemes; `McModel::martingale` is deleted.

## Alternatives considered
- Keep the gate: loses variance reduction under FT for no bias gain.

## Consequences
- FT + `control_variate` now reduces the standard error; FT discretisation bias is unchanged.
- Seeded prices for BS and Heston QE are bit-identical to before (32 combinations of product, model, antithetic and control checked).
- Tests (`quantlib_engines/tests/monte_carlo_engine.rs`): `heston_terminal_spot_is_a_martingale` (FT at 4 steps/year, σ = 1, 1e6 paths: forward within 3 SE), `european_control_variate_reduces_heston_error` (QE and FT: SE more than halved, within 3 SE of FFT).
