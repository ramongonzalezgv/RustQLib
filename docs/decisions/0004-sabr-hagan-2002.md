# 0004. SABR implied vol: Hagan et al. (2002) eq. (2.17a) exactly
Status: Accepted (2026-10-11, PR #25, issue #1)

## Context
All three SABR variants deviated from Hagan et al. (2002), "Managing Smile Risk": the log-moneyness series was added to the time correction instead of dividing (general), applied at β = 1 where it vanishes (lognormal: +2% at K/F = 0.5, ν → 0), and normal (β = 0) kept a spurious ρνα/4 term. `x(z)` cancelled catastrophically and used absolute ATM thresholds.

## Decision
In `quantlib_models/src/sabr_model.rs`:
- General and lognormal share `hagan_lognormal_vol(α, β, ρ, ν, F, K, T)`; lognormal passes β = 1. The series `1 + (1−β)² ln²/24 + (1−β)⁴ ln⁴/1920` is a divisor.
- Normal SABR drops the ρνα/4 term (it carries a β factor) in ATM and non-ATM paths.
- `x(z)` via `ln_1p` on a cancellation-free argument, using `√D − 1 = (z² − 2ρz)/(√D + 1)`, one branch per sign of z; for |z| < 1e-6 the Taylor series `z/x(z) ≈ 1 − ρz/2 + (2 − 3ρ²)z²/12`. One continuous formula across ATM.
- ρ must lie strictly in (−1, 1).

## Alternatives considered
- Patching each variant separately (issue #1's steps 1–3): general and lognormal would stay two code paths.
- Obłój correction or arbitrage-free SABR PDE: out of scope, tracked in #20.

## Consequences
- 13 QuantLib 1.43 `sabrVolatility` values matched to < 1e-12.
- Flat smile at ν = 1e-4 (lognormal and normal) within 1e-6; normal ATM equals α(1 + (2−3ρ²)ν²T/24) to 1e-14; β = 1 − 1e-9 matches lognormal within 1e-7.
- ATM continuity is checked against the local linear trend (1e-8·σ_ATM): the issue's |σ(F(1±ε)) − σ(F)| < 1e-8 cannot hold, the true skew moves the vol ~1.4e-7 at ε = 1e-6.
- Tests: `quantlib_models/tests/sabr_hagan.rs`, `quantlib_engines/tests/european_analytical_sabr.rs` (parity, monotone and convex calls on 50 strikes).
