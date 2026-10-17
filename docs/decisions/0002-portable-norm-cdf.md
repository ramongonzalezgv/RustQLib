# 0002. One portable, tail-accurate normal CDF
Status: Accepted (2026-10-11, PR #23, issue #3)

## Context
`norm_cdf` existed twice as `0.5 * (1 + erf(x/√2))` with `erf` from system libm via `unsafe extern "C"`. The left tail cancels: Φ(−8) had ~10% relative error, Φ(−9) returned 0. MSVC has no `m.lib` (CI failed with `LNK1181`), and on `wasm32-unknown-unknown` the FFI became an unresolved `env.erf` import, blocking #21.

## Decision
`quantlib_core::math` is the only home of:
- `norm_cdf(x) = 0.5 · erfc(−x/√2)` with `erfc` from the pure-Rust `libm` crate;
- `norm_pdf` with a precomputed 1/√(2π);
- `inv_norm_cdf`: Acklam's approximation plus one Halley step (for QMC #11 and implied vol #19).

## Alternatives considered
- Keep system libm: not portable, needs `unsafe`.
- Own Cody (1969) / West (2005) `erfc` (offered in #3): `libm::erfc` used instead.
- Wichura AS241 for the inverse (offered in #3): Acklam + Halley chosen.

## Consequences
- `norm_cdf(−10)` within 1e-13 and `norm_cdf(−5)` within 1e-14 relative of mpmath; `norm_cdf(−37.5) > 0`; symmetry within 2.3e-16 on [−8, 8] (one ulp of 1.0).
- Round trip `inv_norm_cdf(norm_cdf(x))` within 1e-12·max(1, |x|) on [−8, 2]; the upper tail is checked through symmetry, since Φ(x) near 1 cannot be inverted in f64.
- Deep-OTM BS put (S=100, K=20, σ=0.2, r=5%, T=183/365) = 2.1603814591587665e-31 within 1e-10 relative.
- No `unsafe` or `link(name` in library crates; Windows back in CI.
- Tests: `quantlib_core/src/math.rs` unit tests, `deep_otm_put_is_positive_and_accurate` in `quantlib_engines/tests/european_analytical_black_scholes.rs`.
