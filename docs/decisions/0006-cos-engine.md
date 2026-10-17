# 0006. COS (Fang–Oosterlee) engine for few-strike pricing
Status: Accepted (2026-10-11, PR #26, issue #8)

## Context
The 4096-point FFT wastes ~4000 CF evaluations when only one or a few strikes are needed (single prices, bumped Greeks, scattered calibration quotes).

## Decision
`quantlib_engines::CosEngine { n: 256, l: 12.0 }` on the `FourierModel` trait (0005):
- Prices puts (bounded payoff) and gets calls by parity.
- Truncation range `c1 ± L·√(c2 + √c4)` (closed-form cumulants for BS, Black-76, Kou; numerical for Heston, Bates); its width does not depend on the strike, so `price_strip` evaluates the CF once per maturity.
- Exposed in Python as `CosEngine(n=256, l=12.0)`.

## Alternatives considered
- Lewis + Gauss–Laguerre quadrature engine: optional in #8, not done.
- N = 160 (issue target): no L reaches 1e-7 on the Fang–Oosterlee Heston case; the fat left tail needs the c4-widened range and N ≈ 256.
- COS as default for calibration and in the playground: not done.

## Consequences
- Single Heston price 10.9 µs vs 149.8 µs FFT (13.7×); 50-strike strip 32.1 µs vs 151.6 µs (4.7×).
- Fang–Oosterlee Heston call 5.785155450: error 4.5e-8 at N=256. COS-BS vs analytic < 1e-10 at N=128. COS vs fine FFT (2^15) < 1e-6 for Heston/Bates/Kou on a 10×10 grid.
- Tests: `quantlib_engines/tests/cos_engine.rs`, `fourier_allocations.rs`, `quantlib_py/tests/test_cos_engine.py`.
