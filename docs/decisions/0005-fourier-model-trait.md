# 0005. FourierModel trait and one generic FFT engine
Status: Accepted (2026-10-11, PR #26, issue #7)

## Context
`fft_engine.rs` had five near-identical `PricingEngine` impls (Heston, Bates, Kou, BS, Black-76). Every call built a new FFT planner, recomputed model-independent arrays and solved a global 4096-point cubic spline to read one value.

## Decision
- `quantlib_models::FourierModel`: `log_return_cf(u, T)` of ln(S_T/S_0) including the (r−q)T drift, `rate`, `dividend_yield`, `validate`, `cumulants` (numerical by default; closed form for BS, Black-76, Kou). Black-76 sets `FORWARD_QUOTED` and is BS on the forward with q = r, so no separate path.
- One generic impl per Fourier engine; shared plumbing in `quantlib_engines/src/fourier.rs`.
- `FFTEngine::new` builds the plan and quadrature arrays once (`Arc<FftGrid>`, shared by clones), so `FFTEngine` is `Clone`, not `Copy`.
- Local 8-point Lagrange interpolation on the uniform log-strike grid (O(1), allocation-free) replaces the global spline; standard Carr–Madan Simpson weights; `price_strip` for many strikes at one maturity.
- Heston CF: algebraic principal square root, two fewer complex divisions, no `hypot`; same formulation and branch.

## Alternatives considered
- `thread_local!` plan cache keyed by `n` (offered in #7): per-engine precomputation chosen.
- 4-point cubic Lagrange or Catmull-Rom (suggested in #7): an 8-point stencil chosen.

## Consequences
- `fft_engine.rs` 479 → 241 lines. Default Heston bench 284.5 → 150.2 µs (1.89×; the issue's 2× was not reached, the rest is CF cost).
- Prices moved by more than 1e-7 vs the old spline, toward the truth (Heston T=1 error 5.6e-6 vs 3.4e-5).
- FFT-BS/Black vs analytic ≤ 1e-6 on K ∈ [50, 200], T ∈ {0.1, …, 5}; T = 0.1 needs n=8192, b=400 (defaults unchanged: `DEFAULT_N` = 4096, `DEFAULT_B` = 200).
- API change: fields read via `n()`, `b()`, `interp()`.
- Tests: `fft_black_and_bs_engine.rs`, `fft_{heston,bates,kou}_engine.rs`, `fourier_allocations.rs` (≤ 3 allocations per price, independent of strike count; per-thread counter since 6dc12c8).
