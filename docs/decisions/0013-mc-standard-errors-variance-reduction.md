# 0013. MC standard errors, antithetic and control variates
Status: Accepted (2026-10-16, PR #30, issue #11)

## Context
MC engines returned a bare `f64`, so tests used arbitrary tolerances and callers could not tell noise from bias. No variance reduction existed, and `AnalyticalEngine` rejected every `AsianOption`.

## Decision
- `price_with_stats` → `McEstimate { price, std_error, n_paths }` (`monte_carlo_engine/core.rs`).
- `antithetic`: each `Z` is paired with `−Z` and the pair mean is one sample, so the stderr stays correct.
- `control_variate`, β = Cov(Y, X)/Var(X) estimated per run: discounted S_T for European and digital (E = S₀e^{−qT}); the geometric Asian closed form for arithmetic Asians under BS; the arithmetic average of S (E[S_t] = S₀e^{(r−q)t}) otherwise.
- `geometric_asian_black_scholes` in `analytical_engine.rs` (Kemna–Vorst), discrete fixings matching the engine's S₀…S_T monitoring, or continuous; `AnalyticalEngine` prices continuously monitored geometric Asians.
- Both options off by default; Python: `MonteCarloEngine(antithetic=…, control_variate=…)`, `*_with_stats`, `McEstimate`.

## Alternatives considered
- Pilot run for β: extra cost; the in-run estimate's bias is O(1/N).
- QMC (Sobol + Brownian bridge): optional in #11, not implemented.

## Consequences
- Coverage 192/200 = 0.96 inside ±1.96·stderr (BS ATM, 20k paths).
- Antithetic cuts the ATM call stderr by 27.3%; the issue's 30% is unreachable: ratio √(1+ρ) with ρ = −0.473 gives at most 27.4%. The test checks against the theoretical ratio.
- Arithmetic Asian (σ=0.2, T=1, 252 fixings, 50k paths): stderr 0.0347 → 0.00093 (37×). Heston European K=90: 0.039 → 0.017.
- The playground takes the stderr from one run instead of 8 seeded batches.
- Tests (`tests/monte_carlo_engine.rs`): `standard_error_has_nominal_coverage`, `antithetic_variates_cut_the_standard_error`, `geometric_asian_closed_form_matches_monte_carlo`, `geometric_control_variate_cuts_arithmetic_asian_error_tenfold`, `control_variate_matches_large_plain_reference` (`#[ignore]`), `european_control_variate_reduces_heston_error`; `quantlib_py/tests/test_mc_statistics.py`.
