# 0014. Control variates only under martingale-preserving discretisations
Status: Accepted (2026-10-16, PR #30 review round, issue #11)

## Context
Every control mean (E[S_T] = S₀e^{(r−q)T}, E[S_t]) assumes the discretised spot is an exact martingale. Heston full truncation is not, so its control-variate estimate was biased; the independent review of 7c858c4 found it.

## Decision
`McModel::martingale(scheme)` gates the control: BS always, Heston only with QE (whose K₀* correction keeps E[S_{n+1} | S_n] = S_n e^{(r−q)Δ}). Under full truncation `control_variate` is silently ignored (`MonteCarloEngine::estimate`).

## Alternatives considered
- Reject the combination with an error: not chosen; the plain estimate is still valid.

## Consequences
- Where the QE MGF does not exist, the step keeps Andersen's K₀ and the martingale breaks slightly; the control is then marginally biased (noted in `heston.rs`).
- Seeded prices are bit-identical to 7c858c4 except Heston FT + control variate (checked on 50 combinations).
- Test: `full_truncation_ignores_the_control_variate` in `quantlib_engines/tests/monte_carlo_engine.rs`.
