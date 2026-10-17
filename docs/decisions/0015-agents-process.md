# 0015. AGENTS.md: least code, independent write / review / test
Status: Accepted (2026-10-16, PR #30, commit b201bf3)

## Context
Most code is written by AI agents, which tend toward verbose code: speculative options, forwarding wrappers, restating comments, copy-pasted tests. Separate reviewer and tester agents checked the first MC rewrite (7c858c4) and found the biased Heston control variate (0014) and 96 removable lines.

## Decision
`AGENTS.md` at the repo root is binding for every change:
- Write the least code that stays clear: no speculative generality, no forwarding wrappers or one-use helpers, reuse existing types (`SimpleDate`, `MarketCalendar`, `QuantLibError`, the MC driver), comments say why, validate once at construction.
- Tests are few and sharp; Monte Carlo tests are statistical (`k · std_error`); similar cases are table-driven.
- Done means `fmt`, `clippy -D warnings`, `test`, `bench --no-run` pass (plus `maturin` + `pytest` when bindings change).
- Non-trivial issues are written, reviewed and tested by separate agents; reviewer and tester judge only against the issue's acceptance criteria and `AGENTS.md`.

## Alternatives considered
- Single-agent write-and-verify: misses what the author assumed (0014).

## Consequences
- PR #30's review round: MC module 1377 → 1281 lines, six `MonteCarloPricing` impls → three. PR #47 was built this way (69 independent black-box checks).
