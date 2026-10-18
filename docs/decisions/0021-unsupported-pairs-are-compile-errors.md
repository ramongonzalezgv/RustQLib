# 0021. Unsupported engine/product/model pairs are compile errors
Status: Accepted (2026-10-18, issue #57)

## Context
`AnalyticalEngine` had six `PricingEngine` impls (American under BS, Black, SABR; Asian under Black, SABR; digital under SABR) that only returned `Engine("... does not support ...")`. They turned a type error into a runtime error and only their own tests called them. [0012](0012-single-monte-carlo-engine.md) already removed the `MonteCarloEngine` ones.

## Decision
A pair with no pricing method has no impl. `quantlib_engines/src/lib.rs` asserts the absence with `static_assertions::assert_not_impl_any!`, so adding a stub back fails to compile.

## Alternatives rejected
- Keep the stubs for a uniform `PricingEngine<P, M>` surface: no generic caller needs it, and the Python bindings never called them.

## Consequences
- Rust API break: code calling these pairs no longer compiles.
- A real method for a listed pair (e.g. #13 American) adds its impl and drops the pair from the assertion.
- Runtime rejections that depend on values, not types (arithmetic `AsianOption` under BS analytical), stay.
