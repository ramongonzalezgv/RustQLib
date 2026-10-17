# 0003. Web playground: Rust compiled to WASM, thin TypeScript UI
Status: Accepted (2026-10-11, PR #24, issue #21)

## Context
Comparing every engine on the same option exposes inconsistencies (wrong SABR smile, biased MC, binomial odd/even zig-zag). A zero-install page also serves users.

## Decision
- Crate `quantlib_wasm`: request/response serde types and all logic in plain Rust (`src/api.rs`, tested natively); `src/bindings.rs` holds the `wasm-bindgen` exports, compiled only for `wasm32`. Pricing logic stays in the library crates.
- `web/`: Vite + TypeScript + ECharts. Pricing runs in a module Web Worker; a stale run is cancelled by terminating the worker, so inputs stay responsive.
- All inputs are in the `?s=` query parameter (base64url JSON), so a link reproduces the prices.
- MC runs with `with_parallel(false)`; `rayon` is not feature-gated, since rayon-core falls back to the current thread on wasm32.
- Deployed to GitHub Pages by `.github/workflows/pages.yml` on push to `main` (manual-only until #1 was fixed; push trigger in 9f1740a).

## Alternatives considered
- Server-side pricing: GitHub Pages serves static files only.
- A `parallel` cargo feature for rayon: unnecessary, the fallback works.
- Multithreaded WASM (`wasm-bindgen-rayon` + COOP/COEP): deferred.

## Consequences
- WASM 511 KB (188 KB gzipped), JS 612 KB (206 KB gzipped).
- Native and WASM prices agree to 1e-12 relative, not bit-for-bit: native `exp`/`ln` come from system libm.
- Guards: `quantlib_wasm/tests/web.rs` (`wasm-pack test --node`, `NATIVE_REFERENCE`), `web/e2e/smoke.mjs` (Playwright, 16 scenarios after #25), CI `web` job.
- Seeded MC changes (0010) require regenerating `NATIVE_REFERENCE`.
