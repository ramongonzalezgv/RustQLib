# RustQLib pricing playground

A browser UI for pricing single options with RustQLib. The pricing code is the
Rust library itself, compiled to WebAssembly (`quantlib_wasm`) and run in a Web
Worker; nothing is sent to a server.

- Pick a product (European, American, digital, Asian), a model (Black–Scholes,
  Black-76, Heston, Bates, Kou, SABR) and its parameters.
- Every engine that supports the combination prices it side by side
  (analytical, FFT, binomial, Monte Carlo with ± 3 standard errors), with the
  difference to the reference engine and the run time.
- Charts: payoff at expiry vs value today, Monte Carlo sample paths with
  quantile bands and the terminal distribution, and the implied-vol smile.
- The URL holds all inputs (Share button), so a link reproduces the prices.

## Develop

Requirements: Rust with the `wasm32-unknown-unknown` target and Node 22+.

```bash
rustup target add wasm32-unknown-unknown
cd web
npm install
npm run dev      # debug WASM build + Vite dev server
npm run build    # release WASM (wasm-pack) + typecheck + production bundle in dist/
npm run smoke    # Playwright smoke test against dist/ (SCREENSHOTS=1 to save PNGs)
```

`wasm-pack test --node ../quantlib_wasm` checks that WASM prices match the
native ones.

## Layout

| Path | Purpose |
| --- | --- |
| `../quantlib_wasm/src/api.rs` | Request/response types and all pricing logic (plain Rust, tested natively) |
| `../quantlib_wasm/src/bindings.rs` | `wasm-bindgen` exports (wasm32 only) |
| `src/worker.ts` | Loads the WASM module and serves pricing calls off the main thread |
| `src/client.ts` | Promise RPC to the worker; cancels stale runs by restarting it |
| `src/state.ts` | Model parameter metadata, defaults, shareable-URL encoding |
| `src/charts.ts` | ECharts options for the three charts |
| `src/main.ts` | Form, engine table, run loop |

## Deploy

`.github/workflows/pages.yml` builds and publishes `dist/` to GitHub Pages on
every push to `main` (and on demand via `workflow_dispatch`):
<https://ramongonzalezgv.github.io/RustQLib/>
