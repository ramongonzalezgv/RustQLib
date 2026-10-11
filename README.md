# QuantRustLib (`quantlib-rs`)

QuantRustLib is a Rust workspace for **fast, safe option pricing** with a modular design inspired by QuantLib-style separation of concerns:

- **Products** (payoffs + contract details)
- **Models** (stochastic dynamics + parameters)
- **Engines** (pricing algorithms)
- **Core** (dates, calendars, traits, valuation context)

This repository also includes **Python bindings** (`quantlib_py`) for high-throughput pricing from Python.

## Workspace layout

- [`quantlib_core`](./quantlib_core/README.md): shared primitives (dates/calendars), traits, errors
- [`quantlib_products`](./quantlib_products/README.md): products (European, Digital, Asian, American)
- [`quantlib_models`](./quantlib_models/README.md): models (Black(-Scholes), Heston, SABR, Bates, Kou)
- [`quantlib_engines`](./quantlib_engines/README.md): pricing engines (Analytical, Binomial, Monte Carlo, FFT) + batch helpers
- [`quantlib_greeks`](./quantlib_greeks/README.md): Greeks (currently scaffold)
- [`quantlib_utils`](./quantlib_utils/README.md): utilities (calibration engines, objective functions, model calibrators)
- [`quantlib_examples`](./quantlib_examples/README.md): small runnable examples
- [`quantlib_py`](./quantlib_py/README.md): PyO3 bindings (functions + OO “engine” classes) and notebooks
- [`quantlib_wasm`](./web/README.md) + [`web`](./web/README.md): WebAssembly build and browser pricing playground

## Quickstart (Rust)

Build and run the included example binary:

```bash
cargo run -p quantlib_examples
```

Run all tests:

```bash
cargo test --all
```

### Minimal pricing example (Rust)

The library is designed around the trio **product + model + engine**:

```rust
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let option = EuropeanOption::new_spot(
        100.0,
        100.0,
        SimpleDate::new(2027, 2, 10)?,
        Some(SimpleDate::new(2026, 2, 10)?),
        OptionType::Call,
        1,
        None,
    )?;

    let model = BlackScholesModel::new(0.2, 0.05, 0.01)?;
    let engine = AnalyticalEngine;

    let price = engine.calculate_price(&option, &model)?;
    println!("price={price}");
    Ok(())
}
```

## Quickstart (Python bindings)

The `quantlib_py` crate exposes a Python module built with `maturin`.

```bash
cd quantlib_py
maturin develop
python -c "import quantlib_py as ql; print(ql.__doc__)"
```

See [`quantlib_py`](./quantlib_py/README.md) for a detailed API snapshot and NumPy batch guidance.

## Pricing playground (browser)

[`web/`](./web/README.md) is a browser UI that prices single options with this library compiled to WebAssembly: all engines side by side, payoff, Monte Carlo path and vol-smile charts, and shareable links.

```bash
rustup target add wasm32-unknown-unknown
cd web && npm install && npm run dev
```

## Design overview

### Traits (core abstraction layer)

In `quantlib_core`:

- **`FinancialProduct`**: `payoff(&[f64]) -> Vec<f64>`
- **`PricingEngine<P, M>`**: `calculate_price(&P, &M) -> Result<f64, QuantLibError>`

Concrete products/models implement these traits; engines implement `PricingEngine` for supported `(product, model)` pairs.

### Batch valuation

`quantlib_engines` provides `calculate_prices_parallel` and `calculate_prices_parallel_per_model` (Rayon) for pricing many instruments with any engine.

Engines are pure and keep no internal cache: pricing is cheap enough (e.g. ~25 ns analytic, ~11 µs COS Heston) that a cache lookup rarely pays off. Callers that re-price identical inputs (e.g. Monte Carlo on an unchanged book) can memoize on keys they control.

## Supported pricing methods (high level)

- **Analytical**: closed-form for supported combinations (e.g. European/Digital under Black-Scholes, European under Black-forward, European under SABR with implied vol)
- **Monte Carlo**: path simulation for supported combinations (European/Digital/Asian under Black-Scholes and Heston)
- **FFT**: characteristic-function pricing for European options under several models (Black-Scholes, Black-forward, Heston, Bates, Kou)
- **Binomial**: included as an engine module (see `quantlib_engines`)

## Calibration (high level)

- `quantlib_utils` includes bounded calibration engines:
  - Differential Evolution (global)
  - Nelder-Mead (local)
  - Hybrid global-local refinement
- Model-specific calibrators are available for:
  - Black
  - Black-Scholes
  - SABR
  - Heston (FFT-priced objective)
- Results include parameter estimates and fit diagnostics (`rmse`, `mape`, residual vectors).

For exact coverage and constraints, see the engine docs: [`quantlib_engines`](./quantlib_engines/README.md).

## License

MIT. See workspace `Cargo.toml` (`license = "MIT"`).
