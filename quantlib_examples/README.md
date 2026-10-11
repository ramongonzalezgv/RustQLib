# `quantlib_examples`

Small runnable examples for QuantRustLib.

This crate is a binary (`src/main.rs`) that demonstrates the intended **product + model + engine** workflow.

## Run

From the workspace root:

```bash
cargo run -p quantlib_examples
```

## What it does

The current example builds:

- a spot-quoted `EuropeanOption`
- a `BlackScholesModel`
- the `AnalyticalEngine`

and prints the resulting option price.

If you add more examples (FFT, COS, Monte Carlo, batch valuation via `calculate_prices_parallel`), this README is the place to document them.

