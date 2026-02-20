# `quantlib_greeks`

Greeks (sensitivities) for QuantRustLib.

This crate is currently a **scaffold**. It is intended to host:

- analytical Greeks where closed-form formulas exist (e.g. Black-Scholes Delta/Gamma/Vega)
- numerical Greeks (finite differences) wrapping any `PricingEngine`
- pathwise / likelihood ratio estimators for Monte Carlo engines

Keeping Greeks in a dedicated crate helps avoid circular dependencies and keeps pricing engines focused on valuation.

## Testing

```bash
cargo test -p quantlib_greeks
```

