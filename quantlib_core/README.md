# `quantlib_core`

Core primitives and shared abstractions for QuantRustLib.

This crate is intentionally lightweight: it defines the **traits and common types** that products, models, and engines build on.

## What’s inside

- **Traits**
  - `FinancialProduct`: payoff definition
  - `PricingEngine<P, M>`: engine interface for supported `(product, model)` pairs
- **Dates & calendars**
  - `SimpleDate`: simple `YYYY-MM-DD` date type (construction + parsing)
  - `MarketCalendar` + implementations like `WeekdayCalendar`, `ActualCalendar`
- **Errors**
  - `QuantLibError`: validation/date parse/serialization/engine errors
- **Enums & dataclasses**
  - `OptionType` (`Call`/`Put`)
  - `CalibrationResult`

The crate re-exports these key symbols from its root, so most users can:

```rust
use quantlib_core::{OptionType, PricingEngine, QuantLibError, SimpleDate};
```

## Typical usage

You usually don’t price directly from `quantlib_core`; instead you use:

- **Products** from `quantlib_products`
- **Models** from `quantlib_models`
- **Engines** from `quantlib_engines`

For batch pricing, use `calculate_prices_parallel` / `calculate_prices_parallel_per_model` from `quantlib_engines`.

## Testing

Run only core tests:

```bash
cargo test -p quantlib_core
```
