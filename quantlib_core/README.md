# `quantlib_core`

Core primitives and shared abstractions for QuantRustLib.

This crate is intentionally lightweight: it defines the **traits and common types** that products, models, and engines build on, plus a small **valuation context** for batch pricing with caching.

## What’s inside

- **Traits**
  - `FinancialProduct`: payoff definition + parameter map for caching/metadata
  - `StochasticModel`: parameter map for caching/metadata
  - `PricingEngine<P, M>`: engine interface for supported `(product, model)` pairs
- **Dates & calendars**
  - `SimpleDate`: simple `YYYY-MM-DD` date type (construction + parsing)
  - `MarketCalendar` + implementations like `WeekdayCalendar`, `ActualCalendar`
- **Valuation context**
  - `OptionValuationContext<E>`: wraps an engine and provides
    - optional cache (LRU-like)
    - optional parallel batch valuation (Rayon)
- **Errors**
  - `QuantLibError`: validation/date parse/serialization/engine errors
- **Enums & dataclasses**
  - `OptionType` (`Call`/`Put`)
  - `MarketQuote`, `CalibrationResult`

The crate re-exports these key symbols from its root, so most users can:

```rust
use quantlib_core::{OptionType, PricingEngine, SimpleDate, OptionValuationContext, QuantLibError};
```

## Typical usage

You usually don’t price directly from `quantlib_core`; instead you use:

- **Products** from `quantlib_products`
- **Models** from `quantlib_models`
- **Engines** from `quantlib_engines`

That said, `OptionValuationContext` is handy when you want batch valuation, caching, and optional parallelism without writing your own loop.

### Batch valuation with caching

```rust
use std::collections::BTreeMap;

use quantlib_core::{OptionType, OptionValuationContext, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = AnalyticalEngine;
    let ctx = OptionValuationContext::new(
        engine,
        true,   // cache_enabled
        10_000, // cache_maxsize
        true,   // parallel batch valuation
    );

    let model = BlackScholesModel::new(0.2, 0.03, 0.0)?;

    let options = vec![
        EuropeanOption::new_spot(
            100.0, 95.0, SimpleDate::new(2027, 2, 10)?, None, OptionType::Call, 1, None,
        )?,
        EuropeanOption::new_spot(
            100.0, 105.0, SimpleDate::new(2027, 2, 10)?, None, OptionType::Put, 1, None,
        )?,
    ];

    let kwargs = BTreeMap::new(); // included in cache key (useful for engine knobs)
    let prices = ctx.value_options(&options, &model, None, &kwargs)?;
    println!("{prices:?}");

    Ok(())
}
```

## Notes on caching keys

Caching uses `make_cache_key(product, model, kwargs)` which hashes a flattened representation of:

- `product.get_parameters()`
- `model.get_parameters()`
- `kwargs`

If you add new products/models, ensure `get_parameters()` is:

- **Stable**: deterministic ordering (use `BTreeMap`, not `HashMap`)
- **Complete**: include all parameters that affect the price (including `qty` if it scales payoff)

## Testing

Run only core tests:

```bash
cargo test -p quantlib_core
```

