# `quantlib_products`

Financial products (contracts) for QuantRustLib.

Products are responsible for:

- **Validation** of contract inputs (finite numbers, date ordering, etc.)
- **Payoff logic** at maturity (or from simulated paths for path-dependent products)

Pricing is performed by engines in [`quantlib_engines`](../quantlib_engines/README.md).

## Products currently implemented

- **`EuropeanOption`**
  - Quote types: **spot** (`new_spot`) or **forward** (`new_forward`)
  - Helpers: `ttm()`, `days_to_maturity()`, `require_spot()`, `require_forward()`
- **`DigitalOption`**
  - Wraps a `EuropeanOption` + digital payout settings
  - Payoff types: `DigitalPayoffType::{CashOrNothing, AssetOrNothing}`
- **`AsianOption`**
  - Averaging: `AveragingType::{Arithmetic, Geometric}`
  - Helpers: `average_from_paths(&[Vec<f64>])`, `payoff_from_paths(&[Vec<f64>])`
- **`AmericanOption`**
  - Present in the crate (engine support depends on `quantlib_engines`)

All types are re-exported from the crate root:

```rust
use quantlib_products::{EuropeanOption, DigitalOption, AsianOption, AmericanOption};
```

## Usage examples

### European option (spot-quoted)

```rust
use quantlib_core::{OptionType, SimpleDate};
use quantlib_products::EuropeanOption;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = EuropeanOption::new_spot(
        100.0,
        100.0,
        SimpleDate::new(2027, 2, 10)?,
        Some(SimpleDate::new(2026, 2, 10)?),
        OptionType::Call,
        1,
        None,
    )?;

    println!("ttm={}", opt.ttm());
    Ok(())
}
```

### European option (forward-quoted)

Some engines/models are expressed naturally in forward measure (e.g. Black-76 style pricing).

```rust
use quantlib_core::{OptionType, SimpleDate};
use quantlib_products::EuropeanOption;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = EuropeanOption::new_forward(
        101.0,
        100.0,
        SimpleDate::new(2027, 2, 10)?,
        None,
        OptionType::Put,
        1,
        None,
    )?;

    // Engines will error if you use a spot-only model/engine on a forward-quoted option (and vice versa).
    assert!(opt.forward().is_some());
    Ok(())
}
```

### Digital option (cash-or-nothing)

```rust
use quantlib_core::{OptionType, SimpleDate};
use quantlib_products::{DigitalOption, DigitalPayoffType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let digital = DigitalOption::new_spot(
        100.0,
        95.0,
        SimpleDate::new(2027, 2, 10)?,
        None,
        OptionType::Call,
        2,
        10.0, // payout
        DigitalPayoffType::CashOrNothing,
        None,
    )?;

    println!("strike={}", digital.strike());
    Ok(())
}
```

### Asian option payoff from simulated paths

This product exposes helpers for path-dependent payoffs. Engines (Monte Carlo) typically use these internally, but they are also useful for custom simulations.

```rust
use quantlib_core::{OptionType, SimpleDate};
use quantlib_products::{AsianOption, AveragingType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let asian = AsianOption::new(
        100.0,
        100.0,
        SimpleDate::new(2027, 2, 10)?,
        None,
        OptionType::Call,
        1,
        AveragingType::Arithmetic,
        None,
    )?;

    let paths = vec![vec![90.0, 100.0, 110.0], vec![80.0, 90.0, 100.0]];
    let payoffs = asian.payoff_from_paths(&paths)?;
    println!("{payoffs:?}");
    Ok(())
}
```

## Testing

```bash
cargo test -p quantlib_products
```

