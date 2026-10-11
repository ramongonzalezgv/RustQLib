# `quantlib_engines`

Pricing engines (algorithms) for QuantRustLib.

Engines implement `quantlib_core::PricingEngine<P, M>` for supported combinations of:

- product `P` from [`quantlib_products`](../quantlib_products/README.md)
- model `M` from [`quantlib_models`](../quantlib_models/README.md)

## Engines currently exposed

- **`AnalyticalEngine`**
  - Closed-form pricing for supported pairs:
    - `EuropeanOption` under `BlackScholesModel`, `BlackModel`, `SabrModel`
    - `DigitalOption` under `BlackScholesModel`, `BlackModel`
  - Returns `QuantLibError::Engine` for unsupported pairs (e.g. Asian/American under analytical)
- **`FFTEngine`**
  - Lewis/Carr–Madan FFT pricing of `EuropeanOption` under any `quantlib_models::FourierModel`
    (`BlackScholesModel`, `BlackModel`, `HestonModel`, `BatesModel`, `KouModel`)
  - Configurable grid: `n` (FFT size), `b` (integration bound), `interp`
    (`InterpolationKind::Linear`, or `Cubic`: local 8-point Lagrange)
  - The FFT plan and quadrature arrays are built once in `FFTEngine::new`; clones share them
  - `price_strip(model, spot, ttm, strikes, types)` prices many strikes at one exact maturity
    from a single FFT
- **`CosEngine`**
  - Fang–Oosterlee COS pricing for the same models; `CosEngine { n: 256, l: 12.0 }` by default
  - Best for single strikes or short strike lists (about 14× faster than the 4096-point FFT for
    one Heston price); also has `price_strip`
- **`MonteCarloEngine`**
  - A facade that delegates to specialized engines:
    - `MonteCarloEuropeanEngine`
    - `MonteCarloAsianEngine`
    - `MonteCarloDigitalEngine`
  - Supports:
    - `EuropeanOption`, `DigitalOption`, `AsianOption` under `BlackScholesModel`
    - `EuropeanOption`, `DigitalOption`, `AsianOption` under `HestonModel`
  - Heston variance discretization via `HestonScheme` (`with_heston_scheme`):
    - `QuadraticExponential` (default): Andersen (2008) QE with martingale correction, low bias
      even at 10–30 steps per year
    - `FullTruncation`: Euler with full truncation (Lord et al. 2010); needs fine steps
- **`BinomialEngine`**
  - Tree-based engine module (see exported types)
- **Batch helpers**
  - `calculate_prices_parallel`
  - `calculate_prices_parallel_per_model`

All key engines are re-exported from the crate root:

```rust
use quantlib_engines::{AnalyticalEngine, BinomialEngine, CosEngine, FFTEngine, MonteCarloEngine};
```

## Usage examples

### Analytical pricing (European, Black-Scholes)

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

### Fourier pricing (European, Heston)

```rust
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{CosEngine, FFTEngine};
use quantlib_models::HestonModel;
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
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01)?;

    let engine = FFTEngine::default(); // or FFTEngine::new(n, b, interp)?
    let price = engine.calculate_price(&option, &model)?;
    println!("price={price}");

    // Single strikes are cheaper with COS; strips share one CF sweep.
    let cos_price = CosEngine::default().calculate_price(&option, &model)?;
    let strip = engine.price_strip(&model, 100.0, 1.0, &[90.0, 100.0, 110.0], &[OptionType::Call; 3])?;
    println!("cos={cos_price}, strip={strip:?}");
    Ok(())
}
```

### Monte Carlo pricing (European, Black-Scholes)

```rust
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::MonteCarloEngine;
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

    let engine = MonteCarloEngine::new(200_000, 252, Some(42))?
        .with_parallel(true)
        .with_chunk_size(4_096);

    let price = engine.calculate_price(&option, &model)?;
    println!("price={price}");
    Ok(())
}
```

## Notes and constraints

- **Unsupported combinations** return `QuantLibError::Engine` with a descriptive message.
- **Quote convention matters**:
  - Some engines/models require `EuropeanOption::new_spot` (spot quoted),
  - others require `EuropeanOption::new_forward` (forward quoted).
  Engines call `require_spot()` / `require_forward()` to enforce this.
- **Monte Carlo reproducibility**:
  - Provide `seed: Some(u64)` to get deterministic results.
  - Parallel mode is designed to match serial results for seeded runs.
- **FFT domain**:
  - FFT pricing may error if `log_moneyness` is outside the interpolation domain.

## Benchmarks

This crate includes a Criterion benchmark target:

```bash
cargo bench -p quantlib_engines
```

## Testing

```bash
cargo test -p quantlib_engines
```

