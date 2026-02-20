# `quantlib_models`

Stochastic models for QuantRustLib.

Models hold parameters (volatility, rates, jump parameters, etc.) and provide model-specific helpers (e.g. Black-Scholes `d1/d2`, Heston characteristic function) used by pricing engines in [`quantlib_engines`](../quantlib_engines/README.md).

All models implement `quantlib_core::StochasticModel` which supplies a **stable parameter map** used by caching and batch valuation.

## Models currently implemented

- **`BlackScholesModel`** (spot quoted)
  - Parameters: `sigma`, `r`, `q`
  - Helpers: `d1(spot, strike, ttm)`, `d2(spot, strike, ttm)`
- **`BlackModel`** (forward quoted; Black-76 style)
  - Parameters: `sigma`, `r`
- **`HestonModel`** (stochastic volatility)
  - Parameters: `kappa`, `theta`, `sigma`, `rho`, `v0`, `r`, `q`
  - Helper: `characteristic_function(u, ttm)`
- **`BatesModel`** (Heston + lognormal jumps)
  - Used by FFT pricing
- **`KouModel`** (double-exponential jump diffusion)
  - Used by FFT pricing
- **SABR family**
  - `SabrModel` plus variants (normal/lognormal/shifted) re-exported as:
    - `GeneralSabrModel`, `NormalSabrModel`, `LognormalSabrModel`, `ShiftedSabrModel`

## Usage examples

### Black-Scholes model

```rust
use quantlib_models::BlackScholesModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01)?;
    println!("sigma={}, r={}, q={}", model.sigma, model.r, model.q);
    Ok(())
}
```

### Heston model characteristic function (for FFT)

```rust
use num_complex::Complex64;
use quantlib_models::HestonModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01)?;
    let phi = model.characteristic_function(Complex64::new(1.0, -0.5), 1.0);
    println!("phi={phi:?}");
    Ok(())
}
```

## Notes

- **Validation**: constructors call `validate()` and return `QuantLibError::Validation` for invalid inputs.
- **Quote conventions**:
  - `BlackScholesModel` is used with **spot-quoted** products (e.g. `EuropeanOption::new_spot`).
  - `BlackModel` and SABR in this codebase are used with **forward-quoted** products (e.g. `EuropeanOption::new_forward`).
  - Engines will error if the product quote type doesn’t match the engine/model assumption.

## Testing

```bash
cargo test -p quantlib_models
```

