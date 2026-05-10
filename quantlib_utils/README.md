# `quantlib_utils`

Utilities and supporting components for QuantRustLib.

## Calibration Engine

`quantlib_utils` now includes a full calibration framework with:

- **Optimization engines**:
  - `DifferentialEvolution` (global, bounded)
  - `NelderMead` (local, bounded simplex)
  - `HybridGlobalLocal` (DE warm-start + Nelder-Mead refinement)
- **Objective functions**:
  - `LeastSquares`
  - `RelativeLeastSquares`
  - `Huber { delta }` (robust to outliers)
- **Model calibrators**:
  - `calibrate_black_model`
  - `calibrate_black_scholes_model`
  - `calibrate_sabr_model`
  - `calibrate_heston_model` (FFT-based pricing inside objective)

Calibrators return `(model, CalibrationResult)` where `CalibrationResult` includes:

- calibrated parameter map
- objective value
- residual vectors (`errors`, `market_values`, `model_values`)
- fit diagnostics (`rmse`, `mape`, `max_error`)
- iteration count and optimizer status message

## Key Types

- `CalibrationConfig`: optimization method + objective + DE/NM settings
- `VanillaOptionCalibrationQuote`: quote for Black/BS/Heston calibration
  - includes `market_value_kind` (`Price` or `ImpliedVolatility`)
- `SabrCalibrationQuote`: quote for SABR volatility calibration
- `HestonFftConfig`: FFT grid/interpolation settings used while calibrating Heston

## Fixed-Parameter Calibration

To freeze a parameter while calibrating, set equal bounds for that parameter, e.g.
`(0.02, 0.02)`.

## Rust Example (Black-Scholes)

```rust
use quantlib_core::OptionType;
use quantlib_utils::{
    CalibrationConfig, CalibrationTarget, VanillaOptionCalibrationQuote, calibrate_black_scholes_model,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let quotes = vec![
        VanillaOptionCalibrationQuote::price(100.0, 100.0, 1.0, OptionType::Call, 9.75, 1.0),
        VanillaOptionCalibrationQuote::price(100.0, 105.0, 1.0, OptionType::Put, 9.10, 1.0),
    ];

    let config = CalibrationConfig::default();
    let (_model, result) = calibrate_black_scholes_model(
        &quotes,
        CalibrationTarget::Price,
        0.2,
        0.02,
        0.0,
        None,
        &config,
    )?;

    println!("{}", result.summary());
    Ok(())
}
```

## Testing

```bash
cargo test -p quantlib_utils
```
