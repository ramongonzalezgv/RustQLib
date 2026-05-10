//! Utilities crate for QuantRustLib.
//!
//! Includes calibration engines and model-specific calibration helpers.

pub mod calibration;

pub use calibration::{
    BLACK_PARAMETER_NAMES, BLACK_SCHOLES_PARAMETER_NAMES, CalibrationConfig, CalibrationMethod,
    CalibrationTarget, DifferentialEvolutionConfig, HESTON_PARAMETER_NAMES, HestonFftConfig,
    NelderMeadConfig, ObjectiveKind, OptimizationResult, SABR_PARAMETER_NAMES,
    SabrCalibrationQuote, VanillaMarketValueKind, VanillaOptionCalibrationQuote,
    calibrate_black_model, calibrate_black_scholes_model, calibrate_heston_model,
    calibrate_sabr_model, default_black_bounds, default_black_scholes_bounds,
    default_heston_bounds, default_sabr_bounds,
};
