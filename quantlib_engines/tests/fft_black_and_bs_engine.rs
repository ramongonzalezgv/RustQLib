use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::FFTEngine;
use quantlib_models::{BlackModel, BlackScholesModel};
use quantlib_products::EuropeanOption;

#[test]
fn fft_black_scholes_rejects_forward_quoted_option() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_forward(
        102.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("forward option");
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("bs model");
    let engine = FFTEngine::default();

    let err = engine
        .calculate_price(&option, &model)
        .expect_err("FFT BS should reject forward-quoted product");
    assert!(
        err.to_string().contains("spot-based pricing requested"),
        "unexpected error: {err}"
    );
}

#[test]
fn fft_black_rejects_spot_quoted_option() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("spot option");
    let model = BlackModel::new(0.2, 0.03).expect("black model");
    let engine = FFTEngine::default();

    let err = engine
        .calculate_price(&option, &model)
        .expect_err("FFT Black should reject spot-quoted product");
    assert!(
        err.to_string().contains("forward-based pricing requested"),
        "unexpected error: {err}"
    );
}
