use quantlib_core::{OptionType, PricingEngine, SimpleDate, norm_cdf};
use quantlib_engines::{FFTEngine, InterpolationKind};
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

fn black_scholes_call(spot: f64, strike: f64, ttm: f64, sigma: f64, r: f64, q: f64) -> f64 {
    let sd = sigma * ttm.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * sigma * sigma) * ttm) / sd;
    spot * (-q * ttm).exp() * norm_cdf(d1) - strike * (-r * ttm).exp() * norm_cdf(d1 - sd)
}

const MATURITIES: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 5.0];

fn strike_grid() -> Vec<f64> {
    (0..=30).map(|i| 50.0 + 5.0 * f64::from(i)).collect()
}

/// The default log-strike spacing (2π/200 ≈ 0.031) is too coarse for 1e-6 at
/// T = 0.1, where the price curvature lives on a scale of σ√T ≈ 0.06; halving
/// it (n = 8192, b = 400, same dx) brings the error to ~2e-7.
fn engine_for(ttm: f64) -> FFTEngine {
    if ttm < 0.5 {
        FFTEngine::new(1 << 13, 400.0, InterpolationKind::Cubic).expect("engine")
    } else {
        FFTEngine::default()
    }
}

#[test]
fn fft_black_scholes_matches_analytic_on_strike_maturity_grid() {
    let (spot, sigma, r, q) = (100.0, 0.2, 0.03, 0.01);
    let model = BlackScholesModel::new(sigma, r, q).expect("bs model");
    let strikes = strike_grid();

    for ttm in MATURITIES {
        let engine = engine_for(ttm);
        for option_type in [OptionType::Call, OptionType::Put] {
            let types = vec![option_type; strikes.len()];
            let prices = engine
                .price_strip(&model, spot, ttm, &strikes, &types)
                .expect("fft strip");
            for (&strike, &price) in strikes.iter().zip(&prices) {
                let call = black_scholes_call(spot, strike, ttm, sigma, r, q);
                let expected = match option_type {
                    OptionType::Call => call,
                    OptionType::Put => call - spot * (-q * ttm).exp() + strike * (-r * ttm).exp(),
                };
                let err = (price - expected).abs();
                assert!(
                    err <= 1.0e-6,
                    "T={ttm}, K={strike}, {option_type:?}: fft={price}, analytic={expected}, err={err:e}"
                );
            }
        }
    }
}

#[test]
fn fft_black_matches_analytic_on_strike_maturity_grid() {
    let (forward, sigma, r) = (100.0, 0.2, 0.03);
    let model = BlackModel::new(sigma, r).expect("black model");
    let strikes = strike_grid();
    let types = vec![OptionType::Call; strikes.len()];

    for ttm in MATURITIES {
        let engine = engine_for(ttm);
        let prices = engine
            .price_strip(&model, forward, ttm, &strikes, &types)
            .expect("fft strip");
        for (&strike, &price) in strikes.iter().zip(&prices) {
            // Black-76 is Black–Scholes on the forward with q = r.
            let expected = black_scholes_call(forward, strike, ttm, sigma, r, r);
            let err = (price - expected).abs();
            assert!(
                err <= 1.0e-6,
                "T={ttm}, K={strike}: fft={price}, analytic={expected}, err={err:e}"
            );
        }
    }
}

#[test]
fn fft_single_option_price_matches_strip() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");
    let option = EuropeanOption::new_spot(
        100.0,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Put,
        3,
        None,
    )
    .expect("spot option");
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("bs model");
    let engine = FFTEngine::default();

    let single = engine.calculate_price(&option, &model).expect("price");
    let strip = engine
        .price_strip(&model, 100.0, option.ttm(), &[95.0], &[OptionType::Put])
        .expect("strip");
    assert!((single - 3.0 * strip[0]).abs() < 1.0e-12);
}
