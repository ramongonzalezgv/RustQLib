use core::f64::consts::PI;

use num_complex::Complex64;
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{AnalyticalEngine, FFTEngine, InterpolationKind};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::EuropeanOption;

#[test]
fn fft_heston_matches_reference_lewis_integral() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let product = EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let engine = FFTEngine::new(1 << 12, 200.0, InterpolationKind::Cubic).expect("engine");

    let fft_price = engine
        .calculate_price(&product, &model)
        .expect("fft heston call price");

    let reference_price = reference_lewis_call(
        product.require_spot().expect("spot"),
        product.k,
        product.ttm(),
        &model,
        200.0,
        20_000,
    );

    let abs_diff = (fft_price - reference_price).abs();
    assert!(
        abs_diff < 5.0e-3,
        "fft={fft_price}, reference={reference_price}, abs_diff={abs_diff}"
    );
}

#[test]
fn fft_heston_put_call_parity_holds() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let call = EuropeanOption::new_spot(
        100.0,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("call option");

    let put = EuropeanOption::new_spot(
        100.0,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Put,
        1,
        None,
    )
    .expect("put option");

    let model = HestonModel::new(1.7, 0.05, 0.45, -0.5, 0.04, 0.02, 0.01).expect("model");
    let engine = FFTEngine::default();

    let call_price = engine.calculate_price(&call, &model).expect("call");
    let put_price = engine.calculate_price(&put, &model).expect("put");

    let ttm = call.ttm();
    let rhs = call.require_spot().expect("spot") * (-(model.q) * ttm).exp()
        - call.k * (-(model.r) * ttm).exp();

    let lhs = call_price - put_price;
    assert!((lhs - rhs).abs() < 1.0e-8, "lhs={lhs}, rhs={rhs}");
}

#[test]
fn fft_heston_converges_to_black_scholes_for_low_vol_of_vol() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let product = EuropeanOption::new_spot(
        100.0,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let heston = HestonModel::new(8.0, 0.04, 1.0e-4, 0.0, 0.04, 0.02, 0.01).expect("heston");
    let bs = BlackScholesModel::new(0.2, 0.02, 0.01).expect("bs");

    let fft = FFTEngine::new(1 << 12, 200.0, InterpolationKind::Cubic).expect("fft");
    let analytical = AnalyticalEngine;

    let fft_price = fft.calculate_price(&product, &heston).expect("fft price");
    let bs_price = analytical
        .calculate_price(&product, &bs)
        .expect("analytical bs price");

    let abs_diff = (fft_price - bs_price).abs();
    assert!(
        abs_diff < 2.0e-2,
        "fft={fft_price}, bs={bs_price}, abs_diff={abs_diff}"
    );
}

#[test]
fn fft_returns_intrinsic_when_ttm_is_zero() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");

    let option = EuropeanOption::new_spot(
        120.0,
        100.0,
        valuation,
        Some(valuation),
        OptionType::Call,
        3,
        None,
    )
    .expect("option");
    let model = HestonModel::new(1.8, 0.04, 0.5, -0.6, 0.04, 0.02, 0.0).expect("model");
    let engine = FFTEngine::default();

    let price = engine.calculate_price(&option, &model).expect("price");
    assert!((price - 60.0).abs() < 1.0e-12, "price={price}");
}

#[test]
fn fft_returns_error_for_log_moneyness_outside_domain() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    // Very small strike pushes log-moneyness well outside the FFT interpolation grid
    // for this narrow bandwidth configuration.
    let option = EuropeanOption::new_spot(
        100.0,
        1.0e-20,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let engine = FFTEngine::new(256, 20.0, InterpolationKind::Linear).expect("engine");

    let err = engine
        .calculate_price(&option, &model)
        .expect_err("should fail out of interpolation domain");
    let message = err.to_string();
    assert!(
        message.contains("outside FFT interpolation domain"),
        "unexpected error message: {message}"
    );
}

#[test]
fn fft_handles_near_degenerate_heston_parameters() {
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
    .expect("option");

    // With sigma -> 0 and v0 == theta, Heston collapses near constant variance.
    let heston = HestonModel::new(1.0e-8, 0.04, 0.0, 0.0, 0.04, 0.02, 0.01).expect("heston");
    let bs = BlackScholesModel::new(0.2, 0.02, 0.01).expect("bs");

    let fft = FFTEngine::new(1 << 12, 200.0, InterpolationKind::Cubic).expect("fft");
    let analytical = AnalyticalEngine;

    let fft_price = fft.calculate_price(&option, &heston).expect("fft price");
    let bs_price = analytical.calculate_price(&option, &bs).expect("bs price");

    let abs_diff = (fft_price - bs_price).abs();
    assert!(
        fft_price.is_finite(),
        "near-degenerate heston fft price should be finite, got {fft_price}"
    );
    assert!(
        abs_diff < 2.0e-2,
        "fft={fft_price}, bs={bs_price}, abs_diff={abs_diff}"
    );
}

fn reference_lewis_call(
    spot: f64,
    strike: f64,
    ttm: f64,
    model: &HestonModel,
    upper_bound: f64,
    n_steps: usize,
) -> f64 {
    assert!(
        n_steps.is_multiple_of(2),
        "n_steps must be even for Simpson integration"
    );

    let du = upper_bound / n_steps as f64;
    let log_moneyness = (spot / strike).ln();
    let i = Complex64::new(0.0, 1.0);

    let mut integral_sum = 0.0;
    for index in 0..=n_steps {
        let u = index as f64 * du;
        let shifted_u = Complex64::new(u, -0.5);
        let phi = model.characteristic_function(shifted_u, ttm);
        let oscillatory = (i * u * log_moneyness).exp();
        let integrand = (oscillatory * phi / (u * u + 0.25)).re;

        let weight = if index == 0 || index == n_steps {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };

        integral_sum += weight * integrand;
    }

    let integral = du * integral_sum / 3.0;
    spot * (-(model.q) * ttm).exp()
        - (spot * strike).sqrt() * (-(model.r) * ttm).exp() * integral / PI
}
