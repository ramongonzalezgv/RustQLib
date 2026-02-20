use core::f64::consts::PI;

use num_complex::Complex64;
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{FFTEngine, InterpolationKind};
use quantlib_models::{BatesModel, HestonModel};
use quantlib_products::EuropeanOption;

#[test]
fn fft_bates_matches_reference_lewis_integral() {
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

    let model =
        BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0, 0.35, -0.08, 0.25).expect("model");
    let engine = FFTEngine::new(1 << 12, 200.0, InterpolationKind::Cubic).expect("engine");

    let fft_price = engine
        .calculate_price(&product, &model)
        .expect("fft bates call price");

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
fn fft_bates_put_call_parity_holds() {
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

    let model =
        BatesModel::new(1.7, 0.05, 0.45, -0.5, 0.04, 0.02, 0.01, 0.4, -0.1, 0.2).expect("model");
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
fn fft_bates_reduces_to_heston_when_jump_intensity_is_zero() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_spot(
        100.0,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let bates =
        BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0, 0.0, -0.08, 0.25).expect("bates");
    let heston = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("heston");
    let engine = FFTEngine::new(1 << 12, 200.0, InterpolationKind::Cubic).expect("fft");

    let bates_price = engine.calculate_price(&option, &bates).expect("bates");
    let heston_price = engine.calculate_price(&option, &heston).expect("heston");

    assert!(
        (bates_price - heston_price).abs() < 1.0e-10,
        "bates={bates_price}, heston={heston_price}"
    );
}

fn reference_lewis_call(
    spot: f64,
    strike: f64,
    ttm: f64,
    model: &BatesModel,
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
