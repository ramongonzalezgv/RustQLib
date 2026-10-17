mod common;

use common::{assert_close, forward_option, ttm_of};
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::{BlackModel, SabrModel};

const DAYS: i64 = 365;
const R: f64 = 0.03;

fn models() -> Vec<(&'static str, SabrModel)> {
    vec![
        (
            "general",
            SabrModel::general(2.5, 0.5, -0.3, 0.4, R, None).unwrap(),
        ),
        (
            "lognormal",
            SabrModel::lognormal(0.25, -0.3, 0.4, R, None).unwrap(),
        ),
        (
            "normal",
            SabrModel::normal(25.0, -0.3, 0.4, R, None).unwrap(),
        ),
    ]
}

fn price_at(model: &SabrModel, f: f64, k: f64, option_type: OptionType) -> f64 {
    AnalyticalEngine
        .calculate_price(&forward_option(f, k, DAYS, option_type), model)
        .expect("SABR price")
}

fn price(model: &SabrModel, k: f64, option_type: OptionType) -> f64 {
    price_at(model, 100.0, k, option_type)
}

/// 50 strikes from 60 to 158 (moderate parameters, where Hagan's
/// approximation does not produce a negative density).
fn strikes() -> Vec<f64> {
    (0..50).map(|i| 60.0 + 2.0 * f64::from(i)).collect()
}

#[test]
fn put_call_parity() {
    let df = (-R * ttm_of(DAYS)).exp();
    for (name, model) in models() {
        for k in strikes() {
            let lhs = price(&model, k, OptionType::Call) - price(&model, k, OptionType::Put);
            let rhs = df * (100.0 - k);
            assert!((lhs - rhs).abs() < 1e-12, "{name} K={k}: {lhs} vs {rhs}");
        }
    }
}

#[test]
fn calls_are_decreasing_and_convex_in_strike() {
    for (name, model) in models() {
        let calls: Vec<f64> = strikes()
            .into_iter()
            .map(|k| price(&model, k, OptionType::Call))
            .collect();
        for w in calls.windows(2) {
            assert!(w[1] < w[0], "{name}: not decreasing {w:?}");
        }
        for w in calls.windows(3) {
            let butterfly = w[0] - 2.0 * w[1] + w[2];
            assert!(
                butterfly > 0.0,
                "{name}: negative butterfly {butterfly} at {w:?}"
            );
        }
    }
}

/// (name, shifted model, base model priced on the shifted F and K, total shift).
fn shifted_models() -> Vec<(&'static str, SabrModel, SabrModel, f64)> {
    let general = SabrModel::general(0.04, 0.5, -0.3, 0.4, R, None).unwrap();
    let lognormal = SabrModel::lognormal(0.04, -0.3, 0.4, R, None).unwrap();
    let normal = SabrModel::normal(0.004, -0.3, 0.4, R, None).unwrap();
    let shifted = |base: &SabrModel, s| SabrModel::shifted(base.clone(), s).unwrap();
    vec![
        ("shifted general", shifted(&general, 0.03), general, 0.03),
        (
            "shifted lognormal",
            shifted(&lognormal, 0.03),
            lognormal.clone(),
            0.03,
        ),
        ("shifted normal", shifted(&normal, 0.03), normal, 0.03),
        (
            "nested shifted lognormal",
            shifted(&shifted(&lognormal, 0.01), 0.02),
            lognormal,
            0.03,
        ),
    ]
}

#[test]
fn shifted_sabr_prices_on_shifted_forward_and_strike() {
    let (f, k) = (0.01, 0.02);
    for (name, model, base, s) in shifted_models() {
        let sigma = model.implied_volatility(f, k, ttm_of(DAYS)).unwrap();
        for option_type in [OptionType::Call, OptionType::Put] {
            let actual = price_at(&model, f, k, option_type);
            // Black-76 for lognormal-type bases; the base model's own Bachelier path for Normal.
            let expected = if base.beta() == 0.0 {
                price_at(&base, f + s, k + s, option_type)
            } else {
                AnalyticalEngine
                    .calculate_price(
                        &forward_option(f + s, k + s, DAYS, option_type),
                        &BlackModel::new(sigma, R).unwrap(),
                    )
                    .unwrap()
            };
            assert_close(actual, expected, 1e-12, name);
        }
    }
}

/// `EuropeanOption` rejects negative F and K, so F = 0 and K = 0 are the
/// non-positive inputs reachable through the engine.
#[test]
fn shifted_sabr_prices_zero_forward_and_strike_with_put_call_parity() {
    let df = (-R * ttm_of(DAYS)).exp();
    for (name, model, ..) in shifted_models() {
        for (f, k) in [(0.0, 0.01), (0.01, 0.0)] {
            let call = price_at(&model, f, k, OptionType::Call);
            let put = price_at(&model, f, k, OptionType::Put);
            assert!(call.is_finite() && call > 0.0, "{name}: call {call}");
            assert_close(call - put, df * (f - k), 1e-12, name);
        }
    }
}
