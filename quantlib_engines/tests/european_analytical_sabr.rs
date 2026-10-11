mod common;

use common::{forward_option, ttm_of};
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::SabrModel;

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

fn price(model: &SabrModel, k: f64, option_type: OptionType) -> f64 {
    AnalyticalEngine
        .calculate_price(&forward_option(100.0, k, DAYS, option_type), model)
        .expect("SABR price")
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
