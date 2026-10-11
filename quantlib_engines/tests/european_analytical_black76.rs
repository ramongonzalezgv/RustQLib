mod common;

use common::{assert_close, forward_option, spot_option, ttm_of};
use proptest::prelude::*;
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::{BlackModel, BlackScholesModel};

fn black_price(f: f64, k: f64, days: i64, option_type: OptionType, model: &BlackModel) -> f64 {
    AnalyticalEngine
        .calculate_price(&forward_option(f, k, days, option_type), model)
        .expect("Black-76 price")
}

#[test]
fn matches_published_reference_value() {
    // Hull, "Options, Futures and Other Derivatives", Black's model example:
    // F=K=20, r=9%, sigma=25%, T=4/12 gives put = 1.1166.
    // With an integer day count T=122/365 the exact value (independent
    // closed form via Python `math.erf`) is 1.118075498308455.
    let model = BlackModel::new(0.25, 0.09).expect("model");
    assert_close(
        black_price(20.0, 20.0, 122, OptionType::Put, &model),
        1.118_075_498_308_455,
        1e-9,
        "Black-76 ATM put",
    );
}

fn arb_inputs() -> impl Strategy<Value = (f64, f64, i64, f64, f64)> {
    (
        50.0..150.0f64, // forward
        50.0..150.0f64, // strike
        7i64..1500,     // days to expiry
        0.05..0.8f64,   // sigma
        0.0..0.08f64,   // r
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn put_call_parity((f, k, days, sigma, r) in arb_inputs()) {
        let model = BlackModel::new(sigma, r).unwrap();
        let df = (-r * ttm_of(days)).exp();
        let c = black_price(f, k, days, OptionType::Call, &model);
        let p = black_price(f, k, days, OptionType::Put, &model);
        prop_assert!((c - p - df * (f - k)).abs() < 1e-9 * f);
    }

    #[test]
    fn no_arbitrage_bounds((f, k, days, sigma, r) in arb_inputs()) {
        let model = BlackModel::new(sigma, r).unwrap();
        let df = (-r * ttm_of(days)).exp();
        let tol = 1e-10 * f;
        let c = black_price(f, k, days, OptionType::Call, &model);
        let p = black_price(f, k, days, OptionType::Put, &model);
        prop_assert!(c >= df * (f - k).max(0.0) - tol && c <= df * f + tol);
        prop_assert!(p >= df * (k - f).max(0.0) - tol && p <= df * k + tol);
    }

    /// Black-76 on F = S e^{(r-q)T} must equal Black–Scholes on S.
    #[test]
    fn equals_black_scholes_on_forward(
        (s, k, days, sigma, r) in arb_inputs(),
        q in 0.0..0.05f64,
    ) {
        let t = ttm_of(days);
        let f = s * ((r - q) * t).exp();
        let black = BlackModel::new(sigma, r).unwrap();
        let bs = BlackScholesModel::new(sigma, r, q).unwrap();
        for option_type in [OptionType::Call, OptionType::Put] {
            let p76 = black_price(f, k, days, option_type, &black);
            let pbs = AnalyticalEngine
                .calculate_price(&spot_option(s, k, days, option_type), &bs)
                .unwrap();
            prop_assert!((p76 - pbs).abs() < 1e-9 * s, "black={p76} bs={pbs}");
        }
    }
}
