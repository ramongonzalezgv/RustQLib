mod common;

use common::{assert_close, spot_option, ttm_of};
use proptest::prelude::*;
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;

fn price(s: f64, k: f64, days: i64, option_type: OptionType, model: &BlackScholesModel) -> f64 {
    AnalyticalEngine
        .calculate_price(&spot_option(s, k, days, option_type), model)
        .expect("analytical price")
}

#[test]
fn matches_published_reference_values() {
    // Textbook case S=K=100, T=1, r=5%, sigma=20%, q=0.
    // Call 10.450583572185565, put 5.573526022256971 (e.g. Hull, "Options,
    // Futures and Other Derivatives", and any closed-form BS calculator).
    let model = BlackScholesModel::new(0.2, 0.05, 0.0).expect("model");
    assert_close(
        price(100.0, 100.0, 365, OptionType::Call, &model),
        10.450_583_572_185_565,
        1e-8,
        "BS call",
    );
    assert_close(
        price(100.0, 100.0, 365, OptionType::Put, &model),
        5.573_526_022_256_971,
        1e-8,
        "BS put",
    );
}

#[test]
fn zero_vol_limit_is_discounted_forward_intrinsic() {
    let (s, r, q, days) = (100.0, 0.05, 0.02, 365);
    let t = ttm_of(days);
    let model = BlackScholesModel::new(1e-8, r, q).expect("model");
    for k in [80.0, 95.0, 120.0] {
        let fwd_intrinsic = (s * (-q * t).exp() - k * (-r * t).exp()).max(0.0);
        assert_close(
            price(s, k, days, OptionType::Call, &model),
            fwd_intrinsic,
            1e-8,
            "sigma->0 call",
        );
    }
}

fn arb_inputs() -> impl Strategy<Value = (f64, f64, i64, f64, f64, f64)> {
    (
        50.0..150.0f64, // spot
        50.0..150.0f64, // strike
        7i64..1500,     // days to expiry
        0.05..0.8f64,   // sigma
        0.0..0.08f64,   // r
        0.0..0.05f64,   // q
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn put_call_parity((s, k, days, sigma, r, q) in arb_inputs()) {
        let model = BlackScholesModel::new(sigma, r, q).unwrap();
        let t = ttm_of(days);
        let c = price(s, k, days, OptionType::Call, &model);
        let p = price(s, k, days, OptionType::Put, &model);
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        prop_assert!((c - p - rhs).abs() < 1e-9 * s, "c-p={} rhs={}", c - p, rhs);
    }

    #[test]
    fn no_arbitrage_bounds((s, k, days, sigma, r, q) in arb_inputs()) {
        let model = BlackScholesModel::new(sigma, r, q).unwrap();
        let t = ttm_of(days);
        let (ds, dk) = (s * (-q * t).exp(), k * (-r * t).exp());
        let tol = 1e-10 * s;
        let c = price(s, k, days, OptionType::Call, &model);
        let p = price(s, k, days, OptionType::Put, &model);
        prop_assert!(c >= (ds - dk).max(0.0) - tol && c <= ds + tol, "call {c} outside bounds");
        prop_assert!(p >= (dk - ds).max(0.0) - tol && p <= dk + tol, "put {p} outside bounds");
    }

    #[test]
    fn monotone_in_spot_strike_and_vol((s, k, days, sigma, r, q) in arb_inputs()) {
        let model = BlackScholesModel::new(sigma, r, q).unwrap();
        let higher_vol = BlackScholesModel::new(sigma * 1.1, r, q).unwrap();
        let tol = 1e-10 * s;
        let c = price(s, k, days, OptionType::Call, &model);
        let p = price(s, k, days, OptionType::Put, &model);

        prop_assert!(price(s * 1.01, k, days, OptionType::Call, &model) >= c - tol);
        prop_assert!(price(s * 1.01, k, days, OptionType::Put, &model) <= p + tol);
        prop_assert!(price(s, k * 1.01, days, OptionType::Call, &model) <= c + tol);
        prop_assert!(price(s, k * 1.01, days, OptionType::Put, &model) >= p - tol);
        prop_assert!(price(s, k, days, OptionType::Call, &higher_vol) >= c - tol);
        prop_assert!(price(s, k, days, OptionType::Put, &higher_vol) >= p - tol);
    }

    #[test]
    fn convex_in_strike((s, k, days, sigma, r, q) in arb_inputs()) {
        let model = BlackScholesModel::new(sigma, r, q).unwrap();
        let h = 1.0;
        let butterfly = price(s, k - h, days, OptionType::Call, &model)
            - 2.0 * price(s, k, days, OptionType::Call, &model)
            + price(s, k + h, days, OptionType::Call, &model);
        prop_assert!(butterfly >= -1e-9 * s, "butterfly={butterfly}");
    }
}

#[test]
fn deep_otm_put_is_positive_and_accurate() {
    // S=100, K=20, sigma=20%, r=5%, q=0, T=183/365. mpmath (50 digits):
    // 2.1603814591587665e-31. With 1 + erf(x) this used to round to 0.
    let model = BlackScholesModel::new(0.2, 0.05, 0.0).expect("model");
    let put = price(100.0, 20.0, 183, OptionType::Put, &model);
    let expected = 2.160_381_459_158_766_5e-31;
    assert!(put > 0.0);
    assert!(
        ((put - expected) / expected).abs() < 1e-10,
        "put={put}, expected={expected}"
    );
}
