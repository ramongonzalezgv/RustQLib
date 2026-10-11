mod common;

use common::{american_option, assert_close, spot_option};
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::{AnalyticalEngine, BinomialEngine};
use quantlib_models::BlackScholesModel;

fn model() -> BlackScholesModel {
    BlackScholesModel::new(0.25, 0.05, 0.02).expect("model")
}

#[test]
fn european_converges_to_black_scholes() {
    let model = model();
    for (k, option_type) in [
        (90.0, OptionType::Call),
        (100.0, OptionType::Call),
        (110.0, OptionType::Put),
    ] {
        let option = spot_option(100.0, k, 365, option_type);
        let analytic = AnalyticalEngine.calculate_price(&option, &model).unwrap();
        let err = |n: usize| {
            let tree = BinomialEngine::new(n)
                .calculate_price(&option, &model)
                .unwrap();
            (tree - analytic).abs()
        };

        assert_close(
            BinomialEngine::new(2000)
                .calculate_price(&option, &model)
                .unwrap(),
            analytic,
            1e-2 / 2.0,
            "CRR N=2000 vs BS",
        );
        // CRR error is O(1/N) with odd/even oscillation: compare envelopes of
        // consecutive step counts so the zig-zag cannot fake a failure.
        let coarse = err(50).max(err(51));
        let fine = err(800).max(err(801));
        assert!(
            fine < coarse / 4.0,
            "no convergence for K={k}: err(50/51)={coarse:e}, err(800/801)={fine:e}"
        );
    }
}

#[test]
fn american_is_worth_at_least_european() {
    let model = model();
    let engine = BinomialEngine::new(500);
    for k in [80.0, 100.0, 120.0] {
        for option_type in [OptionType::Call, OptionType::Put] {
            let european = engine
                .calculate_price(&spot_option(100.0, k, 365, option_type), &model)
                .unwrap();
            let american = engine
                .calculate_price(&american_option(100.0, k, 365, option_type), &model)
                .unwrap();
            assert!(
                american >= european - 1e-12,
                "K={k} {option_type:?}: american={american} < european={european}"
            );
        }
    }
}

#[test]
fn american_call_without_dividends_equals_european() {
    // Merton: early exercise of a call is never optimal when q = 0.
    let model = BlackScholesModel::new(0.25, 0.05, 0.0).expect("model");
    let engine = BinomialEngine::new(500);
    for k in [80.0, 100.0, 120.0] {
        let european = engine
            .calculate_price(&spot_option(100.0, k, 365, OptionType::Call), &model)
            .unwrap();
        let american = engine
            .calculate_price(&american_option(100.0, k, 365, OptionType::Call), &model)
            .unwrap();
        assert_close(american, european, 1e-10, "American call, q=0");
    }
}

#[test]
fn american_put_dominates_intrinsic_and_has_early_exercise_premium() {
    let model = BlackScholesModel::new(0.25, 0.08, 0.0).expect("model");
    let engine = BinomialEngine::new(500);
    for k in [90.0, 100.0, 130.0] {
        let american = engine
            .calculate_price(&american_option(100.0, k, 365, OptionType::Put), &model)
            .unwrap();
        let european = engine
            .calculate_price(&spot_option(100.0, k, 365, OptionType::Put), &model)
            .unwrap();
        assert!(
            american >= (k - 100.0f64).max(0.0),
            "K={k}: below intrinsic"
        );
        assert!(
            american > european + 1e-3,
            "K={k}: expected an early-exercise premium, american={american}, european={european}"
        );
    }
}

#[test]
fn american_put_matches_published_reference() {
    // Hull, "Options, Futures and Other Derivatives" (American put, S=50,
    // K=50, r=10%, sigma=40%, T=5 months): the converged binomial value is
    // 4.2842 (Hull quotes 4.29 for N=500). T = 152/365 ~ 5 months here,
    // so allow a looser tolerance for the day-count difference.
    let model = BlackScholesModel::new(0.40, 0.10, 0.0).expect("model");
    let american = BinomialEngine::new(2000)
        .calculate_price(&american_option(50.0, 50.0, 152, OptionType::Put), &model)
        .unwrap();
    assert_close(american, 4.2842, 1e-2, "Hull American put");
}

#[test]
fn cache_does_not_mix_european_and_american_prices() {
    // Regression: both exercise styles used to share one cache key, so pricing
    // the European first made the American return the European price.
    let model = BlackScholesModel::new(0.25, 0.08, 0.0).expect("model");
    let engine = BinomialEngine::new(200);
    let european = engine
        .calculate_price(&spot_option(100.0, 110.0, 365, OptionType::Put), &model)
        .unwrap();
    let american = engine
        .calculate_price(&american_option(100.0, 110.0, 365, OptionType::Put), &model)
        .unwrap();
    let american_fresh = BinomialEngine::new(200)
        .calculate_price(&american_option(100.0, 110.0, 365, OptionType::Put), &model)
        .unwrap();
    assert_eq!(american, american_fresh);
    assert!(american > european);
}
