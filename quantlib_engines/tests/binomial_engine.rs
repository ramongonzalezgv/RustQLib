mod common;

use common::{american_option, assert_close, spot_option};
use std::sync::Arc;

use quantlib_core::{MarketCalendar, OptionType, PricingEngine, SimpleDate, WeekdayCalendar};
use quantlib_engines::{AnalyticalEngine, BinomialEngine};
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

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
fn one_engine_prices_european_and_american_independently() {
    // Regression: a cache inside the engine once keyed both exercise styles the
    // same, so pricing the European first made the American return its price.
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

#[test]
fn one_engine_reprices_options_that_differ_only_by_valuation_date_or_calendar() {
    // Regression for #2: the engine's old cache key omitted the valuation date
    // and the calendar, so these all returned the first option's price.
    let model = model();
    let engine = BinomialEngine::new(200);
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");
    let option = |valuation: (i32, u32, u32), weekday_calendar: bool| {
        let valuation = SimpleDate::new(valuation.0, valuation.1, valuation.2).expect("date");
        let calendar = weekday_calendar
            .then(|| Arc::new(WeekdayCalendar::default()) as Arc<dyn MarketCalendar>);
        EuropeanOption::new_spot(
            100.0,
            100.0,
            expiry,
            Some(valuation),
            OptionType::Call,
            1,
            calendar,
        )
        .expect("option")
    };

    let early = engine
        .calculate_price(&option((2026, 2, 10), false), &model)
        .unwrap();
    let late = engine
        .calculate_price(&option((2026, 8, 10), false), &model)
        .unwrap();
    let weekday = engine
        .calculate_price(&option((2026, 2, 10), true), &model)
        .unwrap();

    assert!(early > late, "early={early}, late={late}");
    assert_ne!(early, weekday, "calendar must change the price");
    assert_eq!(
        early,
        BinomialEngine::new(200)
            .calculate_price(&option((2026, 2, 10), false), &model)
            .unwrap()
    );
}
