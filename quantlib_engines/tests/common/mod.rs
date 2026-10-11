//! Shared helpers for the engine integration tests.
#![allow(dead_code)]

use quantlib_core::{OptionType, SimpleDate};
use quantlib_products::{AmericanOption, EuropeanOption};

/// Fixed valuation date. With no calendar, `ttm = days / 365`.
pub fn valuation() -> SimpleDate {
    SimpleDate::new(2026, 1, 1).expect("valid date")
}

pub fn ttm_of(days: i64) -> f64 {
    days as f64 / 365.0
}

pub fn spot_option(s: f64, k: f64, days: i64, option_type: OptionType) -> EuropeanOption {
    let valuation = valuation();
    EuropeanOption::new_spot(
        s,
        k,
        valuation.add_days(days),
        Some(valuation),
        option_type,
        1,
        None,
    )
    .expect("valid spot option")
}

pub fn forward_option(f: f64, k: f64, days: i64, option_type: OptionType) -> EuropeanOption {
    let valuation = valuation();
    EuropeanOption::new_forward(
        f,
        k,
        valuation.add_days(days),
        Some(valuation),
        option_type,
        1,
        None,
    )
    .expect("valid forward option")
}

pub fn american_option(s: f64, k: f64, days: i64, option_type: OptionType) -> AmericanOption {
    let valuation = valuation();
    AmericanOption::new_spot(
        s,
        k,
        valuation.add_days(days),
        Some(valuation),
        option_type,
        1,
        None,
    )
    .expect("valid American option")
}

pub fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{what}: actual={actual}, expected={expected}, |diff|={diff:e} > tol={tol:e}"
    );
}

/// Mean and standard error of independent estimates (e.g. MC prices from
/// different seeds). The standard error is that of the mean of `samples`.
pub fn mean_and_stderr(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    assert!(n >= 2.0, "need at least two samples");
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}
