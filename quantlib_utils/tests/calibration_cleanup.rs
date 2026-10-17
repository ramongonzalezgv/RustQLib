//! Acceptance tests for #52: fixed rates, rejected Volatility target, one closed form, FFT default.

use quantlib_core::{OptionType, PricingEngine, QuantLibError, SimpleDate};
use quantlib_engines::{AnalyticalEngine, CosEngine, FFTEngine, InterpolationKind};
use quantlib_models::{BlackModel, BlackScholesModel, HestonModel};
use quantlib_products::EuropeanOption;
use quantlib_utils::{
    BLACK_PARAMETER_NAMES, BLACK_SCHOLES_PARAMETER_NAMES, CalibrationConfig, CalibrationTarget,
    HESTON_PARAMETER_NAMES, HestonFftConfig, NelderMeadConfig, VanillaOptionCalibrationQuote,
    calibrate_black_model, calibrate_black_scholes_model, calibrate_heston_model,
    default_black_bounds, default_black_scholes_bounds, default_heston_bounds,
};

const DAYS: [i64; 4] = [90, 180, 365, 540];
const STRIKES: [f64; 5] = [85.0, 95.0, 100.0, 105.0, 115.0];
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];

fn valuation() -> SimpleDate {
    SimpleDate::new(2026, 1, 1).expect("date")
}

fn spot_option(s: f64, k: f64, days: i64, ty: OptionType) -> EuropeanOption {
    let v = valuation();
    EuropeanOption::new_spot(s, k, v.add_days(days), Some(v), ty, 1, None).expect("option")
}

fn forward_option(f: f64, k: f64, days: i64, ty: OptionType) -> EuropeanOption {
    let v = valuation();
    EuropeanOption::new_forward(f, k, v.add_days(days), Some(v), ty, 1, None).expect("option")
}

/// Quotes at `days / 365` (the engine's day count without a calendar), priced by `price`.
fn quotes_from(price: impl Fn(f64, i64, OptionType) -> f64) -> Vec<VanillaOptionCalibrationQuote> {
    let mut quotes = Vec::new();
    for days in DAYS {
        for k in STRIKES {
            for ty in TYPES {
                quotes.push(VanillaOptionCalibrationQuote::price(
                    100.0,
                    k,
                    days as f64 / 365.0,
                    ty,
                    price(k, days, ty),
                    1.0,
                ));
            }
        }
    }
    quotes
}

const SIGMA: f64 = 0.25;
const R: f64 = 0.03;
const Q: f64 = 0.01;

fn bs_quotes() -> Vec<VanillaOptionCalibrationQuote> {
    let truth = BlackScholesModel::new(SIGMA, R, Q).expect("model");
    quotes_from(|k, days, ty| {
        AnalyticalEngine
            .calculate_price(&spot_option(100.0, k, days, ty), &truth)
            .expect("price")
    })
}

fn black_quotes() -> Vec<VanillaOptionCalibrationQuote> {
    let truth = BlackModel::new(SIGMA, R).expect("model");
    quotes_from(|k, days, ty| {
        AnalyticalEngine
            .calculate_price(&forward_option(100.0, k, days, ty), &truth)
            .expect("price")
    })
}

fn vol_quotes() -> Vec<VanillaOptionCalibrationQuote> {
    vec![VanillaOptionCalibrationQuote::implied_volatility(
        100.0,
        100.0,
        1.0,
        OptionType::Call,
        0.2,
        1.0,
    )]
}

fn assert_validation(err: QuantLibError) -> String {
    match err {
        QuantLibError::Validation(msg) => msg,
        other => panic!("expected Validation, got {other:?}"),
    }
}

#[test]
fn a1_volatility_target_rejected_for_black_and_black_scholes() {
    let cfg = CalibrationConfig::default();
    let quotes = vol_quotes();
    let black = calibrate_black_model(&quotes, CalibrationTarget::Volatility, 0.2, R, None, &cfg)
        .expect_err("black");
    let bs = calibrate_black_scholes_model(
        &quotes,
        CalibrationTarget::Volatility,
        0.2,
        R,
        Q,
        None,
        &cfg,
    )
    .expect_err("bs");
    for err in [black, bs] {
        assert!(assert_validation(err).contains("#19"));
    }
}

#[test]
fn non_finite_rates_are_validation_errors() {
    let cfg = CalibrationConfig::default();
    let (bq, sq) = (black_quotes(), bs_quotes());
    for r in [f64::NAN, f64::INFINITY] {
        let e = calibrate_black_model(&bq, CalibrationTarget::Price, 0.2, r, None, &cfg)
            .expect_err("black r");
        assert_validation(e);
    }
    for (r, q) in [(f64::NAN, Q), (R, f64::INFINITY), (R, f64::NAN)] {
        let e = calibrate_black_scholes_model(&sq, CalibrationTarget::Price, 0.2, r, q, None, &cfg)
            .expect_err("bs r/q");
        assert_validation(e);
    }
    for (r, q) in [(f64::NAN, Q), (R, f64::INFINITY)] {
        let init = HestonModel {
            kappa: 1.5,
            theta: 0.04,
            sigma: 0.4,
            rho: -0.5,
            v0: 0.04,
            r,
            q,
        };
        let e = calibrate_heston_model(&sq, init, None, &cfg, HestonFftConfig::default())
            .expect_err("heston r/q");
        assert_validation(e);
    }
}

#[test]
fn a2_black_scholes_round_trip_keeps_r_q_fixed() {
    let (model, result) = calibrate_black_scholes_model(
        &bs_quotes(),
        CalibrationTarget::Price,
        0.18,
        R,
        Q,
        None,
        &CalibrationConfig::default(),
    )
    .expect("calibration");
    assert!((model.sigma - SIGMA).abs() < 1e-4, "sigma {}", model.sigma);
    assert_eq!((model.r, model.q), (R, Q));
    let keys: Vec<_> = result
        .calibrated_params
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["sigma"]);
}

#[test]
fn a2_black_round_trip_keeps_r_fixed() {
    let (model, result) = calibrate_black_model(
        &black_quotes(),
        CalibrationTarget::Price,
        0.18,
        R,
        None,
        &CalibrationConfig::default(),
    )
    .expect("calibration");
    assert!((model.sigma - SIGMA).abs() < 1e-4, "sigma {}", model.sigma);
    assert_eq!(model.r, R);
    let keys: Vec<_> = result
        .calibrated_params
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["sigma"]);
}

#[test]
fn a2_heston_keeps_r_q_of_initial_model() {
    let truth = HestonModel::new(1.8, 0.05, 0.45, -0.65, 0.04, 0.02, 0.0).expect("model");
    let fft = HestonFftConfig {
        n: 1 << 9,
        b: 120.0,
        interp: InterpolationKind::Cubic,
    };
    let engine = FFTEngine::new(fft.n, fft.b, fft.interp).expect("fft");
    let mut quotes = Vec::new();
    for k in [90.0, 100.0, 110.0] {
        for ty in TYPES {
            let p = engine
                .price_strip(&truth, 100.0, 0.5, &[k], &[ty])
                .expect("price")[0];
            quotes.push(VanillaOptionCalibrationQuote::price(
                100.0, k, 0.5, ty, p, 1.0,
            ));
        }
    }
    let initial = HestonModel::new(1.5, 0.05, 0.6, -0.3, 0.06, 0.0314, 0.0123).expect("initial");
    let config = CalibrationConfig {
        method: quantlib_utils::CalibrationMethod::NelderMead,
        nelder_mead: NelderMeadConfig {
            max_iterations: 30,
            ..NelderMeadConfig::default()
        },
        ..CalibrationConfig::default()
    };
    let (model, result) = calibrate_heston_model(
        &quotes,
        initial,
        Some(default_heston_bounds()),
        &config,
        fft,
    )
    .expect("calibration");
    assert_eq!((model.r, model.q), (initial.r, initial.q));
    let mut keys: Vec<_> = result
        .calibrated_params
        .keys()
        .map(String::as_str)
        .collect();
    let mut want = HESTON_PARAMETER_NAMES.to_vec();
    keys.sort_unstable();
    want.sort_unstable();
    assert_eq!(keys, want);
}

#[test]
fn api_names_and_default_bounds() {
    assert_eq!(BLACK_PARAMETER_NAMES, ["sigma"]);
    assert_eq!(BLACK_SCHOLES_PARAMETER_NAMES, ["sigma"]);
    assert_eq!(
        HESTON_PARAMETER_NAMES,
        ["kappa", "theta", "sigma", "rho", "v0"]
    );
    assert_eq!(default_black_bounds(), [(1e-4, 3.0)]);
    assert_eq!(default_black_scholes_bounds(), [(1e-4, 3.0)]);
    assert_eq!(default_heston_bounds().len(), 5);
}

#[test]
fn a4_calibration_prices_equal_analytical_engine() {
    let cfg = CalibrationConfig::default();
    let (bs, bs_result) = calibrate_black_scholes_model(
        &bs_quotes(),
        CalibrationTarget::Price,
        0.18,
        R,
        Q,
        None,
        &cfg,
    )
    .expect("bs");
    let (black, black_result) = calibrate_black_model(
        &black_quotes(),
        CalibrationTarget::Price,
        0.18,
        R,
        None,
        &cfg,
    )
    .expect("black");
    let (mut i, mut n) = (0, 0);
    for days in DAYS {
        for k in STRIKES {
            for ty in TYPES {
                let want_bs = AnalyticalEngine
                    .calculate_price(&spot_option(100.0, k, days, ty), &bs)
                    .expect("price");
                let want_black = AnalyticalEngine
                    .calculate_price(&forward_option(100.0, k, days, ty), &black)
                    .expect("price");
                assert!(
                    (bs_result.model_values[i] - want_bs).abs() < 1e-12,
                    "bs {k} {days}"
                );
                assert!(
                    (black_result.model_values[i] - want_black).abs() < 1e-12,
                    "black {k} {days}"
                );
                i += 1;
                n += 1;
            }
        }
    }
    assert_eq!((n, bs_result.model_values.len()), (40, 40));
}

#[test]
fn a4_calibration_source_has_no_pricing_formula() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/calibration.rs");
    let src = std::fs::read_to_string(path).expect("source");
    let production = src.split("#[cfg(test)]").next().expect("split");
    for banned in ["norm_cdf", "fn black_price", "fn black_scholes_price"] {
        assert!(
            !production.contains(banned),
            "calibration.rs still contains `{banned}`"
        );
    }
}

/// Domain of Maths item 5 in the spec: max abs call/put price error vs a COS reference, S = 100.
fn max_error(price: impl Fn(&HestonModel, f64, &[f64], &[OptionType]) -> Vec<f64>) -> f64 {
    const SETS: [(f64, f64, f64, f64, f64); 3] = [
        (2.0, 0.04, 0.5, -0.7, 0.04),
        (0.5, 0.09, 1.0, -0.9, 0.09),
        (5.0, 0.02, 0.3, 0.0, 0.01),
    ];
    let strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0];
    let cos = CosEngine::new(1 << 13, 16.0).expect("cos");
    let mut worst = 0.0f64;
    for (kappa, theta, sigma, rho, v0) in SETS {
        let model = HestonModel::new(kappa, theta, sigma, rho, v0, 0.02, 0.01).expect("model");
        for maturity in [0.1, 0.5, 1.0, 2.0] {
            for ty in TYPES {
                let types = [ty; 7];
                let want = cos
                    .price_strip(&model, 100.0, maturity, &strikes, &types)
                    .expect("cos");
                let got = price(&model, maturity, &strikes, &types);
                for (g, w) in got.iter().zip(&want) {
                    worst = worst.max((g - w).abs());
                }
            }
        }
    }
    worst
}

#[test]
fn a5_reference_is_validated_against_large_fft() {
    // COS and FFT(2^16, b = 1600, cubic) are independent methods; measured agreement is ~2e-9.
    let fft = FFTEngine::new(1 << 16, 1600.0, InterpolationKind::Cubic).expect("fft");
    let err = max_error(|m, t, k, ty| fft.price_strip(m, 100.0, t, k, ty).expect("fft"));
    assert!(err <= 1e-7, "reference disagreement {err:e}");
}

#[test]
fn a5_default_heston_fft_grid_meets_price_error_bound() {
    let cfg = HestonFftConfig::default();
    let fft = FFTEngine::new(cfg.n, cfg.b, cfg.interp).expect("fft");
    let err = max_error(|m, t, k, ty| fft.price_strip(m, 100.0, t, k, ty).expect("fft"));
    assert!(
        err <= 1e-2,
        "default grid n={} b={} max error {err:e}",
        cfg.n,
        cfg.b
    );
    assert!(cfg.n.is_power_of_two() && cfg.n <= FFTEngine::DEFAULT_N);
}
