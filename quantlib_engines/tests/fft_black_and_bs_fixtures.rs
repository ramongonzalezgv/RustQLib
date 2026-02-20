use std::path::PathBuf;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{FFTEngine, InterpolationKind};
use quantlib_models::{BlackModel, BlackScholesModel};
use quantlib_products::EuropeanOption;

#[test]
fn fft_engine_matches_black_and_black_scholes_fixtures() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/fft_black_and_bs.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 17 {
            continue;
        }

        let name = fields[0];
        let model_type = fields[1];
        let s = parse_f64(fields[2]);
        let k = parse_f64(fields[3]);
        let f = parse_optional_f64(fields[4]);
        let expiry = parse_date(fields[5]);
        let valuation = parse_date(fields[6]);
        let option_type = fields[7].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[8]);
        let sigma = parse_f64(fields[9]);
        let r = parse_f64(fields[10]);
        let q = parse_f64(fields[11]);
        let n = parse_usize(fields[12]);
        let b = parse_f64(fields[13]);
        let interp = fields[14]
            .parse::<InterpolationKind>()
            .expect("interpolation kind");
        let ttm = parse_f64(fields[15]);
        let expected_price = parse_f64(fields[16]);

        let engine = FFTEngine::new(n, b, interp).expect("fft engine");

        let rust_price = match model_type {
            "black_scholes" => {
                let product =
                    EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                        .expect("spot option");
                let model = BlackScholesModel::new(sigma, r, q).expect("bs model");

                assert!(
                    (product.ttm() - ttm).abs() < 1.0e-12,
                    "{name}: ttm mismatch. rust={}, fixture={ttm}",
                    product.ttm()
                );

                engine
                    .calculate_price(&product, &model)
                    .expect("fft bs pricing should succeed")
            }
            "black" => {
                let forward = f.expect("black model fixture requires forward F");
                let product = EuropeanOption::new_forward(
                    forward,
                    k,
                    expiry,
                    Some(valuation),
                    option_type,
                    qty,
                    None,
                )
                .expect("forward option");
                let model = BlackModel::new(sigma, r).expect("black model");

                assert!(
                    (product.ttm() - ttm).abs() < 1.0e-12,
                    "{name}: ttm mismatch. rust={}, fixture={ttm}",
                    product.ttm()
                );

                engine
                    .calculate_price(&product, &model)
                    .expect("fft black pricing should succeed")
            }
            _ => panic!("{name}: unknown model_type '{model_type}'"),
        };

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        assert!(
            abs_diff < 1.0e-6 || rel_diff < 1.0e-6,
            "{name}: fft mismatch. rust={rust_price}, fixture={expected_price}, abs={abs_diff}, rel={rel_diff}"
        );
    }
}

#[test]
fn fft_black_scholes_rejects_forward_quoted_option() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_forward(
        102.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("forward option");
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("bs model");
    let engine = FFTEngine::default();

    let err = engine
        .calculate_price(&option, &model)
        .expect_err("FFT BS should reject forward-quoted product");
    assert!(
        err.to_string().contains("spot-based pricing requested"),
        "unexpected error: {err}"
    );
}

#[test]
fn fft_black_rejects_spot_quoted_option() {
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
    .expect("spot option");
    let model = BlackModel::new(0.2, 0.03).expect("black model");
    let engine = FFTEngine::default();

    let err = engine
        .calculate_price(&option, &model)
        .expect_err("FFT Black should reject spot-quoted product");
    assert!(
        err.to_string().contains("forward-based pricing requested"),
        "unexpected error: {err}"
    );
}

fn parse_date(value: &str) -> SimpleDate {
    SimpleDate::parse(value).expect("date in YYYY-MM-DD")
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().expect("valid f64")
}

fn parse_optional_f64(value: &str) -> Option<f64> {
    if value.trim().is_empty() {
        None
    } else {
        Some(parse_f64(value))
    }
}

fn parse_i32(value: &str) -> i32 {
    value.parse::<i32>().expect("valid i32")
}

fn parse_usize(value: &str) -> usize {
    value.parse::<usize>().expect("valid usize")
}
