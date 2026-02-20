use std::path::PathBuf;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::{BlackModel, SabrModel};
use quantlib_products::EuropeanOption;

#[test]
fn analytical_engine_matches_black_model_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/european_analytical_black_model.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    let engine = AnalyticalEngine;
    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 12 {
            continue;
        }

        let name = fields[0];
        let _s = parse_f64(fields[1]);
        let k = parse_f64(fields[2]);
        let f = parse_f64(fields[3]);
        let expiry = parse_date(fields[4]);
        let valuation = parse_date(fields[5]);
        let option_type = fields[6].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[7]);
        let sigma = parse_f64(fields[8]);
        let r = parse_f64(fields[9]);
        let ttm = parse_f64(fields[10]);
        let expected_price = parse_f64(fields[11]);

        let product =
            EuropeanOption::new_forward(f, k, expiry, Some(valuation), option_type, qty, None)
                .expect("valid product");
        let model = BlackModel::new(sigma, r).expect("model");

        let rust_price = engine
            .calculate_price(&product, &model)
            .expect("pricing should succeed");

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        assert!(
            (product.ttm() - ttm).abs() < 1.0e-12,
            "{name}: ttm mismatch. rust={}, fixture={ttm}",
            product.ttm()
        );
        assert!(
            abs_diff < 1.0e-6 || rel_diff < 1.0e-6,
            "{name}: price mismatch. rust={rust_price}, fixture={expected_price}, abs_diff={abs_diff}, rel_diff={rel_diff}"
        );
    }
}

#[test]
fn analytical_engine_matches_sabr_model_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/european_analytical_sabr_model.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    let engine = AnalyticalEngine;
    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 17 {
            continue;
        }

        let name = fields[0];
        let model_type = fields[1];
        let _s = parse_f64(fields[2]);
        let k = parse_f64(fields[3]);
        let f = parse_f64(fields[4]);
        let expiry = parse_date(fields[5]);
        let valuation = parse_date(fields[6]);
        let option_type = fields[7].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[8]);
        let alpha = parse_f64(fields[9]);
        let beta = parse_optional_f64(fields[10]);
        let rho = parse_f64(fields[11]);
        let nu = parse_f64(fields[12]);
        let r = parse_f64(fields[13]);
        let shift = parse_optional_f64(fields[14]);
        let ttm = parse_f64(fields[15]);
        let expected_price = parse_f64(fields[16]);

        let product =
            EuropeanOption::new_forward(f, k, expiry, Some(valuation), option_type, qty, None)
                .expect("valid product");

        let model = match model_type {
            "normal" => SabrModel::normal(alpha, rho, nu, r, None).expect("normal sabr"),
            "lognormal" => SabrModel::lognormal(alpha, rho, nu, r, None).expect("lognormal sabr"),
            "general" => SabrModel::general(
                alpha,
                beta.expect("general sabr requires beta"),
                rho,
                nu,
                r,
                None,
            )
            .expect("general sabr"),
            "shifted_lognormal" => {
                let base = SabrModel::lognormal(alpha, rho, nu, r, None).expect("base sabr");
                SabrModel::shifted(base, shift.expect("shift is required")).expect("shifted sabr")
            }
            _ => panic!("unknown SABR model type: {model_type}"),
        };

        let rust_price = engine
            .calculate_price(&product, &model)
            .expect("pricing should succeed");

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        assert!(
            (product.ttm() - ttm).abs() < 1.0e-12,
            "{name}: ttm mismatch. rust={}, fixture={ttm}",
            product.ttm()
        );
        assert!(
            abs_diff < 1.0e-6 || rel_diff < 1.0e-6,
            "{name}: price mismatch. rust={rust_price}, fixture={expected_price}, abs_diff={abs_diff}, rel_diff={rel_diff}"
        );
    }
}

fn parse_date(value: &str) -> SimpleDate {
    SimpleDate::parse(value).expect("date in YYYY-MM-DD")
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().expect("valid f64")
}

fn parse_i32(value: &str) -> i32 {
    value.parse::<i32>().expect("valid i32")
}

fn parse_optional_f64(value: &str) -> Option<f64> {
    if value.trim().is_empty() {
        None
    } else {
        Some(parse_f64(value))
    }
}
