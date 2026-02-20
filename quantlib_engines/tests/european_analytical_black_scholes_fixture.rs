use std::path::PathBuf;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

#[test]
fn analytical_engine_matches_python_fixture() {
    run_fixture("../../QuantLibPY/tests/fixtures/european_analytical_black_scholes.csv");
}

#[test]
fn analytical_engine_matches_edge_case_fixture() {
    run_fixture("../../QuantLibPY/tests/fixtures/european_analytical_black_scholes_edge_cases.csv");
}

fn run_fixture(relative_fixture_path: &str) {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_fixture_path);
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    let engine = AnalyticalEngine;
    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 12 {
            continue;
        }

        let name = fields[0];
        let s = parse_f64(fields[1]);
        let k = parse_f64(fields[2]);
        let expiry = parse_date(fields[3]);
        let valuation = parse_date(fields[4]);
        let option_type = fields[5].parse::<OptionType>().expect("option type");
        let qty = fields[6].parse::<i32>().expect("qty");
        let sigma = parse_f64(fields[7]);
        let r = parse_f64(fields[8]);
        let q = parse_f64(fields[9]);
        let ttm = parse_f64(fields[10]);
        let expected_price = parse_f64(fields[11]);

        let product =
            EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                .expect("valid product");
        let model = BlackScholesModel::new(sigma, r, q).expect("model");

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
fn put_call_parity_holds() {
    let expiry = SimpleDate::new(2027, 2, 10).expect("valid date");
    let valuation = SimpleDate::new(2026, 2, 10).expect("valid date");

    let call = EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("call option");

    let put = EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Put,
        1,
        None,
    )
    .expect("put option");

    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let engine = AnalyticalEngine;

    let call_price = engine.calculate_price(&call, &model).expect("call price");
    let put_price = engine.calculate_price(&put, &model).expect("put price");

    let t = call.ttm();
    let rhs = call.require_spot().expect("spot quote") * (-(model.q) * t).exp()
        - call.k * (-(model.r) * t).exp();
    let lhs = call_price - put_price;

    assert!((lhs - rhs).abs() < 1.0e-6, "lhs={lhs}, rhs={rhs}");
}

fn parse_date(value: &str) -> SimpleDate {
    SimpleDate::parse(value).expect("date in YYYY-MM-DD")
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().expect("valid f64")
}
