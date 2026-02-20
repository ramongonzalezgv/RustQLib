use std::path::PathBuf;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{FFTEngine, InterpolationKind};
use quantlib_models::HestonModel;
use quantlib_products::EuropeanOption;

#[test]
fn fft_heston_engine_matches_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/fft_heston.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 19 {
            continue;
        }

        let name = fields[0];
        let s = parse_f64(fields[1]);
        let k = parse_f64(fields[2]);
        let expiry = parse_date(fields[3]);
        let valuation = parse_date(fields[4]);
        let option_type = fields[5].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[6]);
        let kappa = parse_f64(fields[7]);
        let theta = parse_f64(fields[8]);
        let sigma = parse_f64(fields[9]);
        let rho = parse_f64(fields[10]);
        let v0 = parse_f64(fields[11]);
        let r = parse_f64(fields[12]);
        let q = parse_f64(fields[13]);
        let n = parse_usize(fields[14]);
        let b = parse_f64(fields[15]);
        let interp = fields[16]
            .parse::<InterpolationKind>()
            .expect("interpolation kind");
        let ttm = parse_f64(fields[17]);
        let expected_price = parse_f64(fields[18]);

        let product =
            EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                .expect("valid option");
        let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).expect("heston model");
        let engine = FFTEngine::new(n, b, interp).expect("fft engine");

        let rust_price = engine
            .calculate_price(&product, &model)
            .expect("fft pricing should succeed");

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        assert!(
            (product.ttm() - ttm).abs() < 1.0e-12,
            "{name}: ttm mismatch. rust={}, fixture={ttm}",
            product.ttm()
        );

        assert!(
            abs_diff < 1.0e-7 || rel_diff < 1.0e-7,
            "{name}: fft heston mismatch. rust={rust_price}, fixture={expected_price}, abs={abs_diff}, rel={rel_diff}"
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

fn parse_usize(value: &str) -> usize {
    value.parse::<usize>().expect("valid usize")
}
