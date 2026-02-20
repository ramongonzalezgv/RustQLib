use std::path::PathBuf;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{BinomialEngine, MonteCarloEngine};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AmericanOption, AsianOption, AveragingType, EuropeanOption};

#[test]
fn binomial_engine_matches_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/binomial_black_scholes.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 13 {
            continue;
        }

        let name = fields[0];
        let product_style = fields[1];
        let s = parse_f64(fields[2]);
        let k = parse_f64(fields[3]);
        let expiry = parse_date(fields[4]);
        let valuation = parse_date(fields[5]);
        let option_type = fields[6].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[7]);
        let sigma = parse_f64(fields[8]);
        let r = parse_f64(fields[9]);
        let q = parse_f64(fields[10]);
        let n_steps = parse_usize(fields[11]);
        let expected_price = parse_f64(fields[12]);

        let model = BlackScholesModel::new(sigma, r, q).expect("model");
        let engine = BinomialEngine::new(n_steps);

        let rust_price = if product_style == "american" {
            let product =
                AmericanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                    .expect("american option");
            engine
                .calculate_price(&product, &model)
                .expect("american price")
        } else {
            let product =
                EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                    .expect("european option");
            engine
                .calculate_price(&product, &model)
                .expect("european price")
        };

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        assert!(
            abs_diff < 1.0e-8 || rel_diff < 1.0e-8,
            "{name}: binomial mismatch. rust={rust_price}, fixture={expected_price}, abs={abs_diff}, rel={rel_diff}"
        );
    }
}

#[test]
fn monte_carlo_engine_matches_deterministic_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/monte_carlo_black_scholes_deterministic.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 16 {
            continue;
        }

        let name = fields[0];
        let product_type = fields[1];
        let averaging_type = fields[2];
        let s = parse_f64(fields[3]);
        let k = parse_f64(fields[4]);
        let expiry = parse_date(fields[5]);
        let valuation = parse_date(fields[6]);
        let option_type = fields[7].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[8]);
        let sigma = parse_f64(fields[9]);
        let r = parse_f64(fields[10]);
        let q = parse_f64(fields[11]);
        let n_paths = parse_usize(fields[12]);
        let n_steps = parse_usize(fields[13]);
        let seed = parse_u64(fields[14]);
        let expected_price = parse_f64(fields[15]);

        let model = BlackScholesModel::new(sigma, r, q).expect("model");
        let engine = MonteCarloEngine::new(n_paths, n_steps, Some(seed)).expect("engine");

        let rust_price = if product_type == "asian" {
            let avg = averaging_type
                .parse::<AveragingType>()
                .expect("averaging type");
            let product =
                AsianOption::new(s, k, expiry, Some(valuation), option_type, qty, avg, None)
                    .expect("asian option");
            engine
                .calculate_price(&product, &model)
                .expect("asian price")
        } else {
            let product =
                EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                    .expect("european option");
            engine
                .calculate_price(&product, &model)
                .expect("european price")
        };

        let abs_diff = (rust_price - expected_price).abs();

        assert!(
            abs_diff < 1.0e-10,
            "{name}: monte carlo deterministic mismatch. rust={rust_price}, fixture={expected_price}, abs={abs_diff}"
        );
    }
}

#[test]
fn monte_carlo_heston_matches_fixture() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../QuantLibPY/tests/fixtures/monte_carlo_heston.csv");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|_| panic!("failed to read fixture file: {}", fixture_path.display()));

    for line in fixture_text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 20 {
            continue;
        }

        let name = fields[0];
        let product_type = fields[1];
        let averaging_type = fields[2];
        let s = parse_f64(fields[3]);
        let k = parse_f64(fields[4]);
        let expiry = parse_date(fields[5]);
        let valuation = parse_date(fields[6]);
        let option_type = fields[7].parse::<OptionType>().expect("option type");
        let qty = parse_i32(fields[8]);
        let kappa = parse_f64(fields[9]);
        let theta = parse_f64(fields[10]);
        let sigma = parse_f64(fields[11]);
        let rho = parse_f64(fields[12]);
        let v0 = parse_f64(fields[13]);
        let r = parse_f64(fields[14]);
        let q = parse_f64(fields[15]);
        let n_paths = parse_usize(fields[16]);
        let n_steps = parse_usize(fields[17]);
        let seed = parse_u64(fields[18]);
        let expected_price = parse_f64(fields[19]);

        let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).expect("heston model");
        let engine = MonteCarloEngine::new(n_paths, n_steps, Some(seed)).expect("engine");

        let rust_price = if product_type == "asian" {
            let avg = averaging_type
                .parse::<AveragingType>()
                .expect("averaging type");
            let product =
                AsianOption::new(s, k, expiry, Some(valuation), option_type, qty, avg, None)
                    .expect("asian option");
            engine
                .calculate_price(&product, &model)
                .expect("heston asian price")
        } else {
            let product =
                EuropeanOption::new_spot(s, k, expiry, Some(valuation), option_type, qty, None)
                    .expect("european option");
            engine
                .calculate_price(&product, &model)
                .expect("heston european price")
        };

        let abs_diff = (rust_price - expected_price).abs();
        let rel_diff = abs_diff / expected_price.abs().max(1.0);

        let is_stochastic_reference = name.contains("stochastic");
        let passes = if is_stochastic_reference {
            // The stochastic fixture was generated in Python with Mersenne Twister.
            // Rust uses XorShift here, so we compare statistical consistency.
            abs_diff < 0.6 || rel_diff < 0.08
        } else {
            abs_diff < 1.0e-10
        };

        assert!(
            passes,
            "{name}: heston mc mismatch. rust={rust_price}, fixture={expected_price}, abs={abs_diff}, rel={rel_diff}"
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

fn parse_u64(value: &str) -> u64 {
    value.parse::<u64>().expect("valid u64")
}
