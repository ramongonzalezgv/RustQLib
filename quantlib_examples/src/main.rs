use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

fn main() {
    let product = EuropeanOption::new_spot(
        100.0,
        100.0,
        SimpleDate::new(2027, 2, 10).expect("valid date"),
        Some(SimpleDate::new(2026, 2, 10).expect("valid date")),
        OptionType::Call,
        1,
        None,
    )
    .expect("valid product");

    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("valid model");
    let engine = AnalyticalEngine;

    let price = engine
        .calculate_price(&product, &model)
        .expect("successful pricing");

    println!("European call price: {price:.8}");
}
