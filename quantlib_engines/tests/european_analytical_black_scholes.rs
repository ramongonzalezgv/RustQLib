use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::AnalyticalEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::EuropeanOption;

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
