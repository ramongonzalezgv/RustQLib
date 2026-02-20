//! Scratchpad for local interactive experiments.
//!
//! Usage examples:
//! - `cargo test -p quantlib_engines --test scratchpad -- --nocapture`
//! - `cargo test -p quantlib_engines --test scratchpad scratchpad -- --nocapture`

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{AnalyticalEngine, BinomialEngine, MonteCarloEngine};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AsianOption, AveragingType, EuropeanOption};

fn calculate_payoff(strike: f64, spot: f64) -> Result<f64, String> {
    if strike < 0.0 || spot < 0.0 {
        return Err("strike and spot must be non-negative".to_owned());
    }
    Ok((spot - strike).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratchpad() {
        let spot = 110.0;
        let strike = 100.0;

        let result = calculate_payoff(strike, spot).expect("payoff should compute");

        println!("----------------------------");
        println!("DEBUG PAYOFF: {}", result);
        println!("----------------------------");

        assert!(result > 0.0);
    }

    #[test]
    fn scratchpad_pricing_examples() {
        let option = EuropeanOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("valid date"),
            Some(SimpleDate::new(2026, 2, 10).expect("valid date")),
            OptionType::Call,
            1,
            None,
        )
        .expect("valid option");

        let bs_model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("valid BS model");
        let heston_model =
            HestonModel::new(2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.01).expect("valid Heston model");

        let analytical = AnalyticalEngine;
        let binomial = BinomialEngine::new(500);
        let mc_bs = MonteCarloEngine::new(100_000, 252, Some(42)).expect("MC engine");
        let mc_heston = MonteCarloEngine::new(100_000, 252, Some(42)).expect("MC engine");

        let p_analytical = analytical
            .calculate_price(&option, &bs_model)
            .expect("analytical price");
        let p_binomial = binomial
            .calculate_price(&option, &bs_model)
            .expect("binomial price");
        let p_mc_bs = mc_bs
            .calculate_price(&option, &bs_model)
            .expect("MC BS price");
        let p_mc_heston = mc_heston
            .calculate_price(&option, &heston_model)
            .expect("MC Heston price");

        println!("----------------------------");
        println!("ANALYTICAL BS: {}", p_analytical);
        println!("BINOMIAL BS:   {}", p_binomial);
        println!("MC BS:         {}", p_mc_bs);
        println!("MC HESTON:     {}", p_mc_heston);
        println!("----------------------------");

        // Keep broad so you can edit freely without constant failures.
        assert!(p_analytical.is_finite());
        assert!(p_binomial.is_finite());
        assert!(p_mc_bs.is_finite());
        assert!(p_mc_heston.is_finite());
    }

    #[test]
    fn scratchpad_asian_example() {
        let asian = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("valid date"),
            Some(SimpleDate::new(2026, 2, 10).expect("valid date")),
            OptionType::Call,
            1,
            AveragingType::Arithmetic,
            None,
        )
        .expect("valid Asian option");

        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("valid model");
        let mc = MonteCarloEngine::new(50_000, 252, Some(7)).expect("MC engine");

        let p = mc.calculate_price(&asian, &model).expect("Asian MC price");

        println!("----------------------------");
        println!("ASIAN MC PRICE: {}", p);
        println!("----------------------------");

        assert!(p.is_finite());
    }
}
