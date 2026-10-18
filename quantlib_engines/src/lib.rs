pub mod analytical_engine;
pub mod batch;
pub mod binomial_engine;
pub mod cos_engine;
pub mod fft_engine;
mod fourier;
pub mod monte_carlo_engine;

pub use analytical_engine::{
    AnalyticalEngine, black_price, black_scholes_price, geometric_asian_black_scholes,
};
pub use batch::{calculate_prices_parallel, calculate_prices_parallel_per_model};
pub use binomial_engine::{BinomialEngine, BinomialLevels};
pub use cos_engine::CosEngine;
pub use fft_engine::{FFTEngine, InterpolationKind};
pub use monte_carlo_engine::{
    HestonScheme, McEstimate, MonteCarloConfig, MonteCarloEngine, MonteCarloPricing, SimulatedPaths,
};

// Unsupported product/model pairs are compile errors, not runtime "does not support" stubs (#57).
#[cfg(test)]
mod unsupported_pairs {
    use quantlib_core::PricingEngine;
    use quantlib_models::{BlackModel, BlackScholesModel, HestonModel, SabrModel};
    use quantlib_products::{AmericanOption, AsianOption, DigitalOption};
    use static_assertions::assert_not_impl_any;

    use crate::{AnalyticalEngine, MonteCarloEngine};

    assert_not_impl_any!(MonteCarloEngine: PricingEngine<AmericanOption, BlackScholesModel>, PricingEngine<AmericanOption, HestonModel>);
    assert_not_impl_any!(AnalyticalEngine: PricingEngine<DigitalOption, SabrModel>, PricingEngine<AmericanOption, BlackScholesModel>);
    assert_not_impl_any!(AnalyticalEngine: PricingEngine<AmericanOption, BlackModel>, PricingEngine<AsianOption, BlackModel>);
    assert_not_impl_any!(AnalyticalEngine: PricingEngine<AmericanOption, SabrModel>, PricingEngine<AsianOption, SabrModel>);
}
